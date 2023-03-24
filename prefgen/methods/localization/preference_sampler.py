"""
    This code implements the bayesian posterior sampling procedure
    for sampling from the user preference distribution over attribute
    space. 
"""

import numpy as np
import torch
import pymc as pm
import pymc.sampling_jax as sampling_jax

from prefgen.methods.sampling.utils import LatentAttributeSampler

class PreferenceSampler():
    """
        This class handles sampling from a distribution 
        p(a|Q) where `a` is a similarity embedding of attributes and 
        Q is a sequence of queries q_i = (a_p, a_n) where a_p is 
        preffered to a_n by the user. 
    """

    def __init__(
        self, 
        latent_sampler: LatentAttributeSampler,
        noise_constant=1.0, 
        noise_model="none", 
        information_decay=0.99, 
        normalization_factor=1.0, 
        linear_decay=False,
        linear_decay_rate=10.0, 
        prior_distribution="gaussian"
    ):
        self.latent_sampler = latent_sampler
        self.noise_constant = noise_constant
        self.noise_model = noise_model
        self.information_decay = information_decay
        self.normalization_factor = normalization_factor
        self.linear_decay = linear_decay
        self.linear_decay_rate = linear_decay_rate
        self.prior_distribution = prior_distribution
        # Up front sample latent distribution
        if self.prior_distribution == "gaussian":
            self.prior_mean, self.prior_cov = self.compute_gaussian_prior_parameters()

    def compute_gaussian_prior_parameters(self, num_samples=1000):
        """
            Samples using the latent sample to infer the parameters
            of the prior distribution.
        """
        samples = self.latent_sampler.randomly_sample_attributes(
            num_samples=num_samples,
            return_dict=False
        )
        samples = samples.detach().cpu().numpy()
        prior_mean = np.mean(samples, axis=0)
        prior_cov = np.cov(samples.T)

        return prior_mean, prior_cov

    def __call__(self, *args, **kwargs):
        """
            Calls sample 
        """
        return self.sample(*args, **kwargs)

    def process_noise_constant(self, positive, negative, num_queries=1):
        """Processes the noise constant based on the noise model"""
        if self.noise_model == "none" or self.noise_model == "constant":
            k = self.noise_constant
            return k * self.information_decay ** num_queries
        elif self.noise_model == "normalized":
            k = 1/2 * self.noise_constant * (
                1 / np.linalg.norm(positive - negative)
            )
            return k * self.information_decay ** num_queries
        elif self.noise_model == "decaying":
            k = self.noise_constant * np.exp(
                -2 * np.linalg.norm(positive - negative)
            )
            return k * self.information_decay ** num_queries
        else:
            raise Exception("Invalid noise model: {}".format(self.noise_model))

    def sample(
        self, 
        query_data, 
        num_samples=500, 
        z_dim=2,
        use_jax=False, 
        device="cuda"
    ):
        """
            Use MCMC with NUTS Sampling to sample form the logistic_response_model 
            posterior predictive distribution
        """
        assert len(query_data) > 0, "No queries to sample from"
        attributes_dim = query_data[0].positive_attribute.shape[0]
        # Unpack query_data
        positives = np.array([
            query.positive_attribute for query in query_data
        ]).squeeze()
        negatives = np.array([
            query.negative_attribute for query in query_data
        ]).squeeze()
        # Define the model
        with pm.Model() as self.logistic_response_model:
            num_queries = len(query_data)
            A_vals = 2 * (positives - negatives)
            tau_vals = []
            k_vals = []
            for query_index in range(num_queries):
                positive = positives[query_index]
                negative = negatives[query_index]
                # Offset
                tau_val = (
                    pm.math.dot(positive, positive) ** 2 - \
                    pm.math.dot(negative, negative) ** 2
                )
                tau_vals.append(tau_val)
                # Compute the noise constant
                k_val = self.process_noise_constant(positive, negative, num_queries=num_queries)
                # Add linear decay
                if self.linear_decay:
                    k_val = k_val * 1 / (self.linear_decay_rate * (num_queries + 1))
                k_vals.append(k_val)

            # Prior over latent space
            # Down scale covariance
            # scaled_cov = self.prior_cov * 1/5
            if self.prior_distribution == "gaussian":
                ideal_point = pm.MvNormal(
                    "ideal_point", 
                    mu=self.prior_mean, 
                    cov=self.prior_cov, 
                    shape=attributes_dim
                )
            else:
                ideal_point = pm.Uniform(
                    "ideal_point", 
                    lower=0.0, 
                    upper=1.0, 
                    shape=attributes_dim
                )
            # Compute the logits
            # log_scale = 1/num_queries
            logits = k_vals * (pm.math.dot(A_vals, ideal_point) - tau_vals)
            # logits = pm.math.sgn(logits)* pm.math.sqrt(pm.math.abs(logits))
            ones = np.ones((num_queries, 1))
            query_response_prob = pm.Bernoulli(
                "query_response_prob", 
                logit_p=logits, 
                observed=ones
            )
            # Decide whether or not to use JAX
            if use_jax: # JAX doesn't work rn
                samples = sampling_jax.sample_numpyro_nuts(
                    1000, 
                    tune=1000, 
                    # target_accept=0.9,
                )
                samples = samples.posterior.ideal_point.squeeze().to_numpy()
                samples = np.reshape(samples, (-1, samples.shape[-1]))
                # samples = samples.get_values("ideal_point")
                return samples
            else:
                try:
                    samples = pm.sample(
                        500, 
                        cores=4,
                        return_inferencedata=False
                    )
                    samples = samples.get_values("ideal_point")
                    # samples = samples[0] # TODO figure out what get_values outputs
                    return samples
                except Exception as e:
                    raise e


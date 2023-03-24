import numpy as np
from enum import Enum
import scipy.special as sc
import scipy as sp
import stan
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import torch
from PIL import Image
from numpy import asarray
import random
import os
import wandb
import multiprocessing

from prefgen.methods.sampling.utils import LatentAttributeSampler

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

"""
    Enum of noise constant types
"""
class KNormalizationType(Enum):
    CONSTANT = 0
    NORMALIZED = 1
    DECAYING = 2

"""
    converts pair to hyperplane weights and bias. 
"""
def pair2hyperplane(query_positive, query_negative, normalization, slice_point=None):
    A_emb = 2 * (query_positive - query_negative)
    if np.linalg.norm(A_emb) == 0:
        A_emb = np.ones_like(query_positive) * 0.000001

    if slice_point is None:
        tau_emb = (
            np.linalg.norm(query_positive) ** 2
            - np.linalg.norm(query_negative) ** 2
        )
    else:
        tau_emb = np.dot(A_emb, slice_point)

    if normalization == KNormalizationType.CONSTANT:
        pass
    elif normalization == KNormalizationType.NORMALIZED:
        A_mag = np.linalg.norm(A_emb)
        A_emb = A_emb / A_mag
        tau_emb = tau_emb / A_mag
    elif normalization == KNormalizationType.DECAYING:
        A_mag = np.linalg.norm(A_emb)
        A_emb = A_emb * np.exp(-A_mag)
        tau_emb = tau_emb * np.exp(-A_mag)
    return (A_emb, tau_emb)

####################################################### Stan Model ########################################################
traditional_model = """
    data {  
        int<lower=0> D;             // space dimension
        int<lower=0> M;             // number of measurements so far
        real k;                     // logistic noise parameter (scale)
        vector[2] bounds;           // hypercube bounds [lower,upper]
        int y[M];                   // measurement outcomes
        vector[D] A[M];             // hyperplane directions
        vector[M] tau;              // hyperplane offsets
    }
    parameters {
        vector<lower=bounds[1],upper=bounds[2]>[D] W;         // the user point
    }
    transformed parameters {
        vector[M] z;
        for (i in 1:M)
            z[i] = dot_product(A[i], W) - tau[i];
    }
    model {
        // prior
        W ~ uniform(bounds[1],bounds[2]);
        // linking observations
        y ~ bernoulli_logit(k * z);
    }
"""

class StanPreferenceSampler():
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
        prior_distribution="uniform",
        bounds=[0.0, 1.0],
    ):
        self.latent_sampler = latent_sampler
        self.noise_constant = noise_constant
        self.noise_model = noise_model
        self.normalization = {
            "none": KNormalizationType.CONSTANT,
            "constant": KNormalizationType.CONSTANT,
            "decaying": KNormalizationType.DECAYING,
            "normalized": KNormalizationType.NORMALIZED,
        }[self.noise_model]
        self.information_decay = information_decay
        self.normalization_factor = normalization_factor
        self.linear_decay = linear_decay
        self.linear_decay_rate = linear_decay_rate
        self.prior_distribution = prior_distribution
        assert prior_distribution == "uniform"
        self.bounds = bounds

    def __call__(self, *args, **kwargs):
        """
            Calls sample 
        """
        return self.sample(*args, **kwargs)

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
        # Make A matrix and Tau Matrix
        A_matrix = np.zeros((len(query_data), attributes_dim))
        tau_matrix = np.zeros(len(query_data))
        for index, query in enumerate(query_data):
            A_matrix[index, :], tau_matrix[index] = pair2hyperplane(
                query.positive_attribute, 
                query.negative_attribute,
                self.normalization
            )
    
        k = self.noise_constant
        data_gen = {
            'D': attributes_dim,
            'k': k,
            'M': len(A_matrix),
            'A': A_matrix,
            'tau': tau_matrix,
            'y': np.ones(len(A_matrix), dtype=np.int32),
            'bounds': self.bounds,
        }
        print("Building Stan Model")
        # with suppress_stdout_stderr():
        # num_samples = iter * chains / 2, unless warmup is changed
        posterior = stan.build(
            traditional_model,
            data=data_gen,
        )

        fit = posterior.sample(
            num_samples=1000,
            num_chains=4,
        )
        W_samples = fit['W'].T

        # Get the posterior samples
        if W_samples.ndim < 2:
            W_samples = W_samples[:, np.newaxis]

        return W_samples
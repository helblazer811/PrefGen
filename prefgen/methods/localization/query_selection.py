"""
    Tools for dynamically or randomly selecting paired queries.
"""

from abc import ABC, abstractmethod
from random import randint
from prefgen.methods.localization.utils import PairedQuery
import torch
import numpy as np

class QuerySelector(ABC):
    """
        Abstract class for selecting queries 
    """

    def __init__(
        self,
        generator,
        latent_sampler,
        num_samples=2000, 
        noise_constant=1.0,
        uniform_queries=False,
        value_range=(0.0, 1)
    ):
        self.generator = generator
        self.latent_sampler = latent_sampler
        self.noise_constant = noise_constant
        self.uniform_queries = uniform_queries
        self.value_range = value_range

        self.latents, self.attributes = self.generate_latent_attribute_pairs(
            num_samples=num_samples
        )
        if isinstance(self.latents, torch.Tensor):
            self.latents = self.latents.cpu().numpy()
        if isinstance(self.attributes, torch.Tensor):
            self.attributes = self.attributes.cpu().numpy()

    def generate_latent_attribute_pairs(self, num_samples=2000):
        """Generates a bunch of latent attribute pairs using the latent sampler"""
        latents, attributes = self.latent_sampler.generate_latent_attribute_pairs(
            num_samples
        )

        return latents, attributes

    def generate_random_query(self):
        """
            Generate a random query 
        """
        if self.uniform_queries:
            _, attributes = self.latent_sampler.generate_latent_attribute_pairs(
                1
            )
            attribute_dim = attributes[0].shape[-1]

            attribute_a = np.random.rand(attribute_dim)
            attribute_b = np.random.rand(attribute_dim)
            # Return the query
            paired_query = PairedQuery(
                answered=False,
                latent_vectors=(
                    None, 
                    None
                ),
                attribute_vectors=(
                    attribute_a, 
                    attribute_b
                ),
            )
            return paired_query
        else:
            # Select random indices from attribute
            while True:
                a_index = randint(0, len(self.latents) - 1)
                b_index = randint(0, len(self.latents) - 1)
                if a_index != b_index:
                    break
            # Return the query
            paired_query = PairedQuery(
                answered=False,
                latent_vectors=(
                    self.latents[a_index], 
                    self.latents[b_index]
                ),
                attribute_vectors=(
                    self.attributes[a_index], 
                    self.attributes[b_index]
                ),
            )

            return paired_query

    def __call__(self, preference_samples):
        """Callable wrapper for select_query"""
        query = self.select_query(preference_samples)
        return query

    @abstractmethod
    def select_query(self, preference_samples):
        """
            Sample andom query
        """
        raise Exception()

class RandomQuerySelector(QuerySelector):
    """
        Selects query randomly from attribute space.  
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_query(self, preference_samples):
        """
            Selects a random query 
        """
        random_query = self.generate_random_query()
        if self.uniform_queries:
            # If it is uniform then the latents will be None here
            # the purpose was to avoid doing latent sampling for all 
            # random queries for when the latent sampler is slow. 
            # 1. Choose a random latent
            random_latent = self.generator.randomly_sample_latent()
            # 2. Modify the latents to have the chosen attributes
            _, latent_a = self.latent_sampler.sample(
                random_query.attribute_vectors[0],
                initial_latent=random_latent,
                verbose_logging=False
            )
            _, latent_b = self.latent_sampler.sample(
                random_query.attribute_vectors[1],
                initial_latent=random_latent,
                verbose_logging=False
            )
            random_query.latent_vectors = (
                latent_a,
                latent_b
            )

        return random_query

class MCMVQuerySelector(QuerySelector):
    """
        Selects query using the MCMV strategy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_noise_constant(self, A_emb, tau_emb):
        """
            Compute a noise constant for the given query
        """
        # Normalized
        A_mag = np.linalg.norm(A_emb)
        A_emb = 1 / A_mag
        noise_constant = self.noise_constant * A_emb
        # Decaying 
        """
        A_mag = np.linalg.norm(A_emb)
        A_emb = A_emb * np.exp(-A_mag)
        tau_emb = tau_emb * np.exp(-A_mag)
        """
        return noise_constant

    def select_query(self, preference_samples, best_of=1000, lambda_pen_MCMV=1):
        """
            Selects a random query 
        """
        # Compute mean and covariance of preference_samples
        preference_covariance = np.cov(preference_samples, rowvar=False)
        preference_mean = np.mean(preference_samples, axis=0)
        # Select the best query of "best_of"
        best_mcmv_val = -1 * float("inf")
        best_query = None
        for query_num in range(best_of):
            # Select a random query
            random_query = self.generate_random_query()
            # Compute the hyperplane information
            A_emb, tau_emb = random_query.compute_hyperplane()
            A_emb = A_emb.detach().cpu().numpy()
            tau_emb = tau_emb.detach().cpu().numpy()
            # Compute the noise constant using the given noise model
            noise_constant = self.compute_noise_constant(A_emb, tau_emb)
            # Compute the MCMV values
            varest = np.dot(A_emb, preference_covariance).dot(A_emb)
            distmu = np.abs(
                (np.dot(A_emb, preference_mean) - tau_emb)
                / np.linalg.norm(A_emb)
            )
            mcmv_value = noise_constant * np.sqrt(varest) - lambda_pen_MCMV * distmu
            # See if mcmv_value is greater than best
            if mcmv_value > best_mcmv_val:
                best_mcmv_val = mcmv_value
                best_query = random_query
        if self.uniform_queries:
            # If it is uniform then the latents will be None here
            # the purpose was to avoid doing latent sampling for all 
            # random queries for when the latent sampler is slow. 
            # 1. Choose a random latent
            random_latent = self.generator.randomly_sample_latent()
            # 2. Modify the latents to have the chosen attributes
            _, latent_a = self.latent_sampler.sample(
                best_query.attribute_vectors[0],
                initial_latent=random_latent,
                verbose_logging=False
            )
            _, latent_b = self.latent_sampler.sample(
                best_query.attribute_vectors[1],
                initial_latent=random_latent,
                verbose_logging=False
            )
            best_query.latent_vectors = (
                latent_a,
                latent_b
            )

        return best_query

        
class ContinuousMCMVQuerySelector(QuerySelector):
    """
        Selects query using the Continuous variation of the MCMV strategy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_query(self, preference_samples):
        """
            Selects a random query 
        """
        # Compute the covariance
        cov = np.cov(preference_samples, rowvar=False)
        # Compute the axis of maximum variance (the eigenvector corresponding to the largest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        axis_of_max_variance = eigenvectors[np.argmax(eigenvalues)]
        # Compute the mean of the samples
        mean = np.mean(preference_samples, axis=0)
        # Sample two random points along the axis of max variance that are equidistance from the mean
        distance_scalar = 1.0
        point_1 = mean + distance_scalar * axis_of_max_variance
        point_2 = mean - distance_scalar * axis_of_max_variance
        # While either are out of range
        distance_scalar = 1.0
        while np.any(point_1 < self.value_range[0]) or np.any(point_1 > self.value_range[1]):
            print(point_1)
            distance_scalar *= 0.9
            point_1 = mean + distance_scalar * axis_of_max_variance

        distance_scalar = 1.0
        while np.any(point_2 < self.value_range[0]) or np.any(point_2 > self.value_range[1]):
            print(point_2)
            distance_scalar *= 0.9
            point_2 = mean - distance_scalar * axis_of_max_variance

        best_query = PairedQuery(
            answered=False,
            latent_vectors=(
                None, 
                None
            ),
            attribute_vectors=(
                point_1, 
                point_2
            ),
        )
        # 1. Choose a random latent
        random_latent = self.generator.randomly_sample_latent()
        # 2. Modify the latents to have the chosen attributes
        _, latent_a = self.latent_sampler.sample(
            best_query.attribute_vectors[0],
            initial_latent=random_latent,
            verbose_logging=False
        )
        _, latent_b = self.latent_sampler.sample(
            best_query.attribute_vectors[1],
            initial_latent=random_latent,
            verbose_logging=False
        )
        best_query.latent_vectors = (
            latent_a,
            latent_b
        )

        return best_query
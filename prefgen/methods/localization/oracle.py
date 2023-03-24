"""
    Tools for answering queries
"""
from prefgen.methods.localization.utils import PairedQuery
import torch
import numpy as np

def gaussian_noise_oracle(query, ideal_point, random_flip_chance=0.0, noise_scale=0.0):
    """
        This is a function that takes in a query and answers
        it after injecting some given amount of noise.  

        Note: query and ideal point are all in attribute space. 
    """
    attribute_a, attribute_b = query.attribute_vectors[0], query.attribute_vectors[1]
    latent_a, latent_b = query.latent_vectors[0], query.latent_vectors[1]
    a_dist = np.linalg.norm(attribute_a - ideal_point) ** 2 + np.random.normal(0, noise_scale)
    b_dist = np.linalg.norm(attribute_b - ideal_point) ** 2
    # Get the ground truth answer
    answer = b_dist < a_dist
    # randomly flip the answer with the given percentage chance
    random_float = np.random.random_sample()
    if random_float < random_flip_chance:
        answer = not answer

    if not answer:
        answered_query = PairedQuery(
            answered=True,
            latent_vectors=query.latent_vectors,
            attribute_vectors=query.attribute_vectors,
            positive_latent=latent_a,
            negative_latent=latent_b,
            positive_attribute=attribute_a,
            negative_attribute=attribute_b,
        )
    else:
        answered_query = PairedQuery(
            answered=True,
            latent_vectors=query.latent_vectors,
            attribute_vectors=query.attribute_vectors,
            positive_latent=latent_b,
            negative_latent=latent_a,
            positive_attribute=attribute_b,
            negative_attribute=attribute_a,
        )

    return answered_query

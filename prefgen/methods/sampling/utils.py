from abc import abstractclassmethod
from typing import Union
import torch

class LatentAttributeSampler(object):

    @abstractclassmethod
    def sample(self):
        """
            Generic latent attribute sampler method. 
        """
        pass

    @abstractclassmethod
    def generate_latent_attribute_pairs(self):
        """
            Generate a bunch of latent attribute pairs.

            This is used when sampling queries. 
        """
        pass

    @abstractclassmethod
    def randomly_sample_attributes(self, num_samples=None, normalize=True, return_dict=False):
        """
            Sample random attributes from the attribute space.
        """
        pass
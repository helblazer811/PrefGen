import torch
import numpy as np

class PairedQuery():
    """
        Data wrapper class for a paired comparison 
        query.  
    """

    def __init__(self, answered=False, latent_vectors=[], attribute_vectors=[],
                    positive_latent=None, negative_latent=None, positive_attribute=None,
                    negative_attribute=None):
        self.answered = answered
        self.latent_vectors = latent_vectors
        self.attribute_vectors = attribute_vectors
        self.positive_latent = positive_latent
        self.negative_latent = negative_latent
        self.positive_attribute = positive_attribute
        self.negative_attribute = negative_attribute

    def compute_hyperplane(self):
        """
            Compute the PairedQuery  
        """
        if self.positive_attribute is None:
            positive = self.attribute_vectors[0]
            negative = self.attribute_vectors[1]
        else:
            positive = self.positive_attribute
            negative = self.negative_attribute

        if isinstance(positive, np.ndarray):
            positive = torch.from_numpy(positive)
        if isinstance(negative, np.ndarray):
            negative = torch.from_numpy(negative)

        A_emb = 2 * (positive - negative)
        if torch.linalg.norm(A_emb) == 0:
            A_emb = torch.ones_like(positive)*0.000001

        tau_emb = (
            torch.linalg.norm(positive)**2 -
            torch.linalg.norm(negative)**2
        )

        return A_emb, tau_emb


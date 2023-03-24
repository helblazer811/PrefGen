"""
    Module for dataclass holding simulation data
"""
import pickle

class LocalizationSimulationData():
    """
        This is a dataclass containing all of the relevant
        information to understand a single run of a localization 
        simulation.  
    """

    def __init__(self):
        self.start_latent = []
        self.target_attributes = []
        self.attributes_over_time = []
        self.latents_over_time = []
        self.preference_samples_over_time = []
        self.queries = []
        self.start_image = None

    def save(self, save_path):
        """Save simulation"""
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, save_path):
        """Load the sim simulation"""
        with open(save_path, "rb") as f:
            localization_sim_data = pickle.load(f)
            return localization_sim_data
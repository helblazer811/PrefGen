"""
    This file handles running an instance of paired query localization
    with a simulated oracle. 
"""
from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.localization.stan_preference_sampler import StanPreferenceSampler
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler
from prefgen.methods.sampling.langevin_dynamics.energy_functions import compute_continuous_conditional_energy
from prefgen.methods.attribute_classifiers.face_identification.id_loss import IDLoss
from prefgen.methods.localization.oracle import gaussian_noise_oracle
from prefgen.methods.localization.query_selection import MCMVQuerySelector, RandomQuerySelector
from prefgen.methods.localization.simulation import LocalizationSimulator

import pickle
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np
import os
import wandb

PRETRAINED_PATH = os.path.join(os.environ["PREFGEN_ROOT"], 'prefgen/pretrained/')

def make_log_directory(save_dir="experiment_logs", base_name="experiment"):
    """
        Makes a save log directory with a name incremented so that it
        is unique. 
    """
    trial_num = 1
    while True:
        directory_name = os.path.join(save_dir, base_name+f"_{trial_num}")
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
            break
        else:
            trial_num += 1

    return directory_name

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser("Training Classifiers is All You Need")

    # parser.add_argument("--root_path", type=str, default=os.path.join(PRETRAINED_PATH, "dataset_styleflow"))
    parser.add_argument("--num_trials", default=1)
    parser.add_argument("--num_queries", default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attribute_names", type=list, default=["yaw", "pitch", "roll", "age"]) #"pitch", "light0", "light3", "age"])
    parser.add_argument("--target_attributes", default=torch.Tensor([0.2, 0.85]).cuda())
    parser.add_argument("--noise_constant", default=50.0)
    parser.add_argument("--information_decay", default=1.0)
    parser.add_argument("--query_selector", default="mcmv")
    parser.add_argument("--noise_model", default="constant")
    parser.add_argument("--linear_decay", default=False)
    parser.add_argument("--linear_decay_rate", default=0.0)
    parser.add_argument("--prior_distribution", default="uniform")
    parser.add_argument("--wandb_logging", default=True)
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    # LD sampling parameters
    parser.add_argument("--latent_sample_caching", default=True)
    parser.add_argument("--verbose_logging", default=False)
    parser.add_argument("--use_jax_sampling", default=False)
    parser.add_argument("--sample_distant_attribute", default=False)
    parser.add_argument("--wandb_group_name", default="GAN Control Localization FFHQ")
    parser.add_argument("--save_directory", default=None)

    args = parser.parse_args()

    # Run the experiment
    # Make StyleGAN Generator
    print("Loading StyleGAN")
    generator = StyleGAN2GANControlWrapper(
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent,
    )
    # Make sampler
    latent_sampler = GANControlSampler(
        generator,
        attribute_names=args.attribute_names
    )
    # Load up the localization simulator 
    localization_simulater = LocalizationSimulator(
        generator=generator,
        attribute_classifier=None,
        query_selector=None,
        query_oracle=gaussian_noise_oracle,
        preference_sampler=None,
        latent_attribute_sampler=latent_sampler,
        num_attributes=len(args.attribute_names)
    )
    trial_save_dir = "experiment_logs/experiment_16/trial_1"
    data_path = os.path.join(trial_save_dir, f"simulation_data.pkl")
    # Load the data
    with open(data_path, "rb") as f:
        simulation_data = pickle.load(f)
    # Load data
    localization_simulater.run_final_localization_plotting(
        simulation_data=simulation_data,
        num_queries=args.num_queries,
        save_directory=trial_save_dir,
    )


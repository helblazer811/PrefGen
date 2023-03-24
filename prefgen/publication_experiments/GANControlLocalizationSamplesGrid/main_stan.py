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
    parser.add_argument("--num_trials", default=5)
    parser.add_argument("--num_queries", default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attribute_names", type=list, default=["yaw", "pitch", "roll", "age"]) #"pitch", "light0", "light3", "age"])
    parser.add_argument("--target_attributes", default=None)
    parser.add_argument("--noise_constant", default=10.0)
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
    # Make the preference samplers
    print("Loading preference sampler")
    # Make the preference sampler
    preference_sampler = StanPreferenceSampler(
        latent_sampler=latent_sampler,
        noise_constant=args.noise_constant,
        noise_model=args.noise_model,
        information_decay=args.information_decay,
        prior_distribution=args.prior_distribution,
        linear_decay=args.linear_decay,
        linear_decay_rate=args.linear_decay_rate,
    )
    # Make query selector
    if args.query_selector == "random":
        query_selector = RandomQuerySelector(
            generator,
            latent_sampler=latent_sampler
        )
    elif args.query_selector == "mcmv":
        query_selector = MCMVQuerySelector(
            generator,
            latent_sampler=latent_sampler,
            noise_constant=args.noise_constant
        )
    else:
        raise Exception(f"Unrecognized query selector: {args.query_selector}")
    # Load up the localization simulator 
    localization_simulater = LocalizationSimulator(
        generator=generator,
        attribute_classifier=None,
        query_selector=query_selector,
        query_oracle=gaussian_noise_oracle,
        preference_sampler=preference_sampler,
        latent_attribute_sampler=latent_sampler,
        num_attributes=len(args.attribute_names)
    )
    # Make the save directory for the run
    if not args.save_directory is None:
        save_directory = args.save_directory
    else:
        save_directory = make_log_directory(
            save_dir="experiment_logs", 
            base_name="experiment"
        )
    # Save the args namespace 
    with open(os.path.join(save_directory, "args_namespace.pkl"), "wb") as f:
        pickle.dump(args, f)
    with open(os.path.join(save_directory, "args_namespace.json"), "w") as f:
        if not args.target_attributes is None:
            args.target_attributes = args.target_attributes.tolist()
        f.write(json.dumps(vars(args), indent=4))
        if not args.target_attributes is None:
            args.target_attributes = torch.Tensor(args.target_attributes)
    # Run each trial
    torch.manual_seed(6)
    for trial_number in range(args.num_trials):
        # Start wandb logging
        if args.wandb_logging:
            run = wandb.init(
                project="GanSearch",
                group=args.wandb_group_name,
                config=args
            )
        # Run through several trials
        trial_save_dir = os.path.join(save_directory, f"trial_{trial_number}")
        os.mkdir(trial_save_dir)
        # Sample a fixed start latent
        start_latent = generator.randomly_sample_latent().to(args.device)
        # Run the simulator
        localization_data = localization_simulater.run_localization_simulation(
            target_attributes=args.target_attributes,
            start_latent=start_latent,
            num_queries=args.num_queries,
            save_directory=trial_save_dir,
            latent_sample_caching=args.latent_sample_caching,
            use_jax_sampling=args.use_jax_sampling,
            sample_distant_attribute=args.sample_distant_attribute,
        )
        # Save the localization data
        localization_data.save(
            os.path.join(trial_save_dir, f"simulation_data.pkl")
        )
        # Wandb finish
        if args.wandb_logging:
            wandb.finish()

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.localization.preference_sampler import PreferenceSampler
from prefgen.methods.localization.stan_preference_sampler import StanPreferenceSampler
from prefgen.methods.localization.simulation_data import LocalizationSimulationData
import prefgen.methods.datasets.lstm as lstm_dataset
from prefgen.methods.localization.utils import PairedQuery
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler

def evaluate_random_mcmc_performance(
    preference_sampler: PreferenceSampler,
    test_dataset, 
    num_examples, 
    num_queries, 
    query_testing_interval,
    num_attributes=4
):
    save_data = {}
    # For each number of query intervals
    for num_queries in range(query_testing_interval, num_queries + 1, query_testing_interval):
        num_queries_data = []
        # For each example in the first `num_examples` examples
        for example_idx in tqdm(range(num_examples)):
            # Select the queries [0: num_queries]
            constraint_attributes = test_dataset[example_idx][-1]
            constraint_attributes = constraint_attributes[:num_queries]
            # Predict using Random the posterior estimate
            simulation_data = LocalizationSimulationData() # Initially empty
            simulation_data.queries = [ 
                PairedQuery(
                    answered=True,
                    attribute_vectors=(
                        individual_constraints[:num_attributes], 
                        individual_constraints[num_attributes:]
                    ),
                    positive_attribute=individual_constraints[:num_attributes],
                    negative_attribute=individual_constraints[num_attributes:],
                )
                for individual_constraints in constraint_attributes
            ]
            # Sample from preference posterior
            preference_samples = preference_sampler(
                simulation_data.queries, 
            )
            preference_samples = np.array(preference_samples)
            simulation_data.preference_samples_over_time.append(
                preference_samples
            )
            # Compute the mean of this preference distribution
            preference_estimate = np.mean(preference_samples, axis=0)
            # Save the example index, the number of queries used, and the final estimate
            num_queries_data.append({
                "example_idx": example_idx,
                "preference_estimate": preference_estimate
            })
        save_data[num_queries] = num_queries_data

    return save_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attribute_names",
        default=["yaw", "age"]
    )
    parser.add_argument(
        "--data_save_path", 
        type=str, 
        default="data/random_mcmc_data_20q.pkl"
    )
    parser.add_argument(
        "--num_examples",
        default=20
    )
    parser.add_argument(
        "--num_queries",
        default=30
    )
    parser.add_argument(
        "--query_testing_interval",
        default=1
    )

    parser.add_argument(
        "--test_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/test_dataset_gan_control.pkl"
        ),
    )
    
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--noise_constant", default=10.0)
    parser.add_argument("--information_decay", default=0.0)
    parser.add_argument("--linear_decay", default=False)
    parser.add_argument("--linear_decay_rate", default=1.0)
    parser.add_argument("--noise_model", default="constant")
    parser.add_argument("--prior_distribution", default="uniform")

    args = parser.parse_args()
    # Make the dataset
    test_dataset = lstm_dataset.ConstraintDataset(args.test_save_path)
    # Make StyleGAN Generator
    stylegan_generator = StyleGAN2GANControlWrapper(
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent,
    )
    latent_sampler = GANControlSampler(
        stylegan_generator,
        attribute_names=args.attribute_names,
    )
    # Load up the prefernece sampler 
    """
    preference_sampler = PreferenceSampler(
        latent_sampler=latent_sampler,
        noise_constant=args.noise_constant,
        noise_model=args.noise_model,
        information_decay=args.information_decay,
        linear_decay=args.linear_decay,
        linear_decay_rate=args.linear_decay_rate,
        prior_distribution=args.prior_distribution,
    )
    """
    preference_sampler = StanPreferenceSampler(
        latent_sampler=latent_sampler,
        noise_constant=args.noise_constant,
        noise_model=args.noise_model,
        information_decay=args.information_decay,
        prior_distribution=args.prior_distribution,
        linear_decay=args.linear_decay,
        linear_decay_rate=args.linear_decay_rate,
    )
    save_data = evaluate_random_mcmc_performance(
        preference_sampler, 
        test_dataset, 
        args.num_examples, 
        args.num_queries, 
        args.query_testing_interval,
        num_attributes=2
    )

    with open(args.data_save_path, "wb") as f:
        pickle.dump(save_data, f)
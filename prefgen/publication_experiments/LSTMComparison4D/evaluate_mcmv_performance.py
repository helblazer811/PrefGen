import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from prefgen.methods.datasets.lstm import ConstraintDataset
from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.localization.oracle import gaussian_noise_oracle
from prefgen.methods.localization.preference_sampler import PreferenceSampler
from prefgen.methods.localization.query_selection import MCMVQuerySelector
from prefgen.methods.localization.simulation import LocalizationSimulator
from prefgen.methods.localization.simulation_data import LocalizationSimulationData
from prefgen.methods.localization.stan_preference_sampler import StanPreferenceSampler
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler
from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.localization.utils import PairedQuery
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier

def evaluate_mcmv_performance(
    generator: StyleGAN2Wrapper,
    classifier, 
    preference_sampler: PreferenceSampler,
    query_selector: MCMVQuerySelector,
    oracle,
    latent_sampler,
    test_dataset, 
    num_examples, 
    num_queries, 
    query_testing_interval,
    experiment_logs_path="logs",
    num_attributes=2,
):
    save_data = {}
    # For each number of query intervals
    for num_queries in tqdm(range(query_testing_interval, num_queries + 1, query_testing_interval)):
        print(f"Evaluating num queries {num_queries}...")
        num_queries_data = []
        # For each example in the first `num_examples` examples
        for example_idx in tqdm(range(num_examples)):
            # Get the target attributes
            target_attributes = test_dataset[example_idx][1]
            # Initialize uniform random preference samples. 
            preference_samples = latent_sampler.randomly_sample_attributes(
                num_samples=1000,
                return_dict=False
            ).detach().cpu().numpy()
            # Go through the given number of queries
            queries = []
            for query_index in range(num_queries):
                # Sample a constraint using the preference sampler
                query = query_selector.select_query(
                    preference_samples
                )
                # Answer the query with the oracle
                answered_query = gaussian_noise_oracle(
                    query,
                    target_attributes,
                    random_flip_chance=0.0
                )
                # Predict using Random the posterior estimate
                queries.append(
                    answered_query
                )
                # Sample from preference posterior
                failure_counter = 0
                while failure_counter < 20:
                    try:
                        preference_samples = preference_sampler(
                            queries, 
                        )
                        preference_samples = np.array(preference_samples)
                        break
                    except:
                        failure_counter += 1
                preference_samples = np.array(preference_samples)
            # Save the most recent preference samples mean
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
        default=["yaw", "pitch", "roll", "age"]
    )
    parser.add_argument(
        "--mcmv_data_path", 
        type=str, 
        default="data/mcmv_data_4d.pkl"
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
            "prefgen/data/lstm_dataset/test_dataset_gan_control_4d.pkl"
        ),
    )
    
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--noise_constant", default=10.0)
    parser.add_argument("--information_decay", default=1.0)
    parser.add_argument("--linear_decay", default=False)
    parser.add_argument("--linear_decay_rate", default=100.0)
    parser.add_argument("--num_query_samples", default=1024.0)
    parser.add_argument("--noise_model", default="constant")
    parser.add_argument("--prior_distribution", default="uniform")
    parser.add_argument("--experiment_logs", default="logs")

    args = parser.parse_args()
    # Make the dataset
    test_dataset = ConstraintDataset(args.test_save_path)
    # Make StyleGAN Generator
    stylegan_generator = StyleGAN2GANControlWrapper(
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent,
    )
    latent_sampler = GANControlSampler(
        stylegan_generator,
        attribute_names=args.attribute_names,
    )
    latents, attributes = latent_sampler.generate_latent_attribute_pairs(
        num_examples=1000
    )
    # Load up the prefernece sampler 
    """
    preference_sampler = PreferenceSampler(
        prior_mean=None, 
        prior_cov=None,
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
    query_selector = MCMVQuerySelector(
        stylegan_generator,
        latent_sampler,
        noise_constant=args.noise_constant,
        num_samples=args.num_query_samples,
    )
    print("Evaluation MCMV Performance")
    save_data = evaluate_mcmv_performance(
        stylegan_generator,
        None,
        preference_sampler, 
        query_selector,
        gaussian_noise_oracle,
        latent_sampler,
        test_dataset, 
        args.num_examples, 
        args.num_queries, 
        args.query_testing_interval,
        experiment_logs_path=args.experiment_logs,
        num_attributes=4
    )

    with open(args.mcmv_data_path, "wb") as f:
        pickle.dump(save_data, f)
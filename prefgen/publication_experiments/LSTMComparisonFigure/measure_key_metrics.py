import argparse
import os
import pickle
from tabulate import tabulate
import torch
import numpy as np
from prefgen.methods.attribute_classifiers.face_identification.id_loss import IDLoss
from prefgen.methods.datasets.lstm import ConstraintDataset

from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.plotting.localization import compute_attribute_rank_loss
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler

def compute_key_metrics(generator, id_loss, latent_sampler: GANControlSampler, method_data, test_dataset):

    def compute_identity_loss(estimate_image, target_image):
        return id_loss(estimate_image.cuda(), target_image.cuda()).detach().cpu().numpy()

    def compute_preference_mse_loss(estimate_attributes, target_attributes):
        return np.linalg.norm(
            estimate_attributes - target_attributes
        ) ** 2

    def compute_percentile_mse_loss(estimate_attributes, target_attributes):
        rank_loss = compute_attribute_rank_loss(
            latent_sampler,
            estimate_attributes,
            target_attributes,
            num_samples=5000, 
            attribute_samples=None
        )
        return rank_loss

    def compute_percentage_of_constraints_satisfied(estimate_attributes, constraints):
        constraint_attributes = constraints
        # Pull out the constraint positive and negatives
        attribute_dim = int(constraint_attributes.shape[-1] / 2)
        constraint_positive = constraint_attributes[:, 0:attribute_dim]
        constraint_negative = constraint_attributes[:, attribute_dim:]
        # assert constraint_positive.shape == constraint_negative.shape
        # assert preference_estimate.shape == constraint_negative.shape, (preference_estimate.shape, constraint_negative.shape)
        # Compute the distances
        anchor_positive_dist = np.linalg.norm(
            preference_estimate - constraint_positive, 
            axis=-1
        )
        anchor_negative_dist = np.linalg.norm(
            preference_estimate - constraint_negative, 
            axis=-1
        )
        constraint_bool = (anchor_positive_dist < anchor_negative_dist)
        return np.sum(constraint_bool) / len(constraint_bool)

    percentile_mse_losses = []
    preference_mse_losses = []
    identity_losses = []
    percent_constraints_satisfied = []

    for data in method_data:
        preference_estimate = data["preference_estimate"]
        example_index = data["example_idx"]
        target_latent, target_attributes, constraints, initial_latent = test_dataset[example_index]
        # Generate the initial image
        _, initial_image = generator.generate_image(latent=initial_latent)
        # Generate the estimate image
        estimate_image, _, _, _ = latent_sampler.sample(
            target_attributes=target_attributes,
            initial_latent=initial_latent,
        )
        # Compute identity loss
        identity_losses.append(
            compute_identity_loss(
                estimate_image=estimate_image,
                target_image=initial_image
            )
        )
        # Compute preference mse loss
        preference_mse_losses.append(
            compute_preference_mse_loss(
                estimate_attributes=preference_estimate,
                target_attributes=target_attributes
            )
        )
        # Compute percentile mse loss
        percentile_mse_losses.append(
            compute_percentile_mse_loss(
                estimate_attributes=preference_estimate,
                target_attributes=target_attributes
            )
        )
        # Compute percentage of constraints satisfied
        percent_constraints_satisfied.append(
            compute_percentage_of_constraints_satisfied(
                estimate_attributes=preference_estimate,
                constraints=constraints
            )
        )

    # Compute mean and standard deviation loss
    data_metrics = {
        "percentile_mse": (np.mean(percentile_mse_losses), np.std(percentile_mse_losses)),
        "preference_mse": (np.mean(preference_mse_losses), np.std(preference_mse_losses)),
        "identity_loss": (np.mean(identity_losses), np.std(identity_losses)),
        "percentage_constraints_satisfied": (np.mean(percent_constraints_satisfied), np.std(percent_constraints_satisfied)),
    }

    return data_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--attribute_names", default=["yaw", "age"])
    parser.add_argument(
        "--random_mcmc_data_path", 
        type=str, 
        default="data/random_mcmc_data_20q.pkl"
    )
    parser.add_argument(
        "--lstm_data_path", 
        type=str, 
        default="data/lstm_data_20q.pkl"
    )
    parser.add_argument(
        "--mcmv_data_path", 
        type=str, 
        default="data/mcmv_data_20q.pkl"
    )
    parser.add_argument(
        "--closed_form_mcmv_data_path",
        type=str,
        default="data/continuous_mcmv_data_20q.pkl"
    )
    parser.add_argument(
        "--test_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/test_dataset_gan_control.pkl"
        ),
    )

    args = parser.parse_args()
    # Load the lstm data and the 
    print("Loading Data...")
    random_mcmc_data = pickle.load(open(args.random_mcmc_data_path, "rb"))
    lstm_data = pickle.load(open(args.lstm_data_path, "rb"))
    mcmv_data = pickle.load(open(args.mcmv_data_path, "rb"))
    closed_form_mcmv_data = pickle.load(open(args.closed_form_mcmv_data_path, "rb"))

    # Make StyleGAN Generator
    stylegan_generator = StyleGAN2GANControlWrapper(
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent,
    )
    latent_sampler = GANControlSampler(
        stylegan_generator,
        attribute_names=args.attribute_names,
    )
    test_dataset = ConstraintDataset(
        args.test_save_path,
        with_initial_latent=True,
    )
    id_loss = IDLoss()

    methods_dict = {
        "LSTM": lstm_data[30],
        "MCMV Closed Form": closed_form_mcmv_data[30],
        "MCMV Best of N": mcmv_data[30],
        "Random Selection": random_mcmc_data[30],
    }
    # Compute each of the key metrics
    table_data = []
    for method_name, method_data in methods_dict.items():
        print(f"Method data: {method_data}")
        print("Computing metric for {}...".format(method_name))
        metrics = compute_key_metrics(
            stylegan_generator,
            id_loss,
            latent_sampler,
            method_data,
            test_dataset
        )

        table_data.append([
            method_name, 
            metrics["identity_loss"],
            metrics["preference_mse"],
            metrics["percentile_mse"],
            metrics["percentage_constraints_satisfied"]
        ])
    # print out a table of key metrics
    table_string = tabulate(
        table_data, 
        headers=['Method', 'ID Loss', 'Preference MSE Loss', 'Percentile MSE Loss', 'Percent Constraints Satisfied'],
        tablefmt='orgtbl'
    )
    print(table_string)
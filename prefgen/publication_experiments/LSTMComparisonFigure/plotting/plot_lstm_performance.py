import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
import os
import torch

from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.plotting.localization import compute_attribute_rank_loss
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier
from prefgen.methods.datasets.lstm import ConstraintDataset
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler

# One line is the performance of the LSTM baseline
# The other line is the random mcmc baseline
def average_distance_in_attribute_space(
    latent_sampler,
    num_points=2000
):
    points = latent_sampler.randomly_sample_attributes(
        num_samples=2000,
        return_dict=False
    ).detach().cpu().numpy()
    
    total_point_distance = 0.0
    for point_index in range(num_points):
        # Choose another random point from points
        other_point_index = np.random.randint(0, num_points)
        # Compute the distance between the two points
        distance = np.linalg.norm(points[point_index] - points[other_point_index])
        # Add the distance to the total distance
        total_point_distance += distance

    # Return the average distance
    return total_point_distance / num_points

def plot_performance(
    latent_sampler,
    lstm_data, 
    average_dist,
    test_dataset, 
    figure_save_path
):
    # Make a matplotlib line plot with two lines
    fig = plt.figure()
    ax = fig.add_subplot(111)

    averages = []
    vars = []
    # For each num queries in each dataset
    for num_queries in lstm_data.keys():
        data_for_num_queries = lstm_data[num_queries]
        percentiles = []
        for _, eval_run_data in enumerate(data_for_num_queries):
            # Get the example target attributes
            example_index = eval_run_data["example_idx"]
            target_attributes = test_dataset[example_index][1]
            preference_estimate = eval_run_data["preference_estimate"].squeeze()
            # Get the difference between the target attributes and the lstm estimate
            print()
            print(preference_estimate)
            print(target_attributes)
            print()
            if isinstance(preference_estimate, torch.Tensor):
                preference_estimate = preference_estimate.detach().cpu().numpy()
            percentile = compute_attribute_rank_loss(
                latent_sampler=latent_sampler,
                current_attributes=preference_estimate,
                target_attributes=target_attributes,
                num_samples=500
            )
            # random_mcmc_diff = np.linalg.norm(target_attributes - random_mcmc_data_instance)
            # random_mcmc_diffs.append(random_mcmc_diff)
            percentiles.append(percentile)
        averages.append(np.mean(percentiles))
        vars.append(np.var(percentiles))

    ax.plot(
        list(lstm_data.keys()), 
        averages, 
        label="LSTM Baseline"
    )
    ax.fill_between(
        list(lstm_data.keys()), 
        np.array(averages) + np.array(vars), 
        np.array(averages) - np.array(vars), 
        alpha=0.5
    )
    # plot a flat line
    """
    ax.plot(
        list(random_mcmc_data.keys()), 
        [average_dist for _ in range(len(random_mcmc_data.keys()))], 
        label="Average Distance in Attribute Space"
    )
    """
    plt.xlabel("Number of Queries")
    plt.ylabel("Percentage Closer to Ground Truth (%)")
    plt.legend()
    plt.ylim(bottom=0.0, top=100.0)
    plt.xticks(list(lstm_data.keys()))
    plt.suptitle("Performance of Preference Estimation")
    plt.title("LSTM Baseline Performance")
    plt.savefig(figure_save_path, dpi=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--attribute_names", default=["yaw", "age"])
    parser.add_argument(
        "--lstm_data_path", 
        type=str, 
        default="data/lstm_data.pkl"
    )
    parser.add_argument(
        "--plot_path",
        default="plots/lstm_performance.png"
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
    test_dataset = ConstraintDataset(args.test_save_path)
    lstm_data = pickle.load(open(args.lstm_data_path, "rb"))

    # Make StyleGAN Generator
    stylegan_generator = StyleGAN2GANControlWrapper(
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent,
    )

    latent_sampler = GANControlSampler(
        stylegan_generator,
        attribute_names=args.attribute_names,
    )

    average_dist = average_distance_in_attribute_space(
        latent_sampler,
    )

    plot_performance(
        latent_sampler,
        lstm_data, 
        average_dist,
        test_dataset, 
        args.plot_path
    )

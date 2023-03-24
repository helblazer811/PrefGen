import matplotlib.pyplot as plt
import argparse
import pickle
from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier
from prefgen.methods.datasets.lstm import ConstraintDataset
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler
import numpy as np
import os

# One line is the performance of the LSTM baseline
# The other line is the random mcmc baseline
def average_distance_in_attribute_space(
    latent_sampler,
    num_samples=2000
):
    points = latent_sampler.randomly_sample_attributes(
        num_samples=num_samples,
        return_dict=False
    ).detach().cpu().numpy()
    total_point_distance = 0.0
    for point_index in range(num_samples):
        # Choose another random point from points
        other_point_index = np.random.randint(0, num_samples)
        # Compute the distance between the two points
        distance = np.linalg.norm(points[point_index] - points[other_point_index])
        # Add the distance to the total distance
        total_point_distance += distance

    # Return the average distance
    return total_point_distance / num_samples

def plot_performance(
    random_mcmc_data, 
    average_dist,
    test_dataset, 
    figure_save_path
):
    # Make a matplotlib line plot with two lines
    fig = plt.figure()
    ax = fig.add_subplot(111)

    random_mcmc_averages = []
    random_mcmc_vars = []
    # For each num queries in each dataset
    for num_queries in random_mcmc_data.keys():
        random_mcmc_data_for_num_queries = random_mcmc_data[num_queries]
        random_mcmc_diffs = []
        for example_index in range(len(random_mcmc_data_for_num_queries)):
            # Get the example target attributes
            target_attributes = test_dataset[example_index][1]
            # Get the difference between the target attributes and the lstm estimate
            random_mcmc_data_instance = random_mcmc_data_for_num_queries[example_index]["preference_estimate"]
            random_mcmc_diff = np.linalg.norm(target_attributes - random_mcmc_data_instance)
            random_mcmc_diffs.append(random_mcmc_diff)

        random_mcmc_averages.append(np.mean(random_mcmc_diffs))
        random_mcmc_vars.append(np.var(random_mcmc_diffs))

    ax.plot(
        list(random_mcmc_data.keys()), 
        random_mcmc_averages, 
        label="Random MCMC Baseline"
    )
    ax.fill_between(
        list(random_mcmc_data.keys()), 
        np.array(random_mcmc_averages) + np.array(random_mcmc_vars), 
        np.array(random_mcmc_averages) - np.array(random_mcmc_vars), 
        alpha=0.5
    )
    # plot a flat line
    ax.plot(
        list(random_mcmc_data.keys()), 
        [average_dist for _ in range(len(random_mcmc_data.keys()))], 
        label="Average Distance in Attribute Space"
    )
    plt.xlabel("Number of Queries")
    plt.ylabel("Preference Estimate Distance to Ground Truth")
    plt.legend()
    plt.ylim(bottom=0.0)
    plt.xticks(list(random_mcmc_data.keys()))
    plt.suptitle("Performance of Preference Estimation")
    plt.title("Random MCMC Performance")
    plt.savefig(figure_save_path, dpi=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--attribute_names", default=["yaw", "age"])
    parser.add_argument(
        "--random_mcmc_data_path", 
        type=str, 
        default="data/random_mcmc_data.pkl"
    )
    parser.add_argument(
        "--plot_path",
        default="plots/random_mcmc_performance.png"
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
    random_mcmc_data = pickle.load(open(args.random_mcmc_data_path, "rb"))

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
        random_mcmc_data, 
        average_dist,
        test_dataset, 
        args.plot_path
    )

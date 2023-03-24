import matplotlib.pyplot as plt
import argparse
import pickle
from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier
from prefgen.methods.datasets.lstm import ConstraintDataset
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

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

def plot_performance_comparison(
    random_mcmc_data, 
    lstm_data, 
    mcmv_data,
    closed_form_mcmv_data,
    average_dist,
    test_dataset, 
    figure_save_path
):
    lstm_averages = []
    lstm_vars = []
    random_mcmc_averages = []
    random_mcmc_vars = []
    mcmv_averages = []
    mcmv_vars = []
    closed_form_mcmv_averages = []
    closed_form_mcmv_vars = []
    # For each num queries in each dataset
    for num_queries in tqdm(lstm_data.keys()):
        lstm_data_for_num_queries = lstm_data[num_queries]
        random_mcmc_data_for_num_queries = random_mcmc_data[num_queries]
        mcmv_data_for_num_queries = mcmv_data[num_queries]
        closed_form_mcmv_data_for_num_queries = closed_form_mcmv_data[num_queries]
        lstm_diffs = []
        random_mcmc_diffs = []
        mcmv_diffs = []
        closed_form_mcmv_diffs = []
        for example_index in tqdm(range(len(lstm_data_for_num_queries))):
            # Get the example target attributes
            target_attributes = test_dataset[example_index][1]
            # Get the difference between the target attributes and the lstm estimate
            lstm_data_instance = lstm_data_for_num_queries[example_index]["preference_estimate"]
            lstm_diff = np.linalg.norm(target_attributes - lstm_data_instance) ** 2
            random_mcmc_data_instance = random_mcmc_data_for_num_queries[example_index]["preference_estimate"]
            random_mcmc_diff = np.linalg.norm(target_attributes - random_mcmc_data_instance) ** 2
            mcmv_data_instance = mcmv_data_for_num_queries[example_index]["preference_estimate"]
            mcmv_data_diff = np.linalg.norm(target_attributes - mcmv_data_instance) ** 2
            closed_form_data_instance = closed_form_mcmv_data_for_num_queries[example_index]["preference_estimate"]
            closed_form_mcmv_diff = np.linalg.norm(target_attributes - closed_form_data_instance) ** 2

            lstm_diffs.append(lstm_diff)
            random_mcmc_diffs.append(random_mcmc_diff)
            mcmv_diffs.append(mcmv_data_diff)
            closed_form_mcmv_diffs.append(closed_form_mcmv_diff)

        lstm_averages.append(np.mean(lstm_diffs))
        random_mcmc_averages.append(np.mean(random_mcmc_diffs))
        mcmv_averages.append(np.mean(mcmv_diffs))
        closed_form_mcmv_averages.append(np.mean(closed_form_mcmv_diffs))
        num_examples = len(lstm_data_for_num_queries)
        lstm_vars.append(np.std(lstm_diffs) / np.sqrt(num_examples))
        random_mcmc_vars.append(np.std(random_mcmc_diffs) / np.sqrt(num_examples))
        mcmv_vars.append(np.std(mcmv_diffs) / np.sqrt(num_examples))
        closed_form_mcmv_vars.append(np.std(closed_form_mcmv_diffs) / np.sqrt(num_examples))

    # Make a matplotlib line plot with two lines
    fig = plt.figure(figsize=(5*0.85, 3.3))
    ax = fig.add_subplot(111)

    ax.plot(
        list(lstm_data.keys()), 
        lstm_averages, 
        label="LSTM Baseline"
    )
    # Plot variance error bars
    ax.fill_between(
        list(lstm_data.keys()), 
        np.array(lstm_averages) + np.array(lstm_vars), 
        np.array(lstm_averages) - np.array(lstm_vars), 
        alpha=0.5
    )
    ax.plot(
        list(lstm_data.keys()), 
        random_mcmc_averages, 
        label="Ours Random Querying"
    )
    ax.fill_between(
        list(lstm_data.keys()), 
        np.array(random_mcmc_averages) + np.array(random_mcmc_vars), 
        np.array(random_mcmc_averages) - np.array(random_mcmc_vars), 
        alpha=0.5
    )
    ax.plot(
        list(mcmv_data.keys()), 
        mcmv_averages, 
        label="Ours Active Querying (Best of N)"
    )
    # Plot variance error bars
    ax.fill_between(
        list(mcmv_data.keys()), 
        np.array(mcmv_averages) + np.array(mcmv_vars), 
        np.array(mcmv_averages) - np.array(mcmv_vars), 
        alpha=0.5
    )

    ax.plot(
        list(closed_form_mcmv_data.keys()), 
        closed_form_mcmv_averages, 
        label="Ours Active Querying (Closed Form)"
    )
    # Plot variance error bars
    ax.fill_between(
        list(closed_form_mcmv_data.keys()), 
        np.array(closed_form_mcmv_averages) + np.array(closed_form_mcmv_vars), 
        np.array(closed_form_mcmv_averages) - np.array(closed_form_mcmv_vars), 
        alpha=0.5
    )
    # plot a flat line
    """
    ax.plot(
        list(lstm_data.keys()), 
        [average_dist for _ in range(len(lstm_data.keys()))], 
        label="Average Attribute Distance"
    )
    """
    plt.xlabel("Number of Queries", fontsize=12)
    plt.ylabel("Preference Estimate MSE to Target", fontsize=12)
    # plt.legend()
    plt.yscale("log")
    num_queries = max(list(lstm_data.keys()))
    plt.xticks(np.arange(5, num_queries + 1, 5))
    plt.xlim(left=1.0, right=num_queries)
    plt.ylim(bottom=0.00005)
    plt.tight_layout()
    plt.savefig(figure_save_path, dpi=300)

def compute_percentile_loss(
    current_attribute,
    target_attribute,
    sampler,
    num_samples=1000
):
    """
        Computes the percentage of faces with attribute vectors 
        CLOSER to the initial attribute vector than the current 
        attribute vector. 
    """
    # Generate a bunch of attribute samplers
    attribute_samples = sampler.randomly_sample_attributes(
        num_samples,
        return_dict=False
    ).detach().cpu().numpy()
    # Compute a bunch of attribute losses
    attribute_losses = []
    for sample_index in range(num_samples):
        attribute = attribute_samples[sample_index]
        attribute_loss = np.linalg.norm(target_attribute - attribute)
        attribute_loss = attribute_loss.item()
        attribute_losses.append(attribute_loss)
    attribute_losses = np.array(attribute_losses)
    attribute_losses = np.sort(attribute_losses) # Sort the losses
    # Compute the losses over time
    attribute_dist_over_time = []
    # Compute attribute loss
    attribute_loss_val = np.linalg.norm(target_attribute - current_attribute)
    attribute_loss_val = attribute_loss_val.item()
    # Measure how the loss ranks in 
    index = np.searchsorted(attribute_losses, attribute_loss_val)
    percentile = index / len(attribute_losses) * 100

    return percentile

def plot_performance_comparison_percentage_closer(
    latent_sampler,
    random_mcmc_data, 
    lstm_data, 
    mcmv_data,
    closed_form_mcmv_data,
    test_dataset, 
    figure_save_path
):
    lstm_averages = []
    lstm_vars = []
    random_mcmc_averages = []
    random_mcmc_vars = []
    mcmv_averages = []
    mcmv_vars = []
    closed_form_mcmv_averages = []
    closed_form_mcmv_vars = []
    # For each num queries in each dataset
    for num_queries in tqdm(lstm_data.keys()):
        lstm_data_for_num_queries = lstm_data[num_queries]
        random_mcmc_data_for_num_queries = random_mcmc_data[num_queries]
        mcmv_data_for_num_queries = mcmv_data[num_queries]
        closed_form_mcmv_data_for_num_queries = closed_form_mcmv_data[num_queries]

        lstm_diffs = []
        random_mcmc_diffs = []
        mcmv_diffs = []
        closed_form_mcmv_diffs = []
        for example_index in tqdm(range(len(lstm_data_for_num_queries))):
            # Get the example target attributes
            target_attributes = test_dataset[example_index][1]
            # Get the difference between the target attributes and the lstm estimate
            lstm_data_instance = lstm_data_for_num_queries[example_index]["preference_estimate"]
            lstm_diff = compute_percentile_loss(
                lstm_data_instance,
                target_attributes,
                latent_sampler,
                num_samples=500
            )
            random_mcmc_data_instance = random_mcmc_data_for_num_queries[example_index]["preference_estimate"]
            random_mcmc_diff = compute_percentile_loss(
                random_mcmc_data_instance,
                target_attributes,
                latent_sampler,
                num_samples=500
            )
            mcmv_data_instance = mcmv_data_for_num_queries[example_index]["preference_estimate"]
            mcmv_data_diff = compute_percentile_loss(
                mcmv_data_instance,
                target_attributes,
                latent_sampler,
                num_samples=500
            )
            closed_form_mcmv_data_instance = closed_form_mcmv_data_for_num_queries[example_index]["preference_estimate"]
            closed_form_mcmv_data_diff = compute_percentile_loss(
                closed_form_mcmv_data_instance,
                target_attributes,
                latent_sampler,
                num_samples=500
            )

            lstm_diffs.append(lstm_diff)
            random_mcmc_diffs.append(random_mcmc_diff)
            mcmv_diffs.append(mcmv_data_diff)
            closed_form_mcmv_diffs.append(closed_form_mcmv_data_diff)

        lstm_averages.append(np.mean(lstm_diffs))
        random_mcmc_averages.append(np.mean(random_mcmc_diffs))
        lstm_vars.append(np.std(lstm_diffs) / np.sqrt(len(lstm_diffs)))
        random_mcmc_vars.append(np.std(random_mcmc_diffs) / np.sqrt(len(random_mcmc_diffs)))
        mcmv_averages.append(np.mean(mcmv_diffs))
        mcmv_vars.append(np.std(mcmv_diffs) / np.sqrt(len(mcmv_diffs)))
        closed_form_mcmv_averages.append(np.mean(closed_form_mcmv_diffs))
        closed_form_mcmv_vars.append(np.std(closed_form_mcmv_diffs) / np.sqrt(len(closed_form_mcmv_diffs)))

    # Make a matplotlib line plot with two lines
    fig = plt.figure(figsize=(5*0.85, 3.3))
    ax = fig.add_subplot(111)

    ax.plot(
        list(lstm_data.keys()), 
        lstm_averages, 
        label="LSTM Baseline"
    )
    # Plot variance error bars
    ax.fill_between(
        list(lstm_data.keys()), 
        np.array(lstm_averages) + np.array(lstm_vars), 
        np.array(lstm_averages) - np.array(lstm_vars), 
        alpha=0.5
    )
    ax.plot(
        list(lstm_data.keys()), 
        random_mcmc_averages, 
        label="Ours Random Querying"
    )
    ax.fill_between(
        list(lstm_data.keys()), 
        np.array(random_mcmc_averages) + np.array(random_mcmc_vars), 
        np.array(random_mcmc_averages) - np.array(random_mcmc_vars), 
        alpha=0.5
    )

    ax.plot(
        list(mcmv_data.keys()), 
        mcmv_averages, 
        label="Ours Active Querying (Best of N)"
    )
    # Plot variance error bars
    ax.fill_between(
        list(mcmv_data.keys()), 
        np.array(mcmv_averages) + np.array(mcmv_vars), 
        np.array(mcmv_averages) - np.array(mcmv_vars), 
        alpha=0.5
    )
    ax.plot(
        list(lstm_data.keys()), 
        closed_form_mcmv_averages, 
        label="Ours Active Querying (Closed Form)"
    )
    ax.fill_between(
        list(lstm_data.keys()), 
        np.array(closed_form_mcmv_averages) + np.array(closed_form_mcmv_vars), 
        np.array(closed_form_mcmv_averages) - np.array(closed_form_mcmv_vars), 
        alpha=0.5
    )

    plt.xlabel("Number of Queries", fontsize=12)
    plt.ylabel("Percentage Closer to True Target", fontsize=12)
    plt.legend()
    plt.ylim(bottom=0.0)
    # plt.yscale("log")
    num_queries = max(list(lstm_data.keys()))
    plt.xticks(np.arange(5, num_queries + 1, 5))
    plt.xlim(left=1, right=num_queries)
    # plt.suptitle("Performance of Preference Estimation")
    plt.tight_layout()
    plt.savefig(figure_save_path, dpi=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--attribute_names", default=["yaw", "pitch", "roll", "age"])
    parser.add_argument(
        "--random_mcmc_data_path", 
        type=str, 
        default="data/random_mcmc_data_4d.pkl"
    )
    parser.add_argument(
        "--lstm_data_path", 
        type=str, 
        default="data/lstm_data_4d.pkl"
    )
    parser.add_argument(
        "--mcmv_data_path", 
        type=str, 
        default="data/mcmv_data_4d.pkl"
    )
    parser.add_argument(
        "--closed_form_mcmv_data_path",
        type=str,
        default="data/continuous_mcmv_data_4d.pkl"
    )

    parser.add_argument(
        "--plot_path",
        default="plots/LSTM_vs_Random_MCMC_Comparison_4d.pdf"
    )
    parser.add_argument(
        "--test_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/test_dataset_gan_control_4d.pkl"
        ),
    )

    args = parser.parse_args()
    # Load the lstm data and the 
    print("Loading Data...")
    random_mcmc_data = pickle.load(open(args.random_mcmc_data_path, "rb"))
    lstm_data = pickle.load(open(args.lstm_data_path, "rb"))
    mcmv_data = pickle.load(open(args.mcmv_data_path, "rb"))
    closed_form_mcmv_data = pickle.load(open(args.closed_form_mcmv_data_path, "rb"))

    test_dataset = ConstraintDataset(args.test_save_path)

    # Make StyleGAN Generator
    stylegan_generator = StyleGAN2Wrapper(
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent,
    )
    print("StyleGAN Generator Loaded!")
    latent_sampler = GANControlSampler(
        stylegan_generator,
        attribute_names=args.attribute_names,
    )
    print("Latent Sampler Loaded!")
    average_dist = average_distance_in_attribute_space(
        latent_sampler
    )
    print("Average Distance in Attribute Space: {}".format(average_dist))
    plot_performance_comparison(
        random_mcmc_data, 
        lstm_data, 
        mcmv_data,
        closed_form_mcmv_data,
        average_dist,
        test_dataset, 
        args.plot_path
    )

    plot_performance_comparison_percentage_closer(
        latent_sampler,
        random_mcmc_data, 
        lstm_data,
        mcmv_data,
        closed_form_mcmv_data,
        test_dataset, 
        "plots/performance_comparison_percentage_closer.pdf"
    )

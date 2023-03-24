import argparse
import matplotlib.pyplot as plt
import pickle
import os
import torch

from prefgen.methods.datasets.lstm import ConstraintDataset
from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler
from prefgen.methods.plotting.utils import convert_stylegan_image_to_matplotlib, plot_ignore_exceptions

import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'Times New Roman'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--attribute_names", default=["yaw", "age"])
    parser.add_argument(
        "--test_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/test_dataset_gan_control.pkl"
        ),
    )
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
        "--num_examples",
        default=2
    )
    parser.add_argument(
        "--which_queries",
        default=[
            1, 2, 3, 4, 5
        ]
    )

    args = parser.parse_args()

    generator = StyleGAN2GANControlWrapper(
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent,
    )
    latent_sampler = GANControlSampler(
        generator,
        attribute_names=args.attribute_names,
    )

    random_mcmc_data = pickle.load(open(args.random_mcmc_data_path, "rb"))
    lstm_data = pickle.load(open(args.lstm_data_path, "rb"))
    mcmv_data = pickle.load(open(args.mcmv_data_path, "rb"))
    test_dataset = ConstraintDataset(args.test_save_path)
    # Make a matplotlib plot showing 5 target images
    fig = plt.figure(
        constrained_layout=True, 
        figsize=(20, 6)
    )
    subfigs = fig.subfigures(
        nrows=1, 
        ncols=3, 
        # width_ratios=[1.0, 1.0, 1.0]
        wspace=0.1
    )

    lstm_axs = subfigs[0].subplots(
        args.num_examples, 
        len(args.which_queries) + 2
    )
    subfigs[0].suptitle("LSTM Baseline")
    lstm_axs[0, 0].set_title("Initial\nImage")
    lstm_axs[0, -1].set_title("Target\nImage")
    active_axs = subfigs[1].subplots(
        args.num_examples, 
        len(args.which_queries) + 2
    )
    subfigs[1].suptitle("Ours Active Query Selection")
    active_axs[0, 0].set_title("Initial\nImage")
    active_axs[0, -1].set_title("Target\nImage")
    random_axs = subfigs[2].subplots(
        args.num_examples, 
        len(args.which_queries) + 2
    )
    subfigs[2].suptitle("Ours Random Query Selection")
    random_axs[0, 0].set_title("Initial\nImage")
    random_axs[0, -1].set_title("Target\nImage")

    for subplot in [lstm_axs, active_axs, random_axs]:
        for ax in subplot.flat:
            ax.set_xticks([])
            ax.set_yticks([])

    for example_index in range(args.num_examples):
        # Plot the initial image
        initial_latent = test_dataset[example_index][0]
        _, initial_image = generator.generate_image(initial_latent)
        initial_image = convert_stylegan_image_to_matplotlib(initial_image)
        target_attributes = test_dataset[example_index][1]
        target_image, _, _, _ = latent_sampler.sample(
            target_attributes=target_attributes,
            initial_latent=initial_latent,
        )
        target_image = convert_stylegan_image_to_matplotlib(target_image)
        lstm_axs[example_index, 0].imshow(initial_image)
        active_axs[example_index, 0].imshow(initial_image)
        random_axs[example_index, 0].imshow(initial_image)
        lstm_axs[example_index, -1].imshow(target_image)
        active_axs[example_index, -1].imshow(target_image)
        random_axs[example_index, -1].imshow(target_image)
        # Plot the target image
        for query_index in args.which_queries:
            if example_index == 0:
                lstm_axs[example_index, query_index].set_title(f"{query_index}")
                active_axs[example_index, query_index].set_title(f"{query_index}")
                random_axs[example_index, query_index].set_title(f"{query_index}")
            # Plot each of the samples 
            lstm_estimate = lstm_data[query_index][example_index]["preference_estimate"]
            lstm_estimate = torch.Tensor(lstm_estimate).squeeze().cuda()
            lstm_image, _, _, _ = latent_sampler.sample(
                initial_latent=initial_latent,
                target_attributes=lstm_estimate
            )
            lstm_image = convert_stylegan_image_to_matplotlib(lstm_image)
            lstm_axs[example_index, query_index].imshow(lstm_image)

            mcmv_estimate = mcmv_data[query_index][example_index]["preference_estimate"]
            mcmv_image, _, _, _ = latent_sampler.sample(
                initial_latent=initial_latent,
                target_attributes=mcmv_estimate
            )
            mcmv_image = convert_stylegan_image_to_matplotlib(mcmv_image)
            active_axs[example_index, query_index].imshow(mcmv_image)

            random_estimate = random_mcmc_data[query_index][example_index]["preference_estimate"]
            random_image, _, _, _ = latent_sampler.sample(
                initial_latent=initial_latent,
                target_attributes=random_estimate
            )
            random_image = convert_stylegan_image_to_matplotlib(random_image)
            random_axs[example_index, query_index].imshow(random_image)

            print(f"Random estimate: {random_estimate}")
            print(f"MCMV estimate: {mcmv_estimate}")
            print(f"LSTM estimate: {lstm_estimate}")
            print(f"Target attributes: {target_attributes}")

    plt.savefig("plots/each_method_query_samples.pdf", dpi=200)


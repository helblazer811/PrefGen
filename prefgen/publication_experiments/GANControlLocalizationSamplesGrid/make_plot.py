import argparse
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.plotting.utils import convert_stylegan_image_to_matplotlib
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler 

mpl.rcParams['font.family'] = 'Times New Roman'

def plot_samples_grid(trail_data, generator, latent_sampler: GANControlSampler, save_path):
    num_queries = len(trial_data[0].latents_over_time) - 1
    fig, axs = plt.subplots(len(trial_data), num_queries + 2, figsize=((num_queries + 2)*0.7, len(trial_data)*0.8))
    for trial_index, data in enumerate(trail_data):
        if trial_index == 0:
            axs[trial_index, 0].set_title("Initial\nImage")
            axs[trial_index, -1].set_title("Target\nImage")
        # Plot the initial image
        _, initial_image = generator.generate_image(latent=data.start_latent)
        initial_image = convert_stylegan_image_to_matplotlib(initial_image)
        axs[trial_index, 0].imshow(initial_image)
        # Plot the target image
        target_attributes = data.target_attributes
        _, target_latent = latent_sampler.sample(target_attributes, data.start_latent, verbose_logging=False)
        _, target_image = generator.generate_image(latent=target_latent)
        target_image = convert_stylegan_image_to_matplotlib(target_image)
        axs[trial_index, -1].imshow(target_image)
        # Plot each of the samples
        latents = data.latents_over_time[1:]
        for latent_index, latent in enumerate(latents):
            if trial_index == 0:
                axs[trial_index, latent_index + 1].set_title(f"Sample\n{latent_index}")
            _, image = generator.generate_image(latent=latent)
            image = convert_stylegan_image_to_matplotlib(image)
            axs[trial_index, latent_index + 1].imshow(image)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(save_path, dpi=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Classifiers is All You Need")

    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--attribute_names", type=list, default=["yaw", "pitch", "roll", "age"]) #"pitch", "light0", "light3", "age"])
    parser.add_argument("--experiment_dir", default="experiment_logs/experiment_3")

    args = parser.parse_args()
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
    # Load up the data from each trial
    trial_data = []
    trials = os.listdir(args.experiment_dir)
    trials = filter(lambda x: x.startswith("trial"), trials)
    for trial_path in trials:
        trial_path = os.path.join(
            args.experiment_dir, 
            trial_path,
            "simulation_data.pkl"
        )
        with open(trial_path, "rb") as f:
            trial_data.append(
                pickle.load(f)
            )
    trial_data = trial_data[:4]
    # Make the plot
    plot_samples_grid(
        trial_data,
        generator,
        latent_sampler,
        "samples_grid.pdf"
    )
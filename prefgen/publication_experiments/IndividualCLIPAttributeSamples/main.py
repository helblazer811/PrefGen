"""
    Here we want to make a figure showing that we can use 
    CLIP based attribute classifiers to uniformly sample
    different intensities of relative features. 
"""
from argparse import Namespace
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import os
import torch
import clip

from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.sampling.clip_stylegan.clip_attribute_classifier import CLIPAttributeClassifier, CLIPCosineSimilarityClassifier
from prefgen.methods.sampling.clip_stylegan.util import sample_clip_attribute

mpl.rcParams['font.family'] = 'Times New Roman'

def generate_latents(
    attribute_classifiers,
    generator,
    num_examples=2,
    attribute_range=(0, 1),
    attribute_interval=0.5,
    use_adam=True,
    gradient_descent_args=None,
):
    # Generate samples
    num_examples = len(attribute_classifiers)
    num_x_images = int((attribute_range[1] - attribute_range[0]) / attribute_interval)
    sample_latents = []
    start_latent, _ = generator.generate_image()
 
    for example_index in tqdm(range(num_examples)):
        gd_args = gradient_descent_args[example_index]
        attribute_classifier = attribute_classifiers[example_index]
        example_att_sample_latents = []
        example_att_sample_latents.append(
            start_latent.detach().cpu().numpy()
        )
        # Generate a random latent vector
        for image_index in tqdm(range(num_x_images)):
            attribute_value = attribute_range[0] + image_index * attribute_interval
            sample_latent, _ = sample_clip_attribute(
                start_latent,
                attribute_value,
                generator=generator,
                attribute_classifier=attribute_classifier,
                gradient_descent_args=gd_args,
                use_adam=use_adam,
            )
            sample_latent = sample_latent.detach().cpu().numpy()
            example_att_sample_latents.append(sample_latent)

        sample_latents.append(
            example_att_sample_latents
        )

    return sample_latents

def plot_attribute_samples(
    latents,
    generator,
    low_prompts=["neutral expression", "dark hair"],
    high_prompts=["angry expression", "red hair"],
    attribute_range=(0, 1),
    attribute_interval=0.5,
    image_save_path="plots/attribute_samples.pdf",
):
    """
        Generates samples of given images using gradient descent to optimize 
        the attribute value, where the attribute corresponds to a text concept. 
    """
    num_x_images = int((attribute_range[1] - attribute_range[0]) / attribute_interval + 1)
    # Make a matplotlib image grid
    fig, ax = plt.subplots(
        len(low_prompts), 
        num_x_images, 
        figsize=(3.25*1.6, 3.25*1.4)
    )
    plt.subplots_adjust(wspace=0.15, hspace=0.09)
    for example_index in tqdm(range(len(low_prompts))):
        # plt.text(0.15, 0.5, low_prompts[example_index], ha='center', va='center', rotation=0, transform=ax[example_index, 0].transAxes, fontsize=12)
        # plt.text(0.85, 0.5, high_prompts[example_index], ha='center', va='center', rotation=0, transform=ax[example_index, -1].transAxes, fontsize=12)
        ax[example_index, 0].set_title(low_prompts[example_index], fontsize=12)
        ax[example_index, -1].set_title(high_prompts[example_index], fontsize=12)

        for sample_index in tqdm(range(len(latents[example_index]))):
            latent = latents[example_index][sample_index]
            latent = torch.Tensor(latent).cuda()
            _, image = generator.generate_image(
                latent=latent
            )
            image = image.squeeze().detach().cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            ax[example_index, sample_index].imshow(image)
            if example_index == 0:
                plt.text(
                    0.12,
                    -0.35,
                    r"$\alpha$"+f"={attribute_range[0] + sample_index * attribute_interval:.2f}",
                    fontsize=12,
                    transform=ax[-1, sample_index].transAxes 
                )
    
    for row in ax:
        for col in row:
            col.set_xticks([])
            col.set_yticks([])

    plt.tight_layout()
    plt.savefig(image_save_path, dpi=600)

if __name__ == "__main__":
    # Make Generator
    generator = StyleGAN2Wrapper()
    # Make the attribute classifier
    # CLIP model and preprocess
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    # Make Classifier
    target_prompts = [
        "person with a suprised expression",
        "person with an angry expression",
        "person with a smiling expression",
        "person with red hair"
    ]
    neutral_prompts = [
        "person with a neutral expression",
        "person with a neutral expression",
        "person with a neutral expression",
        "person with dark hair"
    ]
    attribute_classifiers = [
        CLIPCosineSimilarityClassifier(
            generator,
            neutral_prompts[index],
            target_prompts[index],
            prompt_engineering=True,
            clip_model=clip_model,
            preprocess=preprocess,
        #    scale_output=True
        )
        for index in range(len(target_prompts))
    ]
    """
    gradient_descent_args = [
        Namespace(
            learning_rate=0.002,
            x_diff_regularization=0.0, 
            z_diff_regularization=0.001, 
            id_loss_multiplier=0.2,
            learning_rate_decay=1.0,
            w_space_latent=True,
            extended_latent=True,
            num_steps=300
        ),
        Namespace(
            learning_rate=0.002,
            x_diff_regularization=0.0, 
            z_diff_regularization=0.001, 
            id_loss_multiplier=0.3,
            learning_rate_decay=1.0,
            w_space_latent=True,
            extended_latent=True,
            num_steps=300
        ),
        Namespace(
            learning_rate=0.002,
            x_diff_regularization=0.0, 
            z_diff_regularization=0.001, 
            id_loss_multiplier=0.2,
            learning_rate_decay=1.0,
            w_space_latent=True,
            extended_latent=True,
            num_steps=300
        ),
        Namespace(
            learning_rate=0.005,
            x_diff_regularization=0.0, 
            z_diff_regularization=0.001, 
            id_loss_multiplier=0.1,
            learning_rate_decay=1.0,
            w_space_latent=True,
            extended_latent=True,
            num_steps=300
        ),
    ]
    """
    gradient_descent_args = [
        Namespace(
            learning_rate=0.001,
            x_diff_regularization=0.0, 
            z_diff_regularization=0.001, 
            id_loss_multiplier=0.05,
            learning_rate_decay=1.0,
            w_space_latent=True,
            extended_latent=True,
            num_steps=300
        ),
        Namespace(
            learning_rate=0.001,
            x_diff_regularization=0.0, 
            z_diff_regularization=0.001, 
            id_loss_multiplier=0.1,
            learning_rate_decay=1.0,
            w_space_latent=True,
            extended_latent=True,
            num_steps=300
        ),
        Namespace(
            learning_rate=0.001,
            x_diff_regularization=0.0, 
            z_diff_regularization=0.001, 
            id_loss_multiplier=0.1,
            learning_rate_decay=1.0,
            w_space_latent=True,
            extended_latent=True,
            num_steps=300
        ),
        Namespace(
            learning_rate=0.001,
            x_diff_regularization=0.0, 
            z_diff_regularization=0.001, 
            id_loss_multiplier=0.1,
            learning_rate_decay=1.0,
            w_space_latent=True,
            extended_latent=True,
            num_steps=300
        ),
    ]
    # Plot the attribute samples
    if os.path.exists("cached_latents/attribute_samples.npy"):
        cached_latents = np.load("cached_latents/attribute_samples.npy")
        cached_latents = torch.Tensor(cached_latents).cuda()
        latents = cached_latents
    else:
        torch.manual_seed(12)
        latents = generate_latents(
            attribute_classifiers,
            generator,
            attribute_range=(0, 1),
            attribute_interval=0.2,
            gradient_descent_args=gradient_descent_args
        )
        with open("cached_latents/attribute_samples.npy", "wb") as f:
            np.save(f, latents)

    plot_attribute_samples(
        latents,
        generator,
        low_prompts=["\"neutral face\"", "\"neutral face\"", "\"neutral face\"", "\"dark hair\""],
        high_prompts=["\"surprised face\"", "\"angry face\"", "\"smiling face\"", "\"red hair\""],
        attribute_range=(0, 1),
        attribute_interval=0.2,
        image_save_path="plots/attribute_samples.pdf",
    )
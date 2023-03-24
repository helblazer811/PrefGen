import torch
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler

def sample_triplets(latent_sampler, num_triplets=1000, noise_scale=0.0):
    # Sample random anchor points
    anchor_points = latent_sampler.randomly_sample_attributes( 
        num_samples=num_triplets,
        return_dict=False
    )
    # Sample random positive points
    candidate_a_points = latent_sampler.randomly_sample_attributes(
        num_samples=num_triplets,
        return_dict=False
    )
    candidate_b_points = latent_sampler.randomly_sample_attributes(
        num_samples=num_triplets,
        return_dict=False
    )
    triplets = []
    for triplet_index in tqdm(range(num_triplets)):
        anchor = anchor_points[triplet_index]
        anchor_a_dist = torch.norm(anchor - candidate_a_points[triplet_index])**2 + np.random.normal(0, noise_scale)
        anchor_b_dist = torch.norm(anchor - candidate_b_points[triplet_index])**2
        if anchor_a_dist < anchor_b_dist:
            positive = candidate_a_points[triplet_index]
            negative = candidate_b_points[triplet_index]
        else:
            positive = candidate_b_points[triplet_index]
            negative = candidate_a_points[triplet_index]

        """
        random_num = np.random.rand()
        if random_num < random_flip_chance:
            tmp = negative
            negative = positive
            positive = tmp
        """

        triplet = (anchor, positive, negative)
        triplets.append(
            triplet
        )

    return triplets

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def compute_logistic_likelihood(triplet, k=1.0):
    return sigmoid(k * (torch.norm(triplet[0] - triplet[2])**2 - torch.norm(triplet[0] - triplet[1]))).item()

if __name__ == "__main__":
    # Plot the noise constant likelihood
    stylegan_generator = StyleGAN2GANControlWrapper(
        extended_latent=False,
        w_space_latent=True,
    )
    latent_sampler = GANControlSampler(
        stylegan_generator,
        attribute_names=["yaw", "age"]
    )
    noise_scales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for noise_scale in noise_scales:
        print("Sampling triplets")
        triplets = sample_triplets(latent_sampler, num_triplets=1000, noise_scale=noise_scale)

        noise_constants = torch.logspace(-2, 1.2, 50)

        df = pd.DataFrame()
        print("Computing likelihoods")
        noise_to_likelihoods = {}
        index = 0
        for noise_constant in tqdm(noise_constants):
            noise_constant = noise_constant.item()
            noise_to_likelihoods[noise_constant] = []
            for triplet in tqdm(triplets):
                likelihood = compute_logistic_likelihood(triplet, k=noise_constant)
                df = df.append(
                        pd.DataFrame({
                            "K Constant": f"{noise_constant:.2f}",
                            "Likelihood": likelihood,
                            "index": [index]
                        }, 
                    )
                )
                noise_to_likelihoods[noise_constant].append(likelihood)
                index += 1

        # Plot likelihoods on y and noise constant on x
        sns.boxplot(
            x="K Constant",
            y="Likelihood",
            data=df,
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(f"noise_constant_tuning_noise_scale_{noise_scale}.pdf")
        # Print the averages
        scales = []
        means = []
        for noise_constant in noise_constants:
            noise_constant = noise_constant.item()
            scales.append(noise_scale)
            mean = np.mean(noise_to_likelihoods[noise_constant])
            means.append(mean)

        print("Noise scales: {}".format(scales[np.argmax(means)]))

        fig = plt.figure()
        plt.plot(noise_constants, means, label="Best k constant: {}".format(noise_constants[np.argmax(means)]))
        plt.legend()
        plt.xscale("log")
        plt.savefig(f"line_plot_noise_scale_{noise_scale}.pdf")
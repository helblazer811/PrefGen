from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier

import torch
import pickle
from torch.utils.data import Dataset
import argparse
import numpy as np
import os
from tqdm import tqdm

class ConstraintDataset(Dataset):

    def __init__(self, constraints_path, with_initial_latent=False):
        with open(constraints_path, "rb") as f:
            self.constraints = pickle.load(f)
        self.with_initial_latent = with_initial_latent

    def __len__(self):
        return len(self.constraints)

    def __getitem__(self, idx):
        constraint = self.constraints[idx]
        target_latent = constraint["target_latent"]
        target_attributes = constraint["target_attributes"]
        constraint_attributes = constraint["constraint_attributes"]
        constraints = []
        for pair in constraint_attributes:
            constraints.append(np.concatenate(pair))

        constraints = np.stack(constraints)
        if self.with_initial_latent:
            initial_latent = constraint["initial_latent"]
            return target_latent, target_attributes, constraints, initial_latent
        else:
            return target_latent, target_attributes, constraints

def make_dataset_with_gan_control(
    generator=None,
    num_examples=100,
    num_constraints=1,
    attribute_names=["age"],
    attributes_normalized=True,
    save_path="",
):
    """Similar to make_dataset except generates images
    by feeding attributes in first, and makes triplets from them"""
    if generator is None:
        # Make a Gan Control Stylegan model
        generator = StyleGAN2GANControlWrapper(
            rescale=False
        )
    # Make a GANControlSampler object
    sampler = GANControlSampler(
        generator,
        attribute_names=attribute_names,
    )

    with torch.no_grad():
        dataset = []
        for example_number in tqdm(range(num_examples)):
            # Pick a random target attribute
            target_attributes = sampler.randomly_sample_attributes(
                normalize=attributes_normalized
            )
            _, initial_latent, target_latent, _ = sampler.sample(
                target_attributes=target_attributes,
                attributes_normalized=attributes_normalized
            )
            target_attributes = sampler.convert_controls_dict_to_attribute_vector(
                target_attributes
            )
            target_attributes = target_attributes.detach().cpu().numpy()
            target_latent = target_latent.detach().cpu().numpy()
            # Generate a set of constraints
            constraint_latents = []
            constraint_attributes = []
            for _ in range(num_constraints):
                # Generate two random images
                # Make the first constraint
                constraint_a_attributes = sampler.randomly_sample_attributes(
                    normalize=attributes_normalized
                )
                _, _, constraint_a_latent, _ = sampler.sample(
                    target_attributes=constraint_a_attributes,
                    attributes_normalized=attributes_normalized
                )
                constraint_a_attributes = sampler.convert_controls_dict_to_attribute_vector(
                    constraint_a_attributes
                )
                constraint_a_attributes = constraint_a_attributes.detach().cpu().numpy()
                constraint_a_latent = constraint_a_latent.detach().cpu().numpy()
                # Make the second constraint
                constraint_b_attributes = sampler.randomly_sample_attributes(
                    normalize=attributes_normalized
                )
                _, _, constraint_b_latent, _ = sampler.sample(
                    target_attributes=constraint_b_attributes,
                    attributes_normalized=attributes_normalized
                )
                constraint_b_attributes = sampler.convert_controls_dict_to_attribute_vector(
                    constraint_b_attributes
                )
                constraint_b_attributes = constraint_b_attributes.detach().cpu().numpy()
                constraint_b_latent = constraint_b_latent.detach().cpu().numpy()
                # Figure out which image is closer to the target image
                a_target_dist = np.linalg.norm(constraint_a_attributes - target_attributes)
                b_target_dist = np.linalg.norm(constraint_b_attributes - target_attributes)
                if a_target_dist < b_target_dist:
                    positive_latent = constraint_a_latent
                    negative_latent = constraint_b_latent
                    positive_attributes = constraint_a_attributes
                    negative_attributes = constraint_b_attributes
                else:
                    positive_latent = constraint_b_latent
                    negative_latent = constraint_a_latent
                    positive_attributes = constraint_b_attributes
                    negative_attributes = constraint_a_attributes

                constraint_latents.append((positive_latent, negative_latent))
                constraint_attributes.append((positive_attributes, negative_attributes))

            dataset.append({
                "constraint_latents": constraint_latents,
                "constraint_attributes": constraint_attributes,
                "target_latent": target_latent,
                "target_attributes": target_attributes,
                "initial_latent": initial_latent
            })
    # Pickle the created dataset
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
        
    return dataset

def make_dataset(
    attribute_classifier, 
    generator, 
    num_examples=100, 
    num_constraints=30,
    save_path=""
):
    """Generates a dataset of constraints"""
    dataset = []
    for example_number in tqdm(range(num_examples)):
        # Pick a random target attribute
        target_latent = generator.randomly_sample_latent()
        target_attributes = attribute_classifier(latent=target_latent).detach().cpu().numpy()
        target_latent = target_latent.detach().cpu().numpy()
        # Generate a set of constraints
        constraint_latents = []
        constraint_attributes = []
        for _ in range(num_constraints):
            # Generate two random images
            constraint_a_latent = generator.randomly_sample_latent()
            constraint_a_attributes = attribute_classifier(latent=constraint_a_latent)
            constraint_a_attributes = constraint_a_attributes.detach().cpu().numpy()
            constraint_a_latent = constraint_a_latent.detach().cpu().numpy()

            constraint_b_latent = generator.randomly_sample_latent()
            constraint_b_attributes = attribute_classifier(latent=constraint_b_latent)
            constraint_b_attributes = constraint_b_attributes.detach().cpu().numpy()
            constraint_b_latent = constraint_b_latent.detach().cpu().numpy()
            # Figure out which image is closer to the target image
            a_target_dist = np.linalg.norm(constraint_a_attributes - target_attributes)
            b_target_dist = np.linalg.norm(constraint_b_attributes - target_attributes)
            if a_target_dist < b_target_dist:
                positive_latent = constraint_a_latent
                negative_latent = constraint_b_latent
                positive_attributes = constraint_a_attributes
                negative_attributes = constraint_b_attributes
            else:
                positive_latent = constraint_b_latent
                negative_latent = constraint_a_latent
                positive_attributes = constraint_b_attributes
                negative_attributes = constraint_a_attributes

            constraint_latents.append((positive_latent, negative_latent))
            constraint_attributes.append((positive_attributes, negative_attributes))

        dataset.append({
            "constraint_latents": constraint_latents,
            "constraint_attributes": constraint_attributes,
            "target_latent": target_latent,
            "target_attributes": target_attributes
        })
    # Pickle the created dataset
    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)
        
    return dataset

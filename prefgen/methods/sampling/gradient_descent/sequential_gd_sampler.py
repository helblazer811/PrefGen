
from argparse import Namespace
import torch
import numpy as np

from prefgen.methods.sampling.gradient_descent.adam_sampler import AdamSampler
from prefgen.methods.sampling.gradient_descent.sampler import GradientDescentSampler

class SequentialGradientDescentCLIP():
    """
        Performs gradient descent on latent vectors to encourage
        the generated images to have a particular attribute value.
        
        Sequentially optimizes the latent vector for each attribute
        instead of all at once. 
    """

    def __init__(
        self, 
        generator,
        attribute_classifiers, 
        gradient_descent_parameters=Namespace(
            learning_rate=0.1,
            x_diff_regularization=0.00, 
            z_diff_regularization=0.00, 
            id_loss_multiplier=0.00,
            learning_rate_decay=1.0,
            num_steps=100
        ),
        per_attribute_parameters=None,
        use_adam=False
    ):
        self.generator = generator
        self.attribute_classifiers = attribute_classifiers
        self.use_adam = use_adam
        if per_attribute_parameters is None:
            self.per_attribute_parameters = [gradient_descent_parameters] * len(attribute_classifiers)
        else:
            self.per_attribute_parameters = per_attribute_parameters
        # Initialize gradient descent sampler
        self.gradient_descent_samplers = []
        for attribute_classifier_index in range(len(attribute_classifiers)):
            classifier = attribute_classifiers[attribute_classifier_index]
            gradient_descent_parameters = self.per_attribute_parameters[attribute_classifier_index]

            if use_adam:
                sampler = AdamSampler(
                    classifier,
                    generator,
                    learning_rate=gradient_descent_parameters.learning_rate,
                    x_diff_regularization=gradient_descent_parameters.x_diff_regularization,
                    z_diff_regularization=gradient_descent_parameters.z_diff_regularization,
                    id_loss_multiplier=gradient_descent_parameters.id_loss_multiplier,
                    learning_rate_decay=gradient_descent_parameters.learning_rate_decay,
                    num_steps=gradient_descent_parameters.num_steps,
                )
                self.gradient_descent_samplers.append(sampler)
            else:
                sampler = GradientDescentSampler(
                    classifier,
                    generator,
                    learning_rate=gradient_descent_parameters.learning_rate,
                    x_diff_regularization=gradient_descent_parameters.x_diff_regularization,
                    z_diff_regularization=gradient_descent_parameters.z_diff_regularization,
                    id_loss_multiplier=gradient_descent_parameters.id_loss_multiplier,
                    learning_rate_decay=gradient_descent_parameters.learning_rate_decay,
                    num_steps=gradient_descent_parameters.num_steps
                )
                self.gradient_descent_samplers.append(sampler)

    def measure_attributes(self, latent=None, image=None):
        if latent is None:
            latent, image = self.generator.generate_image()
        else:
            latent, image = self.generator.generate_image(latent=latent)

        attribute_values = []
        for attribute_classifier in self.attribute_classifiers:
            attribute_values.append(
                attribute_classifier(image=image)
            )

        attribute_values = torch.cat(attribute_values, dim=-1)

        return attribute_values

    def generate_latent_attribute_pairs(
        self,
        num_examples=1000
    ):
        latents = []
        attribute_values = []
        with torch.no_grad():
            for _ in range(num_examples):
                # Randomly generate a latent vector and image
                latent, image = self.generator.generate_image()
                # Measure the attribute of the image
                attributes = self.measure_attributes(image=image).detach().cpu().numpy()
                # Add to list
                latents.append(latent)
                attribute_values.append(attributes)

        return latents, attribute_values

    def randomly_sample_attributes(
        self,
        num_samples=1000,
        return_dict=False
    ):
        latents, attribute_values = self.generate_latent_attribute_pairs(
            num_examples=num_samples
        )
        attribute_values = torch.Tensor(attribute_values).cuda()
        return attribute_values


    def sample(
        self, 
        target_attributes, 
        initial_latent=None, 
        num_steps=None, 
        latent_dim=512, 
        verbose_logging=False, 
        best_of_n_chains=1, 
        w_space_latent=None, 
        extended_latent=None,
        learning_rate_decay=1.0,
        target_latent=None, 
        noise_std=0.00,
        return_intermediate_latents=False,
    ):
        if extended_latent is None:
            extended_latent = self.generator.extended_latent
        if w_space_latent is None:
            w_space_latent = self.generator.w_space_latent
        # Initialize latent vector
        if initial_latent is None:
            initial_latent = self.generator.randomly_sample_latent(
                w_space_latent=w_space_latent,
                extended_latent=extended_latent,
            )
        if isinstance(initial_latent, np.ndarray):
            initial_latent = torch.Tensor(initial_latent).cuda()
        current_latent = initial_latent
        intermediate_latents = []
        # Perform gradient descent for each attribute
        # TODO: Make a system for changing the attribute order
        for attribute_index, attribute_value in enumerate(target_attributes):
            current_sampler = self.gradient_descent_samplers[attribute_index]
            current_target_attributes = torch.Tensor([attribute_value]).double().cuda() 
            _, current_latent = current_sampler.sample(
                target_attributes=current_target_attributes,
                initial_latent=current_latent,
                w_space_latent=w_space_latent,
                extended_latent=extended_latent,
                learning_rate_decay=learning_rate_decay,
                verbose_logging=verbose_logging,
            )
            intermediate_latents.append(current_latent)
        
        if return_intermediate_latents:
            return initial_latent, intermediate_latents
        else:
            return initial_latent, current_latent
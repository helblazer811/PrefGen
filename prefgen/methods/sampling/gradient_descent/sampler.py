"""
    Here I implement basic Stochastic Gradient Langevin Dynamics process so that
    I can perform conditional sampling. 
"""
from prefgen.methods.attribute_classifiers.face_identification.load import load_identity_classifier
from prefgen.methods.sampling.utils import LatentAttributeSampler
from prefgen.methods.attribute_classifiers.face_identification.id_loss import IDLoss

import torch
import numpy as np
from tqdm import tqdm
import wandb
import math


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

class GradientDescentSampler(LatentAttributeSampler):
    """
        Implements gradient descent in latent space
    """

    def __init__(
        self, 
        classifier, 
        generator: None, 
        device="cuda",
        learning_rate=0.01,
        x_diff_regularization=0.00, 
        z_diff_regularization=0.00, 
        id_loss_multiplier=0.00,
        learning_rate_decay=1.0,
        wandb_group_name="Gradient Descent Sampling"
    ):
        assert not generator is None
        self.classifier = classifier
        self.generator = generator
        self.id_loss = IDLoss().to(device)
        self.learning_rate = learning_rate
        self.x_diff_regularization = x_diff_regularization
        self.z_diff_regularization = z_diff_regularization
        self.id_loss_multiplier = id_loss_multiplier
        self.learning_rate_decay = learning_rate_decay
        self.wandb_group_name = wandb_group_name

    def randomly_sample_attributes(self, num_samples=1000):
        """
            Sample random attributes from the attribute space.
        """
        attributes = []
        for _ in range(num_samples):
            _, image = self.generator.generate_image()
            attribute = self.classifier(image=image)
            attributes.append(attribute)

        attributes = np.stack(attributes, axis=0)

        return attributes
    
    def compute_combined_loss(
        self,
        initial_latent, 
        current_latent,
        input_image,
        target_attributes,
        verbose_logging=True,
        iteration_num=1,
        w_space_latent=False,
        extended_latent=False
    ):
        # Evaluate classifiers
        classifier_out = self.classifier(
            current_latent, 
        )
        # Compute MSE in attribute space
        attribute_mse = torch.linalg.norm(classifier_out - target_attributes) ** 2
        # Regularize the differnece between start and current images
        z_difference = torch.linalg.norm(current_latent - initial_latent) ** 2
        # z_difference = 0.0
        _, current_image = self.generator.generate_image(
            current_latent, 
            w_space_latent=w_space_latent, 
            extended_latent=extended_latent
        )
        x_difference = torch.linalg.norm(current_image - input_image) ** 2
        # Compute identity loss
        id_loss = self.id_loss(input_image, current_image)
        # Compute the Wspace identity
        # print("id_loss_multiplier, z_diff_regularization, x_diff_regularization")
        # print(self.id_loss_multiplier, self.z_diff_regularization, self.x_diff_regularization)
        # Combine losses
        loss = attribute_mse + \
            self.id_loss_multiplier * id_loss + \
            self.z_diff_regularization * z_difference + \
            self.x_diff_regularization * x_difference 
        
        # Perform logging
        if verbose_logging and iteration_num % 10 == 0:
            # Wandb logging
            wandb.log({
                "Attribute MSE": attribute_mse.detach().cpu().numpy(),
                "Identity Loss": id_loss.detach().cpu().numpy(),
            #    "Identity Loss Grad Mag": id_grad_mag.detach().cpu().numpy(),
            #    "Energy Loss Grad Mag": energy_grad_mag.detach().cpu().numpy(),
            #    "Id Wspace Loss": id_wspace_loss.detach().cpu().numpy(),
                "X Difference Loss": x_difference.detach().cpu().numpy(),
                "Z Difference Loss": z_difference.detach().cpu().numpy(),
                "Target Attributes": target_attributes.detach().cpu().numpy(),
                "Classifier out": str(classifier_out),
                "Combined Loss": loss.detach().cpu().numpy(),
                "Current Image": wandb.Image(current_image),
                "Current Latent Magnitude": torch.norm(current_latent).item(),
                "Start Latent Magnitude": torch.norm(initial_latent).item(),
                "Iteration": iteration_num
            })
        
        return loss

    def sample_gradient_descent(self, initial_latent, input_image, target_attributes,
                                num_steps=100, w_space_latent=False, verbose_logging=False,
                                lr_decay=True, extended_latent=False, noise_std=0.01):
        """
            Samples using gradient descent with a constant
            learning rate and amount of noise.
        """

        with torch.enable_grad():
            target_attributes = target_attributes.detach().clone()
            target_attributes.requires_grad = True
            initial_latent = initial_latent.detach().clone()
            initial_latent.requires_grad = True
            # Ensure gradients are required
            # This treatment of gradients prevents non-leaf node error
            current_latent = initial_latent.clone()
            # current_latent.requires_grad = True
            # Optimizer
            # Log the initial image
            if verbose_logging:
                wandb.log({
                    "Initial Image": wandb.Image(input_image),
                })
            # Run gradient descent
            print("Running Gradient Descent")
            for iteration_num in tqdm(range(num_steps), position=0, leave=True):
                # Compute learning rate
                if lr_decay:
                    learning_rate = get_lr(iteration_num/num_steps, self.learning_rate)
                else:
                    learning_rate = self.learning_rate
                # optimizer.param_groups[0]["lr"] = learning_rate
                # Evaluate classifiers
                current_latent, current_image = self.generator.generate_image(
                    current_latent
                )
                classifier_out = self.classifier(
                    latent=current_latent,
                )
                # Compute MSE in attribute space
                attribute_mse = torch.linalg.norm(classifier_out - target_attributes) ** 2
                # input_image = input_image.detach().clone()
                # input_image.requires_grad = True
                id_loss = self.id_loss(input_image, current_image)
                # Regularize the differnece between start and current images
                # initial_latent = initial_latent.detach().clone()
                # initial_latent.requires_grad = True
                z_difference = ((current_latent - initial_latent) ** 2).sum()
                # input_image = input_image.detach().clone()
                # input_image.requires_grad = True
                x_difference = torch.linalg.norm(current_image - input_image) ** 2
                # Combine losses
                loss = attribute_mse + \
                    self.z_diff_regularization * z_difference + \
                    self.x_diff_regularization * x_difference + \
                    self.id_loss_multiplier * id_loss

                # optimizer.zero_grad()
                # Compute the gradient of the energy
                grad = torch.autograd.grad(-1*loss, [current_latent])[0]
                # Compute next chain value
                current_latent = current_latent + \
                                    learning_rate * grad
                # optimizer.step()
                # lr_scheduler.step()

                if iteration_num % 10 == 0 and verbose_logging:
                    wandb.log({
                        "Attribute MSE": attribute_mse.detach().cpu().numpy(),
                        "Current Attribute": classifier_out.detach().cpu().numpy(),
                        "Identity Loss": id_loss.detach().cpu().numpy(),
                        "X Difference Loss": x_difference.detach().cpu().numpy(),
                        "Z Difference Loss": z_difference.detach().cpu().numpy(),
                        "Combined Loss": loss.detach().cpu().numpy(),
                        "Current Image": wandb.Image(current_image),
                    })

                # Perform gradient step
                current_latent.grad = None
                initial_latent.grad = None
                input_image.grad = None
                target_attributes.grad = None
                loss.grad = None

        final_latent = current_latent.data.detach()
        loss = loss.detach()
        return final_latent, loss

    def sample(self, target_attributes, initial_latent=None, num_steps=100, latent_dim=512, 
                verbose_logging=False, best_of_n_chains=1, w_space_latent=True, learning_rate_decay=0.99,
                target_latent=None, extended_latent=False, noise_std=0.00):
        """
            Samples a vector in StyleGAN2 latent space 
            using Stochastic Gradient Langevin Dymamics. 
        """

        with torch.enable_grad():
            # Initialize latent
            if initial_latent is None:
                # Randomly sample a latent from Gaussian
                initial_latent = self.generator.randomly_sample_latent()
            # Generate an input image
            _, input_image = self.generator.generate_image(
                initial_latent, 
            )
            # Go through n trials and pick the sample with highest likelihood
            best_latent_loss = float("inf")
            best_latent = initial_latent
            for chain_index in range(best_of_n_chains):
                # Setup logging
                if verbose_logging:
                    run = wandb.init(
                        project="GanSearch",
                        group=self.wandb_group_name,
                        config={
                            "learning_rate": self.learning_rate,
                            "learning_rate_decay": self.learning_rate_decay,
                            "x_diff_regularization": self.x_diff_regularization,
                            "z_diff_regularization": self.z_diff_regularization,
                            "id_loss_multiplier": self.id_loss_multiplier,
                        },
                        settings=wandb.Settings(start_method='fork')
                    )
                # Log target image
                if not target_latent is None:
                    _, target_attribute_image = self.generator.generate_image(
                        target_latent, 
                    )
                    wandb.log({
                        "Target attribute image": wandb.Image(target_attribute_image),
                    })
                # Choose which algorithm to use
                final_latent, final_loss = self.sample_gradient_descent(
                    initial_latent, 
                    input_image, 
                    target_attributes,
                    num_steps=num_steps,
                    w_space_latent=w_space_latent,
                    noise_std=noise_std,
                    verbose_logging=verbose_logging,
                )
                # Update best
                if final_loss < best_latent_loss:
                    best_latent = final_latent
                    best_latent_loss = final_loss
                        
                # End wandb run
                if verbose_logging:
                    wandb.finish()

            return initial_latent, final_latent

"""
    Here I implement basic Stochastic Gradient Langevin Dynamics process so that
    I can perform conditional sampling. 
"""
from argparse import Namespace
import numpy as np
import torch
from tqdm.auto import tqdm
import wandb

from prefgen.methods.sampling.utils import LatentAttributeSampler
from prefgen.methods.sampling.langevin_dynamics.energy_functions import compute_continuous_conditional_energy
from prefgen.methods.attribute_classifiers.face_identification.id_loss import IDLoss
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier

class LangevinDynamicsSampler(LatentAttributeSampler):
    """
        Implements Conditional Langevin Dynamics Sampling 
        from the GAN latent space. 
    """

    def __init__(
        self, 
        attribute_names: list = None,
        generator=None, 
        classifier=None,
        energy_function=compute_continuous_conditional_energy, 
        device="cuda",
        sgld_lr=0.01,
        sgld_std=0.01,
        x_diff_regularization=0.00, 
        z_diff_regularization=0.00, 
        id_loss_multiplier=0.00,
        grad_clipping=None,
        lr_decay=1.0,
        annealed_ld=False,
        max_sgld_std=0.1,
        min_sgld_std=1e-4,
        diff_regularization=None,
        wandb_group_name="Langevin Dynamics Sampling",
        args_to_log=Namespace()
    ):
        self.generator = generator
        self.attribute_names = attribute_names
        # TODO generalize this to other classifiers
        if classifier is None:
            assert not self.attribute_names is None
            self.classifier = load_ffhq_wspace_classifier(
                self.generator.map_z_to_w,
                self.attribute_names,
                device=device
            )
        else:
            self.classifier = classifier
        self.energy_function = energy_function
        self.id_loss = IDLoss().to(device)
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.lr_decay = lr_decay
        self.x_diff_regularization = x_diff_regularization
        self.z_diff_regularization = z_diff_regularization
        self.id_loss_multiplier = id_loss_multiplier
        self.grad_clipping = grad_clipping
        self.annealed_ld = annealed_ld
        self.max_sgld_std = max_sgld_std
        self.min_sgld_std = min_sgld_std
        self.wandb_group_name = wandb_group_name
        self.args_to_log = args_to_log

    def sample_uniformly_from_attribute_space(
        num_points=2000,
        save_path=None
    ):
        """
            Uniformly sample random images and embed them in 
            attribute space. 
        """
        attribute_points = []
        # Iterate for num_points
        for point_number in range(num_points):
            with torch.no_grad():
                # Generate a random image
                random_latent = self.generator.randomly_sample_latent()
                # Apply attribute classifier
                attribute_vector = self.classifier(latent=random_latent)
                attribute_points.append(attribute_vector)

        attribute_points = torch.stack(attribute_points)

        return attribute_points.detach().cpu().numpy()
    
    def compute_combined_loss(
        self,
        initial_latent, 
        current_latent,
        input_image,
        target_attributes,
        verbose_logging=False,
        iteration_num=1,
        w_space_latent=False,
        extended_latent=False
    ):
        # Evaluate classifiers
        classifier_out = self.classifier(latent=current_latent)
        # Compute attribute space distance
        attribute_distance = torch.linalg.norm(classifier_out - target_attributes)
        # Compute energy function
        energy_out = self.energy_function(
            current_latent, 
            classifier_out, 
            target_attributes
        )
        energy_out = energy_out.sum()
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
        # Combine losses
        loss = energy_out + \
            self.id_loss_multiplier * id_loss + \
            self.z_diff_regularization * z_difference + \
            self.x_diff_regularization * x_difference

        # Perform logging
        if verbose_logging and iteration_num % 10 == 0:
            # Wandb logging
            wandb.log({
                "Energy Score": energy_out.detach().cpu().numpy(),
                "Identity Loss": id_loss.detach().cpu().numpy(),
            #    "Identity Loss Grad Mag": id_grad_mag.detach().cpu().numpy(),
            #    "Energy Loss Grad Mag": energy_grad_mag.detach().cpu().numpy(),
                "X Difference Loss": x_difference.detach().cpu().numpy(),
                "Z Difference Loss": z_difference.detach().cpu().numpy(),
                "Target Attributes": target_attributes.detach().cpu().numpy(),
                "Classifier out": str(classifier_out),
                "Combined Loss": loss.detach().cpu().numpy(),
                "Current Image": wandb.Image(current_image),
                "Current Latent Magnitude": torch.norm(current_latent).item(),
                "Start Latent Magnitude": torch.norm(initial_latent).item(),
                "Attribute Distance": attribute_distance.detach().cpu().numpy(),
                "Iteration": iteration_num
            })
        
        return loss

    def sample_regular_langevin_dynamics(
        self, 
        initial_latent, 
        input_image, 
        target_attributes,
        num_steps=100, 
        w_space_latent=False, 
        verbose_logging=False,
        extended_latent=False
    ):
        """
            Samples using langevin dynamics with a constant
            learning rate and amount of noise.
        """
        # Ensure gradients are required
        with torch.enable_grad():
            if isinstance(target_attributes, np.ndarray):
                target_attributes = torch.Tensor(target_attributes).cuda()
                target_attributes.requries_grad = True
            if isinstance(initial_latent, np.ndarray):
                initial_latent = torch.Tensor(initial_latent).cuda()
                initial_latent.requires_grad = True

            current_latent = initial_latent.clone()
            learning_rate = self.sgld_lr
            # Run langevin dynamics
            print("Running Langevin Dynamics")
            for iteration_num in tqdm(range(num_steps), position=0, leave=True):
                # Decay learning rate
                learning_rate *= self.lr_decay
                # Compute loss
                loss = self.compute_combined_loss(
                    initial_latent, 
                    current_latent,
                    input_image,
                    target_attributes,
                    verbose_logging=verbose_logging,
                    iteration_num=iteration_num,
                    w_space_latent=w_space_latent,
                    extended_latent=extended_latent
                )         
                # Compute the gradient of the energy
                grad = torch.autograd.grad(-1*loss, [current_latent])[0]
                # Perform grad clipping
                if not self.grad_clipping is None:
                    if torch.linalg.norm(grad) > self.grad_clipping:
                        grad = grad / torch.linalg.norm(grad)
                        grad = self.grad_clipping * grad
                # Generate Gaussian noise
                gaussian_noise = torch.randn_like(current_latent)
                # Compute next chain value
                current_latent = current_latent + \
                                    learning_rate * grad + \
                                    self.sgld_std * gaussian_noise
                # Perform gradient step
                current_latent.grad = None
                initial_latent.grad = None
                input_image.grad = None
                target_attributes.grad = None
                loss.grad = None

        final_latent = current_latent.data.detach()
        loss = loss.detach()

        return final_latent, loss

    def sample(
        self, 
        target_attributes, 
        initial_latent=None, 
        num_steps=100, 
        verbose_logging=False, 
        best_of_n_chains=1, 
        w_space_latent=False, 
        extended_latent=False,
        target_latent=None
    ):
        """
            Samples a vector in StyleGAN2 latent space 
            using Stochastic Gradient Langevin Dymamics. 
        """
        # Initialize latent
        if initial_latent is None:
            # Randomly sample a latent from Gaussian
            initial_latent = self.generator.randomly_sample_latent(
                w_space_latent=w_space_latent, 
                extended_latent=extended_latent
            )
        # Generate an input image
        _, input_image = self.generator.generate_image(
            initial_latent, 
            w_space_latent=w_space_latent,
            extended_latent=extended_latent
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
                    config=self.args_to_log,
                #    settings=wandb.Settings(start_method='fork')
                )
            # Log target image
            if not target_latent is None:
                _, target_attribute_image = self.generator.generate_image(
                    target_latent, 
                    w_space_latent=w_space_latent
                )
                if verbose_logging:
                    wandb.log({
                        "Target attribute image": wandb.Image(target_attribute_image),
                    })
            if verbose_logging:
                wandb.log({
                    "Start Image": wandb.Image(input_image)
                })
            # Choose which algorithm to use
            final_latent, final_loss = self.sample_regular_langevin_dynamics(
                initial_latent, 
                input_image, 
                target_attributes,
                num_steps=num_steps,
                w_space_latent=w_space_latent,
                extended_latent=extended_latent,
                verbose_logging=verbose_logging
            )
            # Update best
            if final_loss < best_latent_loss:
                best_latent = final_latent
                best_latent_loss = final_loss
            # End wandb run
            if verbose_logging:
                wandb.finish()

        return initial_latent, final_latent

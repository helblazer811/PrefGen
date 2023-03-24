import wandb 


class AnnealedLangevinDynamicsSampler():

    def sample_annealed_langevin_dynamics(self, initial_latent, input_image, target_attributes,
                                            num_steps=100, verbose_logging=True, num_phases=10):
        """
            Samples using langevin dynamics using an iterative procedure with 
            multiple levels of noise.  

            This procedure is from the paper: 
            "Generative Modeling by Estimating Gradients of theData Distribution"

            For num_phases:
                Get step size
                For num_steps:
                    sample latent
        """
        def compute_annealing_step_size_schedule(num_phases=1):
            """
                Compute the step size schedule for annealing
            """
            # Compute the noise schedule as a geometric sequence 
            ratio = (self.min_sgld_std / self.max_sgld_std) ** (1 / num_steps)
            std_sizes = [self.max_sgld_std * ratio ** time for time in range(num_steps)]
            # Compute the step sizes using the std_sizes
            step_sizes = []
            fixed_learning_rate = self.sgld_lr # Initial step size
            for step_index in range(num_phases):
                # Update the step size
                step_size = fixed_learning_rate * (std_sizes[step_index] ** 2) / (self.min_sgld_std ** 2)
                step_sizes.append(step_size)

            return step_sizes

        def compute_linear_annealing_step_size_schedule(num_phases=10):
            step_sizes = []
            for phase in range(num_phases):
                step_size = self.max_sgld_std - (self.max_sgld_std - self.min_sgld_std) / num_phases * phase
                step_sizes.append(step_size)

            return step_sizes

        # Ensure gradients are required
        with torch.enable_grad():
            # Run langevin dynamics
            print("Running Langevin Dynamics")
            current_latent = torch.autograd.Variable(initial_latent.clone(), requires_grad=True)
            # Compute step sizes
            # step_sizes = compute_annealing_step_size_schedule(num_phases=num_phases)
            step_sizes = compute_linear_annealing_step_size_schedule()
            # Go through the outer loop of each step size
            for step_size_index, step_size in enumerate(step_sizes):
                for iteration_num in tqdm(range(num_steps), position=0, leave=True):
                    # Evaluate classifiers
                    classifier_out = self.classifier(current_latent)
                    assert classifier_out.requires_grad
                    # Compute energy function
                    energy_out = self.energy_function(
                        current_latent, 
                        classifier_out, 
                        target_attributes
                    )
                    energy_out = energy_out.sum()
                    # Regularize the differnece between start and current images
                    z_difference = torch.linalg.norm(current_latent - initial_latent) ** 2
                    _, current_image = self.generator.generate_image(current_latent)
                    x_difference = torch.linalg.norm(current_image - input_image) ** 2
                    # Compute identity loss
                    id_loss = self.id_loss(input_image, current_image)
                    # Combine losses
                    loss = energy_out + \
                        self.id_loss_multiplier * id_loss + \
                        self.z_diff_regularization * z_difference + \
                        self.x_diff_regularization * x_difference 
                    # Compute the gradient of the energy
                    grad = torch.autograd.grad(-1*loss, [current_latent])[0]
                    # Generate Gaussian noise
                    gaussian_noise = torch.randn_like(grad)
                    # Compute next chain value
                    current_latent = current_latent + \
                                        (step_size / 2) * grad + \
                                        (step_size ** (1/2)) * gaussian_noise
                    # Perform logging
                    if verbose_logging and iteration_num % 10 == 0:
                        # Wandb logging
                        wandb.log({
                            "Energy Score": energy_out.detach().cpu().numpy(),
                            "Identity Loss": id_loss.detach().cpu().numpy(),
                            "X Difference Loss": x_difference.detach().cpu().numpy(),
                            "Z Difference Loss": z_difference.detach().cpu().numpy(),
                            "Classifier out": str(classifier_out),
                            "Combined Loss": loss.detach().cpu().numpy(),
                            "Current Image": wandb.Image(current_image),
                            "Iteration": iteration_num,
                            "Step size": step_size
                        })

        final_latent = current_latent.data.detach()
        return final_latent, loss
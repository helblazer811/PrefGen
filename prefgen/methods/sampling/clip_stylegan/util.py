from argparse import Namespace
import torch
from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.sampling.clip_stylegan.clip_attribute_classifier import CLIPAttributeClassifier
from prefgen.methods.sampling.gradient_descent.adam_sampler import AdamSampler
from prefgen.methods.sampling.gradient_descent.sampler import GradientDescentSampler

def sample_clip_attribute(
    start_latent, 
    attribute_val, 
    generator=None,
    attribute_prompts=[
        "a person",
        "an angry person",
        "a very angry person",
    ],
    attribute_classifier=None,
    use_adam=False,
    verbose_logging=False,
    gradient_descent_args=Namespace(
        learning_rate=1.0,
        id_loss_multiplier=0.1,
        x_diff_regularization=0.0,
        z_diff_regularization=0.0,
        w_space_latent=True,
        extended_latent=False,
        num_steps=100
    )
):
    # Make a CLIP Attribute Classifier form the attribute prompts
    if attribute_classifier is None:
        attribute_classifier = CLIPAttributeClassifier(
            generator=generator,
            attribute_prompts=attribute_prompts,
        )
    if generator is None:
        # Make Generator
        generator = StyleGAN2Wrapper()
    # Perform gradient descent in attribute space to optimize attribute val
    if not use_adam:
        sampler = GradientDescentSampler(
            attribute_classifier, 
            generator,
            learning_rate=gradient_descent_args.learning_rate,
            x_diff_regularization=gradient_descent_args.x_diff_regularization,
            z_diff_regularization=gradient_descent_args.z_diff_regularization,
            id_loss_multiplier=gradient_descent_args.id_loss_multiplier,
            wandb_group_name="February CLIP GD"
        )
    else:
        sampler = AdamSampler(
            attribute_classifier, 
            generator,
            learning_rate=gradient_descent_args.learning_rate,
            x_diff_regularization=gradient_descent_args.x_diff_regularization,
            z_diff_regularization=gradient_descent_args.z_diff_regularization,
            id_loss_multiplier=gradient_descent_args.id_loss_multiplier,
            wandb_group_name="February CLIP GD",
        )
    # Run Sampling
    if not isinstance(attribute_val, torch.Tensor):
        attribute_val = torch.Tensor([attribute_val]).cuda()
    # print(f"Attribute val: {attribute_val.shape}")
    # print(f"Start latent: {start_latent.shape}")
    _, final_latent = sampler.sample(
        attribute_val,
        initial_latent=start_latent, # Random initial latent
        num_steps=gradient_descent_args.num_steps, 
        verbose_logging=verbose_logging,
        w_space_latent=gradient_descent_args.w_space_latent,
        extended_latent=gradient_descent_args.extended_latent,
    )
    # Construct the final image
    modified_latent, modified_image = generator.generate_image(
        final_latent, 
        w_space_latent=gradient_descent_args.w_space_latent,
        extended_latent=gradient_descent_args.extended_latent
    )

    return modified_latent.detach().cpu(), modified_image.detach().cpu()


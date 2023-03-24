import argparse
import math
import os

import torch
import wandb
import torchvision
from torch import optim
from tqdm import tqdm

import clip
from prefgen.methods.attribute_classifiers.face_identification.id_loss import IDLoss

class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]
STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

class AdamSampler():

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
        num_steps=100,
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
        self.num_steps = num_steps

    def sample(
        self, 
        target_attributes, 
        initial_latent=None, 
        num_steps=None, 
        latent_dim=512, 
        verbose_logging=True, 
        best_of_n_chains=1, 
        w_space_latent=True, 
        learning_rate_decay=0.99,
        target_latent=None, 
        extended_latent=False, 
        noise_std=0.00,
        truncation=0.7,
    ):
        # print(f"Target attributes: {target_attributes.shape}")
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

        if num_steps is None:
            num_steps = self.num_steps
        
        with torch.enable_grad():
            # Initialize latent
            if initial_latent is None:
                # Randomly sample a latent from Gaussian
                if len(target_attributes.shape) == 2:
                    num_examples = target_attributes.shape[0]
                    initial_latents = []
                    for i in range(num_examples):
                        initial_latents.append(self.generator.randomly_sample_latent())
                    initial_latent = torch.stack(initial_latents).squeeze()
                else:
                    initial_latent = self.generator.randomly_sample_latent()
                # print(f"Randomly sampled latent: {initial_latent.shape}")
            # Generate an input image
            _, input_image = self.generator.generate_image(
                initial_latent, 
            )
            input_image = input_image.detach().clone()
            input_image.requires_grad = True
            initial_latent = initial_latent.detach().clone()
            initial_latent.requires_grad = True
            current_latent = initial_latent.detach().clone()
            current_latent.requires_grad = True
            target_attributes = target_attributes.detach().clone()
            target_attributes.requires_grad = True
            # Split up current latent into a bunch of latents
            # current_latent = list(torch.unbind(current_latent, dim=0))
            # current_latent = [torch.Tensor(latent.detach().cpu().numpy()).cuda() for latent in current_latent]
            # Optimizer
            optimizer = optim.Adam([current_latent], lr=self.learning_rate)
            # Log the initial image
            if verbose_logging:
                wandb.log({
                    "Initial Image": wandb.Image(input_image),
                })
            # Run gradient descent
            # print("Running Gradient Descent")
            for iteration_num in tqdm(range(num_steps), position=0, leave=True):
                current_latent, current_image = self.generator.generate_image(
                    current_latent
                )
                classifier_out = self.classifier(
                    image=current_image,
                )
                # print(f"Classifier output: {classifier_out.shape}")
                # Compute MSE in attribute space
                attribute_mse = torch.linalg.norm(classifier_out - target_attributes, dim=-1) ** 2
                # print(f"Attribute MSE: {attribute_mse.shape}")
                # input_image = input_image.detach().clone()
                # input_image.requires_grad = True
                id_loss = self.id_loss(input_image, current_image)
                # print(f"ID loss: {id_loss.shape}")
                # Regularize the differnece between start and current images
                # initial_latent = initial_latent.detach().clone()
                # initial_latent.requires_grad = True
                if len(initial_latent.shape) > 2:
                    z_difference = ((current_latent - initial_latent) ** 2).sum((-1, -2))
                else:
                    z_difference = ((current_latent - initial_latent) ** 2).sum()
                # print(f"Z difference: {z_difference.shape}")
                # input_image = input_image.detach().clone()
                # input_image.requires_grad = True
                # print(f"Current image: {current_image.shape}")
                # print(f"Input image: {input_image.shape}")
                if not self.x_diff_regularization == 0.0:
                    if len(input_image.shape) > 2:
                        x_difference = torch.sum(current_image - input_image, dim=(-1, -2, -3)) ** 2
                    else:
                        x_difference = torch.sum(current_image - input_image, dim=-1) ** 2
                else:
                    x_difference = torch.zeros_like(z_difference).cuda()
                # print(f"X difference: {x_difference.shape}")
                # Combine losses
                loss = attribute_mse + \
                    self.z_diff_regularization * z_difference + \
                    self.x_diff_regularization * x_difference + \
                    self.id_loss_multiplier * id_loss
                
                # print(f"Loss: {loss.shape}")
                #for image_index in range(loss.shape[0]):
                # Independently backpropagate each of the losses
                optimizer.zero_grad()
                loss.backward(
                    gradient=torch.ones_like(loss).cuda(),
                )
                #for loss_index in range(len(loss) - 1):
                #    loss[loss_index].backward(retain_graph=True)
                #    #loss[loss_index] = loss[loss_index].detach()
                #    optimizer.step()
                #loss[-1].backward()
                optimizer.step()
                #loss = loss.mean()
                #loss.backward()
                #print(loss)
                #loss.backward(
                #    gradient=torch.ones_like(loss).cuda(),
                #)

                #loss[0].backward(
                #    gradient=torch.ones_like(loss).cuda(), 
                #    inputs=[current_latent[0]]
                #)
                #print(current_latent[0].grad)
                #print(current_latent[1].grad)

                """
                grad = torch.autograd.grad(-1*loss, [current_latent])[0]
                # Compute next chain value
                current_latent = current_latent + \
                                    learning_rate * grad
                """

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
                """
                current_latent.grad = None
                initial_latent.grad = None
                input_image.grad = None
                target_attributes.grad = None
                loss.grad = None
                """

        if verbose_logging:
            run.finish()

        return initial_latent, current_latent

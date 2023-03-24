import os
import torch # type: ignore
import sys
import numpy as np

sys.path.append(
    os.path.join(
        os.environ["PREFGEN_ROOT"],
        "prefgen/methods/sampling/gan_control/gan_control/src"
    )
)

from gan_control.inference.inference import Inference

stylegan_pth = os.path.join(
    os.environ["PREFGEN_ROOT"], 
    "prefgen/pretrained/stylegan2_gan_control/controller_dir/generator/checkpoint/best_fid.pt"
)

controller_path = os.path.join(
    os.environ["PREFGEN_ROOT"],
    "prefgen/pretrained/stylegan2_gan_control/controller_dir"
)
model_dir = os.path.join(
    os.environ["PREFGEN_ROOT"],
    "prefgen/pretrained/stylegan2_gan_control/controller_dir/generator"
)

class StyleGAN2GANControlWrapper():
    """
        Wrapper for the StyleGAN2 Generator form GAN Control 
    """

    def __init__(
        self, 
        w_space_latent=True, 
        extended_latent=False, 
        rescale=False,
        device="cuda",
        image_size=1024,
        latent_dim=512,
        num_mlp=8,
    ):
        self.w_space_latent = w_space_latent
        self.extended_latent = extended_latent
        self.rescale = rescale
        self.latent_dim = latent_dim
        # Load the generator
        self.generator, _, _, _ = Inference.retrieve_model(
            model_dir
        )
        self.generator = self.generator.module.to(device)
        self.mean_w = self.generator.mean_latent(n_latent=2048)
        self.device = device
        """
        self.generator = Generator(
            size=image_size, 
            style_dim=latent_dim, 
            n_mlp=num_mlp
        ).to(device)
        self.generator.load_state_dict(torch.load(stylegan_pth))
        """

    def randomly_sample_latent(
        self, 
        device="cuda", 
        w_space_latent=None, 
        extended_latent=None
    ):
        """Randomly sample a latent vector"""
        z_init = torch.FloatTensor(
            1, 
            self.latent_dim
        ).normal_(0, 1).to(device)
        z_init.requires_grad = True
        
        if extended_latent is None:
            extended_latent = self.extended_latent
        if w_space_latent is None:
            w_space_latent = self.w_space_latent

        if extended_latent:
            w_vectors = self.map_z_to_w(z_init)
            w_vectors = w_vectors.unsqueeze(1).repeat([1, 18, 1])  # bs x n_mappings x 512
            return w_vectors

        if w_space_latent:
            return self.map_z_to_w(z_init)
        else:
            return z_init

    def map_z_to_w(self, z, truncation=0.7):
        """Map z to w"""
        assert z.shape[-1] == self.generator.style_dim, z.shape
        w = self.generator.style(z)  # no truncation

        if truncation < 1:
            truncation_latent = self.mean_w.to(z.device)
            w = truncation_latent + truncation * (w - truncation_latent)

        return w

    def generate_image(
        self, 
        latent=None, 
        w_space_latent=None, 
        extended_latent=None,
    ):
        """Decode a latent vector into an image"""
        if w_space_latent is None:
            w_space_latent = self.w_space_latent
        if extended_latent is None:
            extended_latent = self.extended_latent
            
        if latent is None:
            latent = self.randomly_sample_latent() # latent z vector
        elif isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent).to(self.device)

        if not extended_latent: 
            if not w_space_latent:
                w_vectors = self.map_z_to_w(latent)
            else:
                w_vectors = latent

            if w_vectors.ndim == 2:
                w_vectors = w_vectors.unsqueeze(1).repeat([1, 18, 1])  # bs x n_mappings x 512
        else:
            w_vectors = latent
            
        image, _ = self.generator(
            [w_vectors], 
            randomize_noise=False, 
            input_is_latent=True
        )
        # Rescale image
        if self.rescale:
            image = (image - torch.min(image)) / torch.max(image - torch.min(image))

        return latent, image

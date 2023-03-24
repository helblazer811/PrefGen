import argparse
import torch
import os
import time
from PIL import Image
from torchvision import transforms
from prefgen.methods.encoder.encoder4editing.dataset import E4EDataset

from prefgen.methods.plotting.sampling import plot_initial_and_modified
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler
from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--attribute_names", type=str, default=["yaw", "pitch", "roll", "age"])
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--extended_latent", type=bool, default=True)
    parser.add_argument("--w_space_latent", type=bool, default=True)
    parser.add_argument("--image_path", type=str, default="base_images/LeoDicaprio.png")
    parser.add_argument("--target_attributes", default=None, type=torch.Tensor, nargs="+", help="Target attributes to sample", required=False)

    parser.add_argument("--plot_save_path", type=str, default="plots/gan_control_samples.png")
    parser.add_argument(
        "--inversion_checkpoint", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/pretrained/stylegan2_e4e_inversion/e4e_ffhq_encode.pt"
        )
    )

    args = parser.parse_args()
    # Load the inversion model
    # net, opts = setup_model(args.inversion_checkpoint, "cuda")
    # Make a Gan Control Stylegan model
    generator = StyleGAN2GANControlWrapper(
        rescale=False,
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent
    )
    # Make a GANControlSampler object
    sampler = GANControlSampler(
        generator,
        attribute_names=args.attribute_names,
    )
    # Get an inversion from the dataset
    e4e_dataset = E4EDataset()
    # Compute base image gan latent
    # initial_latent = torch.from_numpy(initial_latent).unsqueeze(0).cuda()
    # Make a list of images
    initial_images = []
    modified_images = []
    for image_index in range(args.num_images):
        with torch.no_grad():
            initial_image, initial_latent = e4e_dataset[image_index]
            initial_latent = initial_latent.unsqueeze(0)
            # Sample an image
            modified_image, initial_latent, modified_latent, _ = sampler.sample(
                target_attributes=args.target_attributes,
                initial_latent=initial_latent
            )
            _, initial_image = generator.generate_image(initial_latent)
            # Add to list
            initial_images.append(initial_image)
            modified_images.append(modified_image)
    # Sample images and save in matplotlib figure
    plot_initial_and_modified(
        initial_images, 
        modified_images,
        plot_save_path=args.plot_save_path
    )
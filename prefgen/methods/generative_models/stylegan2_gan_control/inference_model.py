import os
import torch # type: ignore
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(
    os.path.join(
        os.environ["PREFGEN_ROOT"],
        "prefgen/external_modules/gan_control/src"
    )
)

from gan_control.inference.controller import Controller

CONTROLLER_DIR = os.path.join(
    os.environ["PREFGEN_ROOT"],
    "prefgen/pretrained/stylegan2_gan_control/controller_dir"
)

RESOURCES_DIR = os.path.join(
    os.environ["PREFGEN_ROOT"],
    "prefgen/external_modules/gan_control/resources"
)

if __name__ == "__main__":
    controller = Controller(CONTROLLER_DIR)

    # Loading extracted attributes df
    attributes_df = pd.read_pickle(
        os.path.join(
            RESOURCES_DIR,
            'ffhq_1K_attributes_samples_df.pkl'
        )
    )
    # Generate initial latent
    batch_size = 16
    truncation = 0.7
    resize = 480
    initial_image_tensors, initial_latent_z, initial_latent_w = controller.gen_batch(
        batch_size=batch_size, 
        truncation=truncation
    )
    grid = controller.make_resized_grid_image(
        initial_image_tensors, 
        resize=None, 
        nrow=8
    )
    grid.save("initial_images.png") 
    controller.make_resized_grid_image(initial_image_tensors, resize=resize, nrow=8)
    # Some chosen expressions (most expression are smilles)
    expressions = attributes_df.expression3d.to_list()
    expression_0 = torch.tensor([expressions[2]])
    expression_1 = torch.tensor([expressions[10]])
    expression_2 = torch.tensor([expressions[16]])
    expression_3 = torch.tensor([expressions[39]])
    # print(expression_0)
    image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(
        latent=initial_latent_w, 
        input_is_latent=True, 
        expression=expression_0
    )
    grid = controller.make_resized_grid_image(
        image_tensors, 
        resize=None, 
        nrow=8
    )
    
    grid.save("modified_images.png") 


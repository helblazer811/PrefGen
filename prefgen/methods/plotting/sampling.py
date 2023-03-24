"""
    Plotting for langevin dynamics
"""

import matplotlib.pyplot as plt
from prefgen.methods.plotting.utils import convert_stylegan_image_to_matplotlib, plot_ignore_exceptions
import numpy as np
import torch

@plot_ignore_exceptions
def generate_image_grid_plot(
    images, 
    plot_save_path
):
    """
        Generates and saves a grid of images in at the given path 
    """
    num_images = len(images)
    grid_width = int(np.sqrt(num_images))
    fig, axs = plt.subplots(grid_width, grid_width, dpi=300) 
    fig.add_gridspec(grid_width, grid_width, wspace=0, hspace=0)
    
    for x_ind in range(grid_width):
        for y_ind in range(grid_width):
            image = images[x_ind * grid_width + y_ind]
            image = convert_stylegan_image_to_matplotlib(image)
            axs[y_ind, x_ind].imshow(image)
            axs[y_ind, x_ind].set_xticks([])
            axs[y_ind, x_ind].set_yticks([])

    fig.tight_layout()

    assert not plot_save_path is None

    plt.savefig(plot_save_path)

@plot_ignore_exceptions
def plot_initial_and_modified(
    initial_images, 
    modified_images, 
    plot_save_path=None
):
    """
        Plot and save initial image and attribute
        modified image. 
    """

    if isinstance(initial_images, list):
        assert len(initial_images) == len(modified_images)
        fig, axs = plt.subplots(len(initial_images), 2, figsize=(2, len(initial_images)), dpi=200) 

        for row in axs:
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_aspect('equal')

        axs[0, 0].set_title("Initial Image")
        axs[0, 1].set_title("Modified Image")

        for pair_index in range(len(initial_images)):
            initial_image = initial_images[pair_index]
            modified_image = modified_images[pair_index]
            initial_image = convert_stylegan_image_to_matplotlib(initial_image)
            modified_image = convert_stylegan_image_to_matplotlib(modified_image)

            axs[pair_index, 0].imshow(initial_image)
            axs[pair_index, 1].imshow(modified_image)

        assert not plot_save_path is None

        plt.savefig(plot_save_path)
    else:
        initial_image = initial_images
        modified_image = modified_images
        initial_image = convert_stylegan_image_to_matplotlib(initial_image)
        modified_image = convert_stylegan_image_to_matplotlib(modified_image)

        fig, axs = plt.subplots(1, 2) 
        axs[0].imshow(initial_image)
        axs[1].imshow(modified_image)

        axs[0].set_xlabel("Initial Image")
        axs[1].set_xlabel("Modified Image")

        assert not plot_save_path is None

        plt.savefig(plot_save_path)
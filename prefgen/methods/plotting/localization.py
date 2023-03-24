"""
Plot information about localization of a trial. 
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
import matplotlib
import os
plt.rcParams["font.family"] = "Times New Roman"

from prefgen.methods.localization.simulation_data import LocalizationSimulationData
from prefgen.methods.plotting.utils import convert_stylegan_image_to_matplotlib, plot_ignore_exceptions
from moviepy.editor import ImageSequenceClip
import wandb
import torch
import numpy as np

def append_id(filename, id):
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + [id])

################################## Metrics #####################################
@plot_ignore_exceptions
def compute_id_rank_loss_over_time(
    generator, 
    id_loss, 
    simulation_data, 
    num_samples=500
):
    """
        Computes the percentage of face identities that are CLOSER
        from the initial identity then the current latent vector. 
        We want this quantity to be low. 
    """
    _, initial_image = generator.generate_image(simulation_data.start_latent)
    # Compute a bunch of identity losses
    identity_losses = []
    for image_index in range(num_samples):
        latent  = generator.randomly_sample_latent()
        _, image = generator.generate_image(latent)
        identity_loss = id_loss(initial_image, image).item()
        identity_losses.append(identity_loss)
    identity_losses = np.array(identity_losses)
    identity_losses = np.sort(identity_losses) # Sort the losses
    # print("id losses")
    # print(identity_losses)
    # Compute each individual loss
    id_losses_over_time = []
    for latent in simulation_data.latents_over_time:
        latent = torch.Tensor(latent).to("cuda")
        _, current_image = generator.generate_image(latent)
        # Compute ID Loss
        id_loss_val = id_loss(initial_image, current_image)
        id_loss_val = id_loss_val.item()
        # Measure how the loss ranks in 
        index = np.searchsorted(identity_losses, id_loss_val)
        percentile = index / len(identity_losses) * 100
        # Save it
        assert not percentile is None
        id_losses_over_time.append(percentile)

    return id_losses_over_time

@plot_ignore_exceptions
def compute_attribute_rank_loss(
    latent_sampler,
    current_attributes,
    target_attributes,
    num_samples=500, 
    attribute_samples=None
):
    """
        Computes ranking loss for attribute distance 
    """
    # Compute a bunch of attribute losses
    if attribute_samples is None:
        attribute_samples = latent_sampler.randomly_sample_attributes(
            num_samples,
            return_dict=False
        ).detach().cpu().numpy()
    attribute_losses = []
    for sample_index in range(num_samples):
        attribute = attribute_samples[sample_index]
        attribute_loss = np.linalg.norm(target_attributes - attribute)
        attribute_loss = attribute_loss.item()
        attribute_losses.append(attribute_loss)
    attribute_losses = np.array(attribute_losses)
    attribute_losses = np.sort(attribute_losses) # Sort the losses
    # Compute attribute loss
    attribute_loss_val = np.linalg.norm(target_attributes - current_attributes)
    attribute_loss_val = attribute_loss_val.item()
    # Measure how the loss ranks in 
    index = np.searchsorted(attribute_losses, attribute_loss_val)
    percentile = index / len(attribute_losses) * 100

    return percentile

@plot_ignore_exceptions
def compute_attribute_rank_loss_over_time(
    sampler,
    simulation_data, 
    num_samples=500
):
    """
        Computes the percentage of faces with attribute vectors 
        CLOSER to the initial attribute vector than the current 
        attribute vector. 
    """
    # Get the initial attribute vector
    target_attribute = simulation_data.target_attributes
    # Generate a bunch of attribute samplers
    attribute_samples = sampler.randomly_sample_attributes(
        num_samples,
        return_dict=False
    ).detach().cpu().numpy()
    # Compute a bunch of attribute losses
    attribute_losses = []
    for sample_index in range(num_samples):
        attribute = attribute_samples[sample_index]
        attribute_loss = np.linalg.norm(target_attribute - attribute)
        attribute_loss = attribute_loss.item()
        attribute_losses.append(attribute_loss)
    attribute_losses = np.array(attribute_losses)
    attribute_losses = np.sort(attribute_losses) # Sort the losses
    # Compute the losses over time
    attribute_dist_over_time = []
    for current_attribute in simulation_data.attributes_over_time:
        # Compute attribute loss
        attribute_loss_val = np.linalg.norm(target_attribute - current_attribute)
        attribute_loss_val = attribute_loss_val.item()
        # Measure how the loss ranks in 
        index = np.searchsorted(attribute_losses, attribute_loss_val)
        percentile = index / len(attribute_losses) * 100
        # Save it
        attribute_dist_over_time.append(percentile)
    return attribute_dist_over_time

################################## Plots ########################################

@plot_ignore_exceptions
def save_localization_gif(generator, simulation_data: LocalizationSimulationData, 
                            save_path="", device="cuda", start_end_frames=10):
    """
        Saves a gif of the images samples from start_latent through
        all latents over time.  
    """
    current_latent = simulation_data.start_latent
    frames = []
    # Add duplicate start frames at beggining
    for frame_number in range(start_end_frames):
        latent_vector = simulation_data.start_latent
        _, image = generator.generate_image(torch.Tensor(latent_vector).cuda())
        image = convert_stylegan_image_to_matplotlib(image)
        frames.append(image)
    # Make a frame for each latent vector
    for latent_vector in simulation_data.latents_over_time:
        latent_vector = latent_vector
        _, image = generator.generate_image(torch.Tensor(latent_vector).cuda())
        image = convert_stylegan_image_to_matplotlib(image)
        frames.append(image)
    # Add duplicate end frames at end
    for frame_number in range(start_end_frames):
        latent_vector = simulation_data.latents_over_time[-1]
        _, image = generator.generate_image(torch.Tensor(latent_vector).cuda())
        image = convert_stylegan_image_to_matplotlib(image)
        frames.append(image)
    # save it as a gif
    clip = ImageSequenceClip(frames, fps=5)
    clip.write_gif(save_path, fps=5)

@plot_ignore_exceptions
def plot_localization_attribute_dist_over_time(
    latent_sampler,
    simulation_data: LocalizationSimulationData, 
    save_path=None
):
    """
        Plots various losses for a given localization run over time. 
    """
    if not isinstance(simulation_data, list):
        simulation_data = [simulation_data]
    # Use the seaborn setting
    # Compute the losses
    att_dists = []
    for simulation_trial in simulation_data:
        attribute_dist_over_time = compute_attribute_rank_loss_over_time(
            latent_sampler,
            simulation_trial,
        )
        att_dists.append(np.array(attribute_dist_over_time))
    att_dists = np.array(att_dists)
    # Compute the mean and variances
    att_dist_mean = np.mean(att_dists, axis=0)
    att_dist_std = np.std(att_dists, axis=0)
    # Plot metrics on axis
    fig = plt.figure()
    fig, axs = plt.subplots(1, 1, dpi=400, figsize=(3.5, 3.5))
    plt.title("Localization Over Time")
    query_indices = np.arange(0, len(attribute_dist_over_time))
    axs.plot(
        query_indices, 
        att_dist_mean, 
        color="blue", 
        label="Attribute Distance"
    )
    axs.fill_between(
        query_indices, 
        att_dist_mean - att_dist_std, 
        att_dist_mean + att_dist_std, 
        color="blue", 
        alpha=0.3
    )
    # axs.set_ylim(0, 100)
    axs.set_ylabel("Percentage Closer to Target")
    axs.set_xlabel("Number of Queries")
    plt.legend()
    # Save the plot
    if not save_path is None:
        plt.savefig(save_path, bbox_inches= "tight")
        plt.show()

@plot_ignore_exceptions
def plot_localization_loss_over_time(
    generator, 
    latent_sampler,
    id_loss,
    simulation_data: LocalizationSimulationData, 
    save_path=None,
    add_start_latent_first=False
):
    """
        Plots various losses for a given localization run over time. 
    """
    if not isinstance(simulation_data, list):
        simulation_data = [simulation_data]
    # Compute mean attribute vector
    attribute_vectors = latent_sampler.randomly_sample_attributes(
        num_samples=1000,
        return_dict=False
    ).detach().cpu().numpy()
    mean_attribute_vector = np.mean(attribute_vectors, 0)
    # Use the seaborn setting
    # Compute the losses
    att_dists = []
    id_dists = []
    for simulation_trial in simulation_data:
        if add_start_latent_first:
            # Add initial latent and mean att vector to begginging of
            simulation_trial.attributes_over_time.insert(0, mean_attribute_vector)
            simulation_trial.latents_over_time.insert(0, simulation_trial.start_latent)

        attribute_dist_over_time = compute_attribute_rank_loss_over_time(
            latent_sampler,
            simulation_trial,
        )
        att_dists.append(np.array(attribute_dist_over_time))
        id_losses_over_time = compute_id_rank_loss_over_time(
            generator,
            id_loss, 
            simulation_trial
        )
        id_dists.append(np.array(id_losses_over_time))
    # Compute the mean and variances
    att_dist_mean = np.mean(att_dists, axis=0)
    att_dist_std = np.std(att_dists, axis=0)
    id_dist_mean = np.mean(id_dists, axis=0)
    id_dist_std = np.std(id_dists, axis=0)
    # Plot metrics on axis
    fig = plt.figure()
    fig, axs = plt.subplots(1, 1, dpi=400, figsize=(3.5, 3.5))
    plt.title("Localization Over Time")
    query_indices = np.arange(0, len(attribute_dist_over_time))
    axs.set_xlim(left=0, right=len(query_indices) - 1)
    axs.plot(
        query_indices, 
        id_dist_mean, 
        color="orange", 
        label="Identity Loss"
    )
    axs.fill_between(
        query_indices, 
        id_dist_mean - id_dist_std, 
        id_dist_mean + id_dist_std, 
        color="orange", 
        alpha=0.3
    )
    axs.plot(
        query_indices, 
        att_dist_mean, 
        color="blue", 
        label="Attribute Distance"
    )
    axs.fill_between(
        query_indices, 
        att_dist_mean - att_dist_std, 
        att_dist_mean + att_dist_std, 
        color="blue", 
        alpha=0.3
    )
    # axs.set_ylim(0, 100)
    axs.set_ylabel("Percentage Closer to Target")
    axs.set_xlabel("Number of Queries")
    plt.legend()
    # Save the plot
    if not save_path is None:
        plt.savefig(save_path, bbox_inches= "tight")
        plt.show()

@plot_ignore_exceptions
def plot_multiple_localization_images(
        generator, 
        sampler, 
        simulation_data_list, 
        every_n_queries=5,
        save_path=None
    ):
    """
         Plots a bunch of localizations where every row is a different trial
    """
    num_queries = len(simulation_data_list[0].queries)
    plot_width = num_queries // every_n_queries + 2 # +2 because start image and target image
    plot_height = len(simulation_data_list)
    if plot_height == 1:
        print("This plot only works for multiple trials")
        return
    # Make the plot
    fig = plt.figure()
    fig, axs = plt.subplots(plot_height, plot_width, dpi=200)
    # plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.0, wspace=0.1)
    fig.suptitle("Localization Over Time Several Trials")
    # Set the titles on the top row
    axs[0, 0].set_title("Start Latent")
    axs[0, -1].set_title("Target Latent")
    for latent_index in range(len(simulation_data_list[0].queries)):
        if latent_index % every_n_queries == 0:
            axs[0, latent_index // every_n_queries + 1].set_title(f"{latent_index}")
    # Sample the images and plot them
    for simulation_number, simulation_data in enumerate(simulation_data_list):
        # Plot the start latent
        _, start_image = generator.generate_image(simulation_data.start_latent)
        start_image = convert_stylegan_image_to_matplotlib(start_image)
        axs[simulation_number, 0].imshow(start_image)
        # Plot the target image
        _, target_latent = sampler.sample(
            simulation_data.target_attributes,
            initial_latent=simulation_data.start_latent
        )
        _, target_image = generator.generate_image(target_latent)
        target_image = convert_stylegan_image_to_matplotlib(target_image)
        axs[simulation_number, -1].imshow(target_image)
        # Plot the latent samples
        latent_vectors = simulation_data.latents_over_time 
        for latent_index, latent_vector in enumerate(latent_vectors):
            if latent_index % every_n_queries == 0:
                _, image = generator.generate_image(latent_vector)
                image = convert_stylegan_image_to_matplotlib(image)
                # Plot the image
                axs[simulation_number, latent_index // every_n_queries + 1].imshow(image)
                sampled_image = generator.generate_image(latent_vector)

    plt.subplots_adjust(wspace=0, hspace=0)
    # Save the plot
    if not save_path is None:
        plt.savefig(save_path)
        plt.show()

@plot_ignore_exceptions
def plot_single_localization_with_queries(
    generator, 
    sampler, 
    simulation_data: LocalizationSimulationData, 
    every_n_queries=5, 
    save_path=None
):
    """
        Plots a single localization trial with images corresponding to queries asked
    """
    queries = simulation_data.queries
    latents_over_time = simulation_data.latents_over_time
    start_latent = simulation_data.start_latent
    target_attributes = simulation_data.target_attributes
    # Plot width
    num_queries = len(queries)
    num_samples = num_queries // every_n_queries
    # Make the overall plot
    fig = plt.figure(
        constrained_layout=True, 
        figsize=(5.5, 2.5)
    )
    subfigs = fig.subfigures(
        nrows=2, 
        ncols=1, 
        height_ratios=[1.0, 1.5]
    )
    # Make the samples and init/target
    axs = subfigs[0].subplots(1, num_samples + 2)
    # subfigs[0].suptitle("Samples", y=1.1)
    # Make the plot
    _, start_image = generator.generate_image(start_latent)
    start_image = convert_stylegan_image_to_matplotlib(start_image)
    axs[0].imshow(start_image)
    axs[0].set_title("Start \nImage")
    # Plot image samples
    for latent_index, latent in enumerate(latents_over_time):
        if latent_index % every_n_queries == 0:
            _, image = generator.generate_image(latent)
            image = convert_stylegan_image_to_matplotlib(image)
            # Plot the image
            axs[latent_index // every_n_queries + 1].imshow(image)
            axs[latent_index // every_n_queries + 1].set_title(f"Sample\n{latent_index}")
    # Plot the target image
    _, target_latent = sampler.sample(
        target_attributes,
        initial_latent=start_latent,
        verbose_logging=False
    )
    _, target_image = generator.generate_image(target_latent)
    target_image = convert_stylegan_image_to_matplotlib(target_image)
    axs[-1].imshow(target_image)
    axs[-1].set_title("Target \nImage")
    # Remove all tick labels
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        """
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_aspect('equal')
        """
    # Plot the queries
    query_axs = subfigs[1].subplots(2, num_samples + 2)
    # subfigs[1].suptitle("Queries", y=1.1)

    for query_index, query in enumerate(queries):
        if latent_index % every_n_queries == 0:
            positive_latent = query.positive_latent
            negative_latent = query.negative_latent
            # Generate the images
            _, positive_image = generator.generate_image(positive_latent)
            positive_image = convert_stylegan_image_to_matplotlib(positive_image)
            _, negative_image = generator.generate_image(negative_latent)
            negative_image = convert_stylegan_image_to_matplotlib(negative_image)
            # Show the images
            query_axs[0, query_index // every_n_queries + 1].imshow(positive_image)
            query_axs[1, query_index // every_n_queries + 1].imshow(negative_image)
            # Remove axis lables and ticks
            ax = query_axs[0, query_index // every_n_queries + 1]
            ax.set_title(f"Query {query_index}")

    for row in query_axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            """
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_aspect('equal')
            """
            ax.set_facecolor("white")

    # query_axs[0, 1].set_ylabel("Positive\nImages", ha='right', va='center', rotation=0)
    # query_axs[1, 1].set_ylabel("Negative\nImages ", ha='right', va='center', rotation=0)
    plt.text(0.5, 0.5, "Positive\nImages", ha='center', va='center', rotation=0, transform=query_axs[0, 0].transAxes)
    plt.text(0.5, 0.5, "Negative\nImages", ha='center', va='center', rotation=0, transform=query_axs[1, 0].transAxes)

    # plt.subplots_adjust(wspace=0, hspace=0.0)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # Save the plot
    if not save_path is None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

@plot_ignore_exceptions
def animate_localization_path(attribute_path_dir, time=10.0, save_path=None): 
    # Get the paths in the attribute_path dir
    paths = os.listdir(attribute_path_dir)
    output_paths = [os.path.join(attribute_path_dir, path) for path in paths]
    output_paths.sort()

    if not save_path is None:
        clip = ImageSequenceClip(output_paths, fps=len(output_paths)/time)
        clip.write_gif(save_path, fps=len(output_paths)/time)

@plot_ignore_exceptions
def log_attribute_space_path(
    latent_sampler, 
    simulation_data, 
    save_path=None,
    plot_points=False,
    lims=(0.0, 1.0)
):
    """
        Animate preference path in attribute space
    """
    # Only works if attribute dimension is 2
    assert simulation_data.target_attributes.shape[0] == 2, simulation_data.target_attributes.shape
    # Unpack information from the simulation
    uniform_attribute_points = latent_sampler.randomly_sample_attributes(
        num_samples=1000,
        return_dict=False
    ).detach().cpu().numpy()
    target_attributes = simulation_data.target_attributes
    queries = simulation_data.queries
    # attributes_over_time = np.stack(simulation_data.attributes_over_time)
    attributes_over_time = simulation_data.attributes_over_time
    num_queries = len(simulation_data.attributes_over_time)
    # Plot Gaussians
    fig, axs = plt.subplots(1, 1, dpi=300)
    axs.set_aspect('equal')
    axs.set_ylim(bottom=lims[0], top=lims[1])
    axs.set_xlim(left=lims[0], right=lims[1])
    if plot_points:
        axs.scatter(
            uniform_attribute_points[:, 0], 
            uniform_attribute_points[:, 1], 
            color="black", 
        )
    for query_index in range(num_queries - 1):
        x = [
            attributes_over_time[query_index][0], 
            attributes_over_time[query_index + 1][0]
        ]
        y = [
            attributes_over_time[query_index][1], 
            attributes_over_time[query_index + 1][1]
        ]
        line = Line2D(x, y, color='black', ms=12, linewidth=3)
        axs.add_line(line)
    # Plot the query points
    axs.scatter(
        x=queries[-1].positive_attribute[0],
        y=queries[-1].positive_attribute[1],
        color="green",
        label="Positive Query Attributes",
        zorder=2
    )
    axs.scatter(
        x=queries[-1].negative_attribute[0],
        y=queries[-1].negative_attribute[1],
        color="red",
        label="Negative Query Attributes",
        zorder=2
    )

    # Plot the attribute estimate gaussian
    preference_samples = simulation_data.preference_samples_over_time[-1]
    sns.kdeplot(
        x=preference_samples[:, 0],
        y=preference_samples[:, 1],
        ax=axs,
        thresh=0.2
    )
    axs.scatter(
        target_attributes[0], 
        target_attributes[1], 
        color="green",
        marker="*",
        label="Target Attribute",
        zorder=2
    )
    axs.scatter(
        attributes_over_time[-1][0], 
        attributes_over_time[-1][1], 
        color="blue", 
        marker="*", 
        s=2*matplotlib.rcParams['lines.markersize'] ** 2,
        label="Estimated Attribute",
        zorder=2
    )
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Save the plot
    if not save_path is None:
        plt.savefig(
            append_id(save_path, len(attributes_over_time)),
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        
    # Log the plot to wandb
    if wandb.run is not None:
        wandb.log({
            "Attribute Space Path": plt
        })

@plot_ignore_exceptions
def log_latent_image(generator, latent_vector, query_number=0):
    """
        Logs image generated from latent vector
    """
    latent, image = generator.generate_image(latent=latent_vector)
    # Log Current Image
    wandb.log({
        "Current Latent Image": wandb.Image(image),
        "Query Number": query_number
    })

@plot_ignore_exceptions
def log_per_query_metrics(
    generator, 
    latent_sampler,
    id_loss, 
    simulation_data, 
    query_num=0
):
    """
        Logs a host of metrics
    """
    last_attribute = simulation_data.attributes_over_time[query_num]
    # Compute distance in attribute space over time
    attribute_distance = np.linalg.norm(last_attribute - simulation_data.target_attributes)
    # Comppute attribute ranking loss
    ranked_attribute_distance = compute_attribute_rank_loss(
        latent_sampler=latent_sampler,
        current_attributes=simulation_data.attributes_over_time[-1], 
        target_attributes=simulation_data.target_attributes,
        num_samples=500
    )
    # Compute ID Loss
    _, initial_image = generator.generate_image(
        simulation_data.start_latent
    )
    _, current_image = generator.generate_image(
        simulation_data.latents_over_time[query_num]
    )
    id_loss = id_loss(initial_image, current_image)
    # Wandb logging
    if wandb.run is not None:
        wandb.log({
            "attribute_distance": attribute_distance,
            "ranked attribute distance": ranked_attribute_distance,
            "id_loss": id_loss
        })

@plot_ignore_exceptions
def log_per_query_points(
    generator, 
    latent_sampler,
    simulation_data, 
    query_num=0, 
    save_path=None
):
    """
        Logs a 2D attribute scatter plot and the points for 
        the query with the given index. 
    """
    # Only works if attribute dimension is 2
    assert simulation_data.target_attributes.shape[0] == 2, simulation_data.target_attributes.shape
    # Unpack information from the simulation
    uniform_attribute_points = latent_sampler.randomly_sample_attributes(
        num_samples=1000,
        return_dict=False
    ).detach().cpu().numpy()
    target_attributes = simulation_data.target_attributes
    # Plot Gaussians
    fig, axs = plt.subplots(1, 1, dpi=100)
    axs.scatter(uniform_attribute_points[:, 0], uniform_attribute_points[:, 1], color="black", alpha=0.2)
    axs.scatter(target_attributes[0], target_attributes[1], color="blue", marker="*")
    # Plot the current samples
    preference_samples = simulation_data.preference_samples_over_time[query_num]
    axs.scatter(preference_samples[:, 0], preference_samples[:, 1], color="red", alpha=0.2)
    # Plot the query
    positive_attribute = simulation_data.queries[query_num].positive_attribute
    negative_attribute = simulation_data.queries[query_num].negative_attribute
    axs.scatter(positive_attribute[0], positive_attribute[1], color="green", marker="*")
    axs.scatter(negative_attribute[0], negative_attribute[1], color="red", marker="*")
    # Log the plot to wandb
    if wandb.run is not None:
        wandb.log({
            "Most Recent Query": plt,
            "query_num": query_num
        })
    # Save the plot
    if not save_path is None:
        plt.savefig(save_path)

@plot_ignore_exceptions
def log_target_image(generator, sampler, start_latent, target_attributes, save_path=None):
    """
        Log an image with start_latent and target_attributes
    """
    # Langevin dynamics sampling
    _, final_latent = sampler.sample(
        target_attributes,
        initial_latent=start_latent,
        verbose_logging=False
    )
    # Generate image
    _, image = generator.generate_image(latent=final_latent)
    # Wandb logging
    if wandb.run is not None:
        wandb.log({
            "Target Image": wandb.Image(image)
        })
    # Save the plot
    if not save_path is None:
        plt.savefig(save_path)
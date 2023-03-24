"""
    GAN Control sampling method
"""
import os
import torch
import numpy as np
from prefgen.methods.sampling.utils import LatentAttributeSampler
from prefgen.methods.sampling.gan_control.gan_control.src.gan_control.inference.controller import Controller

controller_path = os.path.join(
    os.environ["PREFGEN_ROOT"],
    "prefgen/pretrained/stylegan2_gan_control/controller_dir",
)

class GANControlSampler(LatentAttributeSampler):
    """
        Sampler class for "sampling" from the GAN latent
        space using the methods from the GAN Control paper.
    """
    attribute_names_to_dimensions = {
        "age": 1,
        "orientation": 3,
        "yaw": 1,
        "pitch": 1,
        "roll": 1,
    }
    attribute_maxes = {
        "age": [75], 
        "orientation": [30.0, 20.0, 10.0],
        "pitch": [20.0],
        "yaw": [30.0],
        "roll": [10.0],
    }
    attribute_mins = {
        "age": [15],
        "orientation": [-30.0, -20.0, 0.0],
        "pitch": [-20.0],
        "yaw": [-30.0],
        "roll": [0.0],
    }

    def __init__(
        self, 
        generator,
        attribute_names=None,
        dataset_name="ffhq",
    ):
        self.generator = generator
        self.attribute_names = attribute_names
        self.dataset_name = dataset_name
        assert self.dataset_name in ["ffhq"]
        # Load up the controller model
        self.controller = Controller(controller_path)
        self.attribute_size = self._compute_attribute_size()

    def _compute_attribute_size(self):
        """
            Compute the size of the attribute vector by summing
            the dimensions of each attribute. 
        """
        return sum([
            self.attribute_names_to_dimensions[attribute_name]
            for attribute_name in self.attribute_names
        ])

    def generate_latent_attribute_pairs(
        self,
        num_examples=1000
    ):
        latents = []
        attributes = []
        for _ in range(num_examples):
            random_attributes = self.randomly_sample_attributes(normalize=True)
            # Sample an image
            modified_image, initial_latent, modified_latent, _ = self.sample(
                target_attributes=random_attributes
            )
            # Add to list
            latents.append(modified_latent)
            attributes.append(
                self.convert_controls_dict_to_attribute_vector(
                    random_attributes
                ).detach().cpu().numpy()
            )

        return latents, attributes

    def randomly_sample_attributes(self, num_samples=None, normalize=True, return_dict=True):
        """
            Sample random attributes from the attribute space. 
        """
        if num_samples is None:
            # TODO Finish this
            attributes = {}
            for attribute_name in self.attribute_names:
                attribute_val = np.random.uniform(
                    low=GANControlSampler.attribute_mins[attribute_name],
                    high=GANControlSampler.attribute_maxes[attribute_name],
                )
                if isinstance(attribute_val, float):
                    attribute_val = torch.Tensor(
                        [attribute_val]
                    )
                else:
                    attribute_val = torch.Tensor(
                        attribute_val
                    )
                attribute_val = attribute_val.cuda()
                attributes[attribute_name] = attribute_val

            if normalize:
                attributes = self.convert_controls_dict_to_attribute_vector(attributes)
                attributes = self.normalize_attribute_vector(attributes)
                assert torch.all(attributes >= 0.0)
                attributes = self.convert_attribute_vector_to_controls_dict(attributes)

            if return_dict:
                return attributes
            else:
                return self.convert_controls_dict_to_attribute_vector(attributes)
        else:
            attributes_list = []
            for _ in range(num_samples):
                attributes = self.randomly_sample_attributes(
                    normalize=normalize,
                    return_dict=return_dict
                )
                attributes_list.append(attributes)

            if return_dict:
                return attributes_list
            else:
                return torch.stack(attributes_list, dim=0)

    def convert_attribute_vector_to_controls_dict(
        self, 
        attribute_vector
    ):
        """
            Takes a concatenated attribute vector and converts it
            into a dictionary of attributes that can be passed to 
            the controller. 
        """
        attribute_dict = {}
        current_index = 0
        for attribute_name in self.attribute_names:
            attribute_dict[attribute_name] = attribute_vector[
                current_index: current_index + self.attribute_names_to_dimensions[attribute_name]
            ]
            current_index += self.attribute_names_to_dimensions[attribute_name]

        return attribute_dict

    def convert_controls_dict_to_attribute_vector(
        self, 
        controls_dict
    ):
        """
            Takes a dictionary of controls and converts it into a 
            concatenated attribute vector. 
        """ 
        # Concatenate in order of attribute names
        attribute_vector = []
        for attribute_names in self.attribute_names:
            attribute_value = controls_dict[attribute_names]
            if isinstance(attribute_value, float):
                attribute_value = torch.Tensor([attribute_value]).cuda()
            elif isinstance(attribute_value, list):
                attribute_value = torch.Tensor(attribute_value).cuda()
            attribute_vector.append(attribute_value)
        
        attribute_vector = torch.cat(attribute_vector, dim=0)
        return attribute_vector
            
    def normalize_attribute_vector(self, attribute_vector, normalization_type="constant"):
        """Squash the attribute vector into the range [0, 1]"""
        if normalization_type == "constant":
            mins = torch.cat([
                torch.Tensor(GANControlSampler.attribute_mins[attribute_name]) for attribute_name in self.attribute_names
            ]).cuda()
            maxes = torch.cat([
                torch.Tensor(GANControlSampler.attribute_maxes[attribute_name]) for attribute_name in self.attribute_names
            ]).cuda()
            return (attribute_vector - mins) / (maxes - mins)
        else:
            raise NotImplementedError()

    def unnormalize_attribute_vector(self, attribute_vector, normalization_type="constant"):
        """Unsquash the attribute vector from the range [0, 1]"""
        if normalization_type == "constant":
            mins = torch.cat([
                torch.Tensor(GANControlSampler.attribute_mins[attribute_name]) for attribute_name in self.attribute_names
            ]).cuda()
            maxes = torch.cat([
                torch.Tensor(GANControlSampler.attribute_maxes[attribute_name]) for attribute_name in self.attribute_names
            ]).cuda()
            return attribute_vector * (maxes - mins) + mins
        else:
            raise NotImplementedError()

    def post_process_controls_dict(self, controls_dict):
        """
            Post-processes the controls dict to ensure that the
            values are within the correct ranges. 
        """
        if "orientation" in controls_dict:
            return controls_dict
        if "pitch" in controls_dict or "yaw" in controls_dict or "roll" in controls_dict:
            controls_dict_copy = controls_dict.copy()
            orientation = torch.Tensor([0.0, 0.0, 0.0]).cuda()
            if "yaw" in controls_dict:
                orientation[0] = controls_dict["yaw"].item()
                del controls_dict_copy["yaw"]
            if "pitch" in controls_dict:
                orientation[1] = controls_dict["pitch"].item()
                del controls_dict_copy["pitch"]
            if "roll" in controls_dict:
                orientation[2] = controls_dict["roll"].item()
                del controls_dict_copy["roll"]
            controls_dict_copy["orientation"] = orientation
            return controls_dict_copy
        else:
            return controls_dict

    def sample(
        self, 
        target_attributes=None, 
        initial_latent=None, 
        verbose_logging=True, 
        target_latent=None,
        truncation=0.7,
        attributes_normalized=True,
    ):
        """
            Sample from the GAN latent space using the GAN Control
            method.
        """
        # Generate input image corresponding to initial latent
        if initial_latent is None:
            initial_latent = self.generator.randomly_sample_latent()
        if isinstance(initial_latent, np.ndarray):
            initial_latent = torch.Tensor(initial_latent).cuda()
        # Sample random target attribute if necessary
        if target_attributes is None:
            target_attributes = self.randomly_sample_attributes()
            target_attributes = self.convert_controls_dict_to_attribute_vector(target_attributes)
        else:
            if isinstance(target_attributes, dict):
                target_attributes = self.convert_controls_dict_to_attribute_vector(target_attributes)
        if isinstance(target_attributes, np.ndarray):
            target_attributes = torch.Tensor(target_attributes).cuda()
        # Un normalize target attributes
        if attributes_normalized:
            target_attributes = self.unnormalize_attribute_vector(target_attributes)
        # Convert target attributes to controls dict
        controls_dict = self.convert_attribute_vector_to_controls_dict(
            target_attributes
        )
        # print(f"Pre processed controls dict: {controls_dict}")
        # Post process the controls dict
        controls_dict = self.post_process_controls_dict(controls_dict)
        # print(f"Controls dict: {controls_dict}")
        # Perform GAN Control sampling
        image_tensors, _, modified_latent_w = self.controller.gen_batch_by_controls(
            latent=initial_latent, 
            input_is_latent=True, 
            **controls_dict
        )

        if verbose_logging:
            return image_tensors, initial_latent, modified_latent_w, target_attributes
        else:
            return initial_latent, modified_latent_w
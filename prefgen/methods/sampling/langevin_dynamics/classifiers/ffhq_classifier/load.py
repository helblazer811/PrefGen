import os
import torch

from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.ffhq_data import att_dict
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.latent_model import DenseEmbedder

"""
    This is where we implment the basic attribute classifier needed 
    for conditional sampling. 
"""
PRETRAINED_PATH = os.path.join(
    os.environ["PREFGEN_ROOT"], 
    'prefgen/pretrained/'
)

def load_ffhq_wspace_classifier(
        map_z_to_w, 
        attribute_names, 
        pretrained_path=PRETRAINED_PATH, 
        device="cuda", 
        latent_dim=512, 
        w_space_latent=False
    ):
    """
        Load an attribute classifier. 
    """
    LOAD_PATHS = lambda att_name: os.path.join(
        pretrained_path, 
        f'dense_embedder_w/best_valid_ckpt_{att_name}.pt'
    )
    # Load theårelevant classifiers
    classifiers = []
    for i, att_name in enumerate(attribute_names):
        assert not att_name is None
        load_path = LOAD_PATHS(att_name)
        assert os.path.exists(load_path), load_path
        classifier_ckpt_dict = torch.load(load_path)
        # Get num classes
        num_classes_list = [att_dict[att_name][1]]
        # Initialize Classifier Function
        classifier_i = DenseEmbedder(
            input_dim=latent_dim, 
            up_dim=128, 
            norm=None, 
            num_classes_list=num_classes_list # TODO this may not be the case
        )
        # Load model weights
        classifier_i.load_state_dict(
            classifier_ckpt_dict["state_dict"]
        )
        classifier_i.to(device)
        classifier_i.eval()
        classifiers.append(classifier_i)

    def classifier_function(latent=None):
        """
            Runs multiple classifiers and concatenates
            the outputs. 
        """
        output_values = []
        for classifier in classifiers:
            if w_space_latent:
                w_vector = latent
            else:
                w_vector = map_z_to_w(latent)
            
            value = classifier(w_vector)
            assert isinstance(value, torch.Tensor)
            output_values.append(value)

        output_vector = torch.cat(output_values, dim=-1).to(device)
        output_vector = output_vector.squeeze(0)
        return output_vector

    return classifier_function

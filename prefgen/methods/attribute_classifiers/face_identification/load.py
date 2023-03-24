import os
import torch

from prefgen.methods.attribute_classifiers.face_identification.w_space_to_identity import DenseEmbedder

def load_identity_classifier(map_z_to_w, device="cuda"):
    """
        Loads a model that maps w-space vectors from
        StyleGAN to Arcface vectors. 
    """
    default_path = os.path.join(
        os.environ["PREFGEN_ROOT"],
        "prefgen/pretrained/ArcFaceWSpace/identity_wspace.pt"
    )
    # Initialize Classifier Function
    face_identity_network = DenseEmbedder(
        input_dim=512, 
        up_dim=512, 
        norm=None, 
        num_classes_list=[512]
    )
    # Load model weights
    face_identity_network.load_state_dict(torch.load(default_path))
    face_identity_network.to(device)
    face_identity_network.eval()

    def classifier_function(latent_vector):
        """
        #    Maps to w and then applies expression network
        """
        w_vector = map_z_to_w(latent_vector)
        return face_identity_network(w_vector)
    
    return classifier_function
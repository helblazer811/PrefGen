from prefgen.external_modules.FECNet.models.FECNet import load_weights
from prefgen.methods.attribute_classifiers.facial_expression.densenet import DenseNet
import torch
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn

class FECNet(nn.Module):
    """FECNet model with optional loading of pretrained weights.
    Model parameters can be loaded based on pretraining on the Google facial expression comparison
    dataset (https://ai.google/tools/datasets/google-facial-expression/). Pretrained state_dicts are
    automatically downloaded on model instantiation if requested and cached in the torch cache.
    Subsequent instantiations use the cache rather than redownloading.
    Keyword Arguments:
        pretrained {str} -- load pretraining weights
    """
    def __init__(self, pretrained=False):
        super(FECNet, self).__init__()
        growth_rate = 64
        depth = 100
        block_config = [5]
        efficient = True
        self.Inc = InceptionResnetV1(
            pretrained='vggface2', 
            device='cuda'
        ).eval()
        for param in self.Inc.parameters():
            param.requires_grad = False
        self.dense = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_classes=16,
            small_inputs=True,
            efficient=efficient,
            num_init_features=512
        ).cuda()

        if (pretrained):
            load_weights(self)

    def forward(self, x):
        feat = self.Inc(x)[1]
        out = self.dense(feat)
        return out

def load_fecnet_default():
    """
        Loads the default FECNet model.
    """
    return FECNet(pretrained=True)

def load_facenet_model(
        embedding_dim=16,
        save_path=None,
    ):
    """
        FaceNet model  
    """
    # Create an inception resnet (in eval mode):
    model = InceptionResnetV1(pretrained='vggface2')
    # Change the last linear layer from the model
    model.last_linear = torch.nn.Linear(
        1792, 
        embedding_dim, 
        bias=False
    )
    model.last_bn = torch.nn.BatchNorm1d(
        embedding_dim, 
        eps=0.001, 
        momentum=0.1, 
        affine=True, 
        track_running_stats=True
    )
    # Load saved weights
    if not save_path is None:
        model.load_state_dict(torch.load(save_path))
    model.eval()

    return model
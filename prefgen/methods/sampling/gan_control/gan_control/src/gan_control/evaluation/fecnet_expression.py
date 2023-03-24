# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms, utils
from PIL import Image

from gan_control.utils.logging_utils import get_logger

_log = get_logger(__name__)

def calc_fecnet_expression_from_features(features):
    return features

def calc_fecnet_expression_from_tensor_images(fecnet_expression_loss_class, tensor_images):
    with torch.no_grad():
        features_list = fecnet_expression_loss_class.calc_features(tensor_images)
    features = features_list[-1]
    fecnet_expressions = calc_fecnet_expression_from_features(features)
    return fecnet_expressions

if __name__ == '__main__':
    import argparse
    from gan_control.datasets.ffhq_dataset import get_ffhq_data_loader
    from gan_control.utils.file_utils import read_json
    from gan_control.utils.ploting_utils import plot_hist
    from gan_control.losses.loss_model import LossModelClass

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--number_os_samples', type=int, default=70000)
    args = parser.parse_args()
    config = read_json(args.config_path, return_obj=True)
    loader = get_ffhq_data_loader(
        config.data_config, 
        batch_size=args.batch_size, 
        training=True, 
        size=config.model_config['size']
    )
    fecnet_expression_loss_class = None
    fecnet_expression_loss_class = LossModelClass(
        config.training_config['fecnet_expression_loss'], 
        loss_name='fecnet_expression_loss', 
        mini_batch_size=args.batch_size, 
        device="cuda"
    )

    tensor_images, _ = next(loader)

    raise NotImplementedError()

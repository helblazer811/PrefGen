import argparse
import os

from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier
import prefgen.methods.datasets.lstm as lstm_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Setup LSTM Localization Dataset")

    parser.add_argument("--num_examples", default=5000)
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--num_constraints", default=30)
    parser.add_argument("--attribute_names", default=["yaw", "age"])
    parser.add_argument(
        "--train_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/train_dataset_gan_control.pkl"
        ),
    )
    parser.add_argument(
        "--test_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/test_dataset_gan_control.pkl"
        ),
    )

    args = parser.parse_args()
    # Make the dataset
    """
    lstm_dataset.make_dataset_with_gan_control(
        num_examples=args.num_examples,
        num_constraints=args.num_constraints,
        save_path=args.train_save_path,
        attribute_names=args.attribute_names
    )
    """

    lstm_dataset.make_dataset_with_gan_control(
        num_examples=200,
        num_constraints=args.num_constraints,
        save_path=args.test_save_path,
        attribute_names=args.attribute_names
    )
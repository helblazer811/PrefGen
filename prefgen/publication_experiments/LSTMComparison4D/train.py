import argparse
import os
import torch
import pickle
from prefgen.experiments.lstm_baseline.evaluate_lstm_performance import evaluate_lstm_model_performance

from prefgen.methods.generative_models.stylegan2.stylegan_wrapper import StyleGAN2Wrapper
from prefgen.methods.generative_models.stylegan2_gan_control.model import StyleGAN2GANControlWrapper
from prefgen.methods.sampling.gan_control.sampler import GANControlSampler
from prefgen.methods.sampling.langevin_dynamics.classifiers.ffhq_classifier.load import load_ffhq_wspace_classifier
from prefgen.methods.datasets.lstm import ConstraintDataset
from prefgen.methods.lstm_localization.model import PreferenceLSTM
from prefgen.methods.lstm_localization.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Setup LSTM Localization Dataset")

    parser.add_argument("--num_epochs", default=30)
    parser.add_argument("--extended_latent", default=False)
    parser.add_argument("--w_space_latent", default=True)
    parser.add_argument("--num_constraints", default=30)
    parser.add_argument("--batch_size", default=10)
    parser.add_argument("--attribute_names", default=["yaw", "pitch", "roll", "age"])
    parser.add_argument(
        "--train_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/train_dataset_gan_control_4d.pkl"
        ),
    )
    parser.add_argument(
        "--test_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/data/lstm_dataset/test_dataset_gan_control_4d.pkl"
        ),
    )
    parser.add_argument(
        "--model_save_path", 
        default=os.path.join(
            os.environ["PREFGEN_ROOT"],
            "prefgen/pretrained/lstm_models/lstm_model_4d.pth"
        ),
    )
    parser.add_argument(
        "--save_eval_path", 
        type=str, 
        default="data/lstm_data_4d.pkl"
    )

    args = parser.parse_args()

    stylegan_generator = StyleGAN2GANControlWrapper(
        extended_latent=args.extended_latent,
        w_space_latent=args.w_space_latent,
    )
    latent_sampler = GANControlSampler(
        stylegan_generator,
        attribute_names=args.attribute_names,
    )
    # Make the dataset
    train_dataset = ConstraintDataset(args.train_save_path)
    test_dataset = ConstraintDataset(args.test_save_path)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    # Load model
    model = PreferenceLSTM(
        attribute_size=len(args.attribute_names)
    )

    train(
        model, 
        latent_sampler,
        train_dataset, 
        test_dataset, 
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_save_path
    )

    eval_data = evaluate_lstm_model_performance(
        model,
        test_dataset,
        num_examples=20,
        num_queries=30,
        query_testing_interval=1,
    )

    with open(args.save_eval_path, "wb") as f:
        pickle.dump(eval_data, f)
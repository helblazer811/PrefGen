import argparse
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl

from prefgen.methods.plotting.localization import compute_attribute_rank_loss
mpl.rcParams['font.family'] = 'Times New Roman'

def plot_noise_robustness(
    mcmv_data,
    save_path="plots/noise_robustness.pdf"
):
    fig, axs = plt.subplots(1, 2, figsize=(6.875/3 * 2.4, 3.0))
    # Make pandas dataframe out of data
    df = pd.DataFrame()
    index = 0
    for noise_level in mcmv_data.keys():
        for (error, cov_det) in mcmv_data[noise_level]:
            df = df.append(
                pd.DataFrame(
                    {
                        "Degree of Query Noise": noise_level,
                        "Preference Estimate MSE to Target": error,
                        "Determinant of Posterior\nCovariance Matrix": cov_det,
                        "Method": "LSTM Baseline"
                    },
                    index=[index]
                )
            )
            index += 1
    # Plot violin plots
    # set color palette
    # sns.set_palette("classic")
    sns.boxplot(
        df,
        x="Degree of Query Noise",
        y="Preference Estimate MSE to Target",
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
        #hue="Method",
        # palette={
        #    "Active Querying": plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
        # }
        ax=axs[0]
    )
    plt.yscale("log")

    sns.boxplot(
        df,
        x="Degree of Query Noise",
        y="Determinant of Posterior\nCovariance Matrix",
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
        #hue="Method",
        # palette={
        #    "Active Querying": plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
        # }
        ax=axs[1]
    )

    sns.set(font_scale = 1.2)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    # plt.ylim(bottom=0.000001, top=0.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


def plot_noise_robustness_percentile(
    mcmv_data,
    save_path="plots/noise_robustness_with_uncertainty.pdf"
):
    fig, axs = plt.subplots(1, 2, figsize=(6.875/3 * 2.0, 2.4))
    # Make pandas dataframe out of data
    df = pd.DataFrame()
    index = 0
    for noise_level in mcmv_data.keys():
        for index, (error, cov_det, percentile_error) in enumerate(mcmv_data[noise_level]):
            df = df.append(
                pd.DataFrame(
                    {
                        "Degree of Query Noise": noise_level,
                        "Percentage Closer\nto True Target": percentile_error,
                        "Determinant of Posterior\nCovariance Matrix": cov_det,
                        "Method": "LSTM Baseline"
                    },
                    index=[index]
                )
            )
            index += 1
    # Plot violin plots
    # set color palette
    # sns.set_palette("classic")
    sns.boxplot(
        df,
        x="Degree of Query Noise",
        y="Percentage Closer\nto True Target",
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
        #hue="Method",
        # palette={
        #    "Active Querying": plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
        # }
        ax=axs[0]
    )
    plt.yscale("log")

    sns.boxplot(
        df,
        x="Degree of Query Noise",
        y="Determinant of Posterior\nCovariance Matrix",
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
        #hue="Method",
        # palette={
        #    "Active Querying": plt.rcParams['axes.prop_cycle'].by_key()['color'][1],
        # }
        ax=axs[1]
    )

    sns.set(font_scale = 1.2)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    # plt.ylim(bottom=0.000001, top=0.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mcmv_data_path", 
        type=str, 
        default="data/mcmv_data_random_flip_2d_noise_scales.pkl"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="plots/noise_robustness_with_uncertainty.pdf"
    )

    args = parser.parse_args()

    mcmv_data = pickle.load(open(args.mcmv_data_path, "rb"))
    #del mcmv_data[0.5]

    plot_noise_robustness_percentile(
        mcmv_data,
        save_path=args.save_path
    )
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def prepare_dataframe_for_multiple_hyperparams_sets(
    selected_hyperparams_models,
    names_of_settings,
    numbers_of_models,
    suffixes,
    number_of_tasks,
):
    """
    Load file with results for consecutive models and prepare a merged
    dataframe for different runs for a given setup.

    Arguments:
    ----------
       *selected_hyperparams_models*: list of strings with main paths for models
       *names_of_settings': list of strings with names displayed in a legend
       *numbers_of_models*: list of of lists of integers with numbers of models
                            to count for each hyperparams setup
       *suffix*: list of strings with names of files for consecutive results
       *number_of_tasks*: integer representing the total number of tasks

    Returns a merged dataframe
    """
    dataframes = []
    for hyperparams, model_runs, model_name, cur_suffix in zip(
        selected_hyperparams_models,
        numbers_of_models,
        names_of_settings,
        suffixes,
    ):
        for model in model_runs:
            dataframe = pd.read_csv(
                f"{hyperparams}{model}/{cur_suffix}", sep=";"
            )
            dataframe = dataframe.loc[
                dataframe["after_learning_of_task"] == (number_of_tasks - 1)
            ][["tested_task", "accuracy"]]
            dataframe.insert(
                0, "model_setting", [model_name for i in range(number_of_tasks)]
            )
            dataframes.append(dataframe)
    dataframe_merged = pd.concat(dataframes, axis=0, ignore_index=True)
    dataframe_merged = dataframe_merged.astype({"tested_task": int})
    dataframe_merged["tested_task"] += 1
    return dataframe_merged


def plot_different_setups_consecutive_tasks(
    dataframe, dataset_name, filepath, name=None
):
    """
    Plot results for different configs of hyperparameters
    during consecutive continual learning tasks.

    Arguments:
    ----------
      *dataframe*: Pandas Dataframe containing columns: tested_task, accuracy
                   and model_setting. First of them represents the number of the
                   currently evaluated task, accuracy represents the corresponding
                   overall accuracy and model_settings mean the hyperparameters'
                   config.
      *dataset_name*: string representing current dataset for the plot title
      *filepath_with_name*: string representing path for the file with plot
      *name*: optional string representing name of the plot file
    """
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            "grid.linewidth": 0.4,
        }
    )
    if dataset_name == "Permuted MNIST":
        # mean and 95% confidence intervals
        errors = ("ci", 95)
        height = 4
        aspect = 1.3
        fontsize = 11
    elif dataset_name == "Split MNIST":
        errors = None
        height = 2.5
        aspect = 1.5
        fontsize = 8
    else:
        raise NotImplementedError
    ax = sns.relplot(
        data=dataframe,
        x="tested_task",
        y="accuracy",
        kind="line",
        hue="model_setting",
        errorbar=errors,
        height=height,
        aspect=aspect,
    )
    ax.set(
        xticks=[i + 1 for i in range(number_of_tasks)],
        xlabel="Number of task",
        ylabel="Accuracy [%]",
    )
    if dataset_name == "Permuted MNIST":
        sns.move_legend(
            ax,
            "upper right",
            bbox_to_anchor=(0.61, 0.96),
            fontsize=fontsize,
            title="",
        )
        plt.title(f"Results for different hyperparameters for {dataset_name}")
    elif dataset_name == "Split MNIST":
        if dataframe_merged["model_setting"].unique().shape[0] >= 6:
            legend_fontsize = 7
        else:
            legend_fontsize = fontsize
        sns.move_legend(
            ax,
            "lower center",
            ncol=2,
            bbox_to_anchor=(0.38, 0.95),
            columnspacing=0.8,
            fontsize=legend_fontsize,
            title="",
        )
    plt.xlabel("Number of task", fontsize=fontsize)
    plt.ylabel(r"Accuracy [\%]", fontsize=fontsize)
    os.makedirs(filepath, exist_ok=True)
    if name is None:
        name = f"hyperparams_{dataset_name.replace(' ', '_')}"
    plt.savefig(f"{filepath}/{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_mean_accuracy_for_CL_tasks_matrix(
    main_path_for_models,
    numbers_of_models,
    suffix,
    filepath,
    version="greater",
    name=None,
    title=None,
):
    """
    Plot a matrix of mean overall accuracy for consecutive continual
    learning tasks, taking into account several runs for a given
    architecture setup.

    Arguments:
    ----------
       *main_path_for_models*: string with main path for models
       *numbers_of_models*: list of of lists of integers with numbers of models
                            to count for a given hyperparams setup
       *suffix*: string with name of files for consecutive results
       *filepath*: string representing path for the file with plot
       *version*: 'greater': fitted for 10 tasks,
                  'smaller': fitted for 5 tasks
       *name*: optional string representing name of the plot file
       *title*: optional string representing title of the plot
    """
    dataframes = []
    for model_no in numbers_of_models:
        load_path = f"{main_path_for_models}{model_no}/{suffix}"
        dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
        dataframe = dataframe.astype(
            {"after_learning_of_task": "int32", "tested_task": "int32"}
        )
        dataframes.append(dataframe)
    merged_dataframe = (
        pd.concat(dataframes)
        .groupby(["after_learning_of_task", "tested_task"], as_index=False)[
            "accuracy"
        ]
        .agg(list)
    )
    merged_dataframe["mean_accuracy"] = [
        np.mean(x) for x in merged_dataframe["accuracy"].to_numpy()
    ]
    merged_dataframe["after_learning_of_task"] += 1
    merged_dataframe["tested_task"] += 1
    table = merged_dataframe.pivot(
        "after_learning_of_task", "tested_task", "mean_accuracy"
    )
    plt.rcParams.update({"text.usetex": False})
    if version == "greater":
        size_kws = 8.5
        title_size = 8
    elif version == "smaller":
        size_kws = 4
        title_size = 3.75
    else:
        raise ValueError("Wrong value of version argument!")
    p = sns.heatmap(
        table,
        annot=True,
        fmt=".1f",
        linewidth=0.2,
        annot_kws={"size": size_kws},
    )
    plt.xlabel("Number of the tested task")
    plt.ylabel("Number of the previously learned task")
    if title is not None:
        plt.title(title, fontsize=title_size)
    figure = plt.gcf()
    if version == "greater":
        figure.set_size_inches(5.5, 3.75)
    elif version == "smaller":
        figure.set_size_inches(1.75, 1)
        p.set_xticklabels(
            np.unique(merged_dataframe["tested_task"].to_numpy()), size=size_kws
        )
        p.set_yticklabels(
            np.unique(merged_dataframe["tested_task"].to_numpy()), size=size_kws
        )
        p.xaxis.get_label().set_fontsize(size_kws)
        p.yaxis.get_label().set_fontsize(size_kws)
        p.collections[0].colorbar.ax.tick_params(labelsize=size_kws)
    else:
        raise ValueError("Wrong value of version argument!")
    os.makedirs(filepath, exist_ok=True)
    if name is None:
        name = "best_hyperparams_mean_accuracy"
    plt.savefig(f"{filepath}/{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Different sets of hyperparameters - Permuted MNIST
    # Exemplary analysis - one should create similar code for own experiments
    plot_path = "./Plots"
    main_path_for_models = "./Models/"
    selected_hyperparams_models = [
        f"{main_path_for_models}grid_search_1/",  # 0-4
        f"{main_path_for_models}grid_search_2/",  # 0-4
        f"{main_path_for_models}grid_search_3/",  # 5-9
        f"{main_path_for_models}grid_search_4/",  # 0-4
    ]
    names_of_settings = [
        r"$\beta = 0.0005, \lambda = 0.001$, masked $L_1$, $p = 0$",
        r"$\beta = 0.001, \lambda = 0.001$, masked $L_1$, $p = 0$",
        r"$\beta = 0.001, \lambda = 0.001$, non-masked $L_1$, $p = 0$",
        r"$\beta = 0.005, \lambda = 0.001$, masked $L_1$, $p = 0$",
    ]
    numbers_of_models = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4],
    ]
    suffixes = ["results_mask_sparsity_0.csv"] * len(numbers_of_models)
    number_of_tasks = 10
    dataset_name = "Permuted MNIST"
    dataframe_merged = prepare_dataframe_for_multiple_hyperparams_sets(
        selected_hyperparams_models,
        names_of_settings,
        numbers_of_models,
        suffixes,
        number_of_tasks,
    )
    plot_different_setups_consecutive_tasks(
        dataframe_merged, dataset_name, plot_path, name=None
    )

    # For SplitMNIST concatenate grid search files OR best embedding, mean after two seeds
    # and different betas and lambdas as well as with 3D scatter plot with True/False
    # as well as for different sparsity params, rather for better batch sizes

    ####################################################################
    filepath = "./Plots/"
    # Best set of hyperparams: Permuted MNIST 10 tasks, HyperMask
    numbers_of_models = [0, 1, 2, 3, 4]
    suffix = "results_mask_sparsity_0.csv"
    title = "Mean accuracy for 5 runs of the best HyperMask setting for Permuted MNIST"
    name = "hypermask_best_setting_Permuted_MNIST_accuracy_matrix"
    # Best set of hyperparams: Permuted MNIST 10 tasks, HNET
    numbers_of_models = [1, 2, 3, 4, 5]
    suffix = "results.csv"
    title = "Mean accuracy for 5 runs of HNET for Permuted MNIST"
    name = "HNET_Permuted_MNIST_accuracy_matrix"
    # Best set of hyperparams: Split MNIST, HNET
    numbers_of_models = [1, 2, 3, 4, 5]
    suffix = "results.csv"
    title = "  Mean accuracy for 5 runs of HNET for Split MNIST"
    name = "HNET_Split_MNIST_accuracy_matrix"
    # Best set of hyperparams: Split MNIST, HyperMask
    numbers_of_models = [0, 1, 2, 3, 4]
    suffix = "results_mask_sparsity_30.csv"
    title = "  Mean accuracy for 5 runs of HyperMask for Split MNIST"
    name = "hypermask_best_setting_Split_MNIST_accuracy_matrix"
    # Best set of hyperparams: CIFAR, 100 tasks, ZenkeNet/ResNet, HyperMask
    for net in ["Zenke", "Res"]:
        numbers_of_models = [0, 1, 2, 3, 4]
        suffix = "results_mask_sparsity_0.csv"
        title = f"  Mean accuracy for 5 runs of HyperMask (with {net}Net) for CIFAR-100"
        name = f"hypermask_best_setting_CIFAR_{net}Net_accuracy_matrix"
        plot_mean_accuracy_for_CL_tasks_matrix(
            main_path_for_models,
            numbers_of_models,
            suffix,
            filepath,
            version="greater",
            name=name,
            title=title,
        )

    plot_path = "./Plots"
    selected_hyperparams_models = [
        4 * [f"{main_path_for_models}part_2/"],
        4 * [f"{main_path_for_models}part_2/"],
        6 * [f"{main_path_for_models}part_2/"],
        4 * [f"{main_path_for_models}part_2/"],
        6 * [f"{main_path_for_models}part_2/"],
        2 * [f"{main_path_for_models}part_2/"]
        + 2 * [f"{main_path_for_models}part_3/"]
        + 2 * [f"{main_path_for_models}part_1/"],
    ]
    names_of_settings = [
        [
            r"$\lambda = 0.001$, masked $L^1$",
            r"$\lambda = 0.001$, pure $L^1$",
            r"$\lambda = 0.0001$, masked $L^1$",
            r"$\lambda = 0.0001$, pure $L^1$",
        ],
        [
            r"$\lambda = 0.001$, $\beta = 0.001$",
            r"$\lambda = 0.0001$, $\beta = 0.001$",
            r"$\lambda = 0.001$, $\beta = 0.01$",
            r"$\lambda = 0.0001$, $\beta = 0.01$",
        ],
        [
            r"$p = 0$, masked $L^1$",
            r"$p = 0$, pure $L^1$",
            r"$p = 30$, masked $L^1$",
            r"$p = 30$, pure $L^1$",
            r"$p = 70$, masked $L^1$",
            r"$p = 70$, pure $L^1$",
        ],
        [
            r"$\beta = 0.001$, masked $L^1$",
            r"$\beta = 0.001$, pure $L^1$",
            r"$\beta = 0.01$, masked $L^1$",
            r"$\beta = 0.01$, pure $L^1$",
        ],
        [
            r"$H_{\Phi}$: [10, 10], masked $L^1$",
            r"$H_{\Phi}$: [10, 10], pure $L^1$",
            r"$H_{\Phi}$: [25, 25], masked $L^1$",
            r"$H_{\Phi}$: [25, 25], pure $L^1$",
            r"$H_{\Phi}$: [50, 50], masked $L^1$",
            r"$H_{\Phi}$: [50, 50], pure $L^1$",
        ],
        [
            r"emb.: 128, masked $L^1$",
            r"emb.: 128, pure $L^1$",
            r"emb.: 24, masked $L^1$",
            r"emb.: 24, pure $L^1$",
            r"emb.: 96, masked $L^1$",
            r"emb.: 96, pure $L^1$",
        ],
    ]
    numbers_of_models = [
        [[64, 65], [66, 67], [72, 73], [74, 75]],
        [[64, 65], [72, 73], [208, 209], [216, 217]],
        [[48, 49], [50, 51], [64, 65], [66, 67], [80, 81], [82, 83]],
        [[64, 65], [66, 67], [208, 209], [210, 211]],
        [[16, 17], [18, 19], [64, 65], [66, 67], [112, 113], [114, 115]],
        [[64, 65], [66, 67], [64, 65], [66, 67], [64, 65], [66, 67]],
    ]
    suffixes = [
        4 * ["results_mask_sparsity_30.csv"],
        4 * ["results_mask_sparsity_30.csv"],
        2 * ["results_mask_sparsity_0.csv"]
        + 2 * ["results_mask_sparsity_30.csv"]
        + 2 * ["results_mask_sparsity_70.csv"],
        4 * ["results_mask_sparsity_30.csv"],
        6 * ["results_mask_sparsity_30.csv"],
        6 * ["results_mask_sparsity_30.csv"],
    ]
    names = [
        "hyperparams_split_MNIST_lambdas_and_masks",
        "hyperparams_split_MNIST_lambdas_and_betas",
        "hyperparams_split_MNIST_sparsity_and_masks",
        "hyperparams_split_MNIST_betas_and_masks",
        "hyperparams_split_MNIST_hypernetworks_and_masks",
        "hyperparams_split_MNIST_embeddings_and_masks",
    ]
    number_of_tasks = 5
    dataset_name = "Split MNIST"
    for (
        cur_set_of_models,
        cur_settings,
        cur_numbers,
        cur_suffixes,
        cur_name,
    ) in zip(
        selected_hyperparams_models,
        names_of_settings,
        numbers_of_models,
        suffixes,
        names,
    ):
        assert (
            len(cur_set_of_models)
            == len(cur_settings)
            == len(cur_numbers)
            == len(cur_suffixes)
        )
        dataframe_merged = prepare_dataframe_for_multiple_hyperparams_sets(
            cur_set_of_models,
            cur_settings,
            cur_numbers,
            cur_suffixes,
            number_of_tasks,
        )
        plot_different_setups_consecutive_tasks(
            dataframe_merged, dataset_name, plot_path, name=cur_name
        )

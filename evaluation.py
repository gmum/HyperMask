from datasets import (
    set_hyperparameters,
    prepare_permuted_mnist_tasks,
    prepare_split_cifar100_tasks,
    prepare_split_cifar100_tasks_aka_FeCAM,
    prepare_split_mnist_tasks,
)
from main import (
    apply_mask_to_weights_of_network,
    calculate_accuracy,
    get_number_of_batch_normalization_layer,
    load_pickle_file,
    prepare_network_sparsity,
    set_seed,
)
from hypnettorch.mnets import MLP
from hypnettorch.hnets import HMLP
from hypnettorch.mnets.resnet import ResNet
from ResNetF import ResNetF
from MLP_F import MLPFeCAM
from ZenkeNet64 import ZenkeNet
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from numpy.testing import assert_almost_equal
from collections import defaultdict
from sklearn.manifold import TSNE


def load_dataset(dataset, path_to_datasets, hyperparameters):
    if dataset == "PermutedMNIST":
        return prepare_permuted_mnist_tasks(
            path_to_datasets,
            hyperparameters["shape"],
            hyperparameters["number_of_tasks"],
            hyperparameters["padding"],
            hyperparameters["no_of_validation_samples"],
        )
    elif dataset == "CIFAR100":
        return prepare_split_cifar100_tasks(
            path_to_datasets,
            validation_size=hyperparameters["no_of_validation_samples"],
            use_augmentation=hyperparameters["augmentation"],
        )
    elif dataset == "SplitMNIST":
        return prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=hyperparameters["no_of_validation_samples"],
            use_augmentation=hyperparameters["augmentation"],
            number_of_tasks=hyperparameters["number_of_tasks"],
        )
    elif dataset == "CIFAR100_FeCAM_setup":
        return prepare_split_cifar100_tasks_aka_FeCAM(
            path_to_datasets,
            number_of_tasks=hyperparameters["number_of_tasks"],
            no_of_validation_samples_per_class=hyperparameters[
                "no_of_validation_samples_per_class"
            ],
            use_augmentation=hyperparameters["augmentation"],
        )
    else:
        raise ValueError("This dataset is currently not handled!")


def prepare_target_network(hyperparameters, output_shape):
    if hyperparameters["target_network"] == "MLP":
        target_network = MLP(
            n_in=hyperparameters["shape"],
            n_out=output_shape,
            hidden_layers=hyperparameters["target_hidden_layers"],
            use_bias=hyperparameters["use_bias"],
            no_weights=False,
        ).to(hyperparameters["device"])
    elif hyperparameters["target_network"] == "MLP_FeCAM":
        target_network = MLPFeCAM(
            n_in=hyperparameters["shape"],
            n_out=output_shape,
            hidden_layers=hyperparameters["target_hidden_layers"],
            use_bias=hyperparameters["use_bias"],
            no_weights=False,
        ).to(hyperparameters["device"])
    elif hyperparameters["target_network"] == "ResNet":
        target_network = ResNet(
            in_shape=(hyperparameters["shape"], hyperparameters["shape"], 3),
            num_classes=output_shape,
            use_bias=hyperparameters["use_bias"],
            n=hyperparameters["resnet_number_of_layer_groups"],
            k=hyperparameters["resnet_widening_factor"],
            no_weights=False,
            use_batch_norm=hyperparameters["use_batch_norm"],
            bn_track_stats=False,
        ).to(hyperparameters["device"])
    elif hyperparameters["target_network"] == "ResNetF":
        target_network = ResNetF(
            in_shape=(hyperparameters["shape"], hyperparameters["shape"], 3),
            num_classes=output_shape,
            use_bias=hyperparameters["use_bias"],
            n=hyperparameters["resnet_number_of_layer_groups"],
            k=hyperparameters["resnet_widening_factor"],
            no_weights=False,
            use_batch_norm=hyperparameters["use_batch_norm"],
            bn_track_stats=False,
        ).to(hyperparameters["device"])
    elif hyperparameters["target_network"] == "ZenkeNet":
        if hyperparameters["dataset"] in ["CIFAR100", "CIFAR100_FeCAM_setup"]:
            architecture = "cifar"
        elif hyperparameters["dataset"] == "TinyImageNet":
            architecture = "tiny"
        else:
            raise ValueError("This dataset is currently not implemented!")
        target_network = ZenkeNet(
            in_shape=(hyperparameters["shape"], hyperparameters["shape"], 3),
            num_classes=output_shape,
            arch=architecture,
            no_weights=False,
        ).to(hyperparameters["device"])
    else:
        raise NotImplementedError
    return target_network


def prepare_and_load_weights_for_models(
    path_to_stored_networks,
    path_to_datasets,
    number_of_model,
    dataset,
    seed,
    part=0,
    fecam_validation=False,
):
    """
    Prepare hypernetwork and target network and load stored weights
    for both models. Also, load experiment hyperparameters.

    Arguments:
    ----------
       *path_to_stored_networks*: (string) path for all models
                                  located in subfolders
       *number_of_model*: (int) a number of the currently loaded model
       *dataset*: (string) the name of the currently analyzed dataset,
                           one of the followings: 'PermutedMNIST',
                           'SplitMNIST', 'CIFAR100' or 'CIFAR100_FeCAM_setup'
       *seed*: (int) defines a seed value for deterministic calculations
       *part*: (optional int) important for CIFAR100: [0 for ResNet,
               1 for ZenkeNet] and for CIFAR100_FeCAM_setup [0 for ResNet
               with 5 tasks, 1 for ResNet with 10 tasks, 2 for ResNet with
               20 tasks, 3 for ZenkeNet with 5 tasks, 4 for ZenkeNet with
               10 tasks, 5 for ZenkeNet with 20 tasks]
       *fecam_validation*: (optional Boolean) if True, the validation set would
                           have 0 elements because FeCAM uses all training samples;
                           by default it is False. Also, another target network
                           has to be loaded

    Returns a dictionary with the following keys:
       *hypernetwork*: an instance of HMLP class
       *hypernetwork_weights*: loaded weights for the hypernetwork
       *target_network*: an instance of MLP or ResNet class
       *target_network_weights*: loaded weights for the target network
       *hyperparameters*: a dictionary with experiment's hyperparameters
    """
    assert dataset in [
        "PermutedMNIST",
        "CIFAR100",
        "SplitMNIST",
        "CIFAR100_FeCAM_setup",
    ]
    path_to_model = f"{path_to_stored_networks}{number_of_model}/"
    hyperparameters = set_hyperparameters(dataset, grid_search=False, part=part)
    if fecam_validation:
        hyperparameters["no_of_validation_samples_per_class"] = 0
        if dataset == "CIFAR100_FeCAM_setup":
            hyperparameters["target_network"] = "ResNetF"
        elif dataset in ["PermutedMNIST", "SplitMNIST"]:
            hyperparameters["target_network"] = "MLP_FeCAM"
    set_seed(seed)
    # Load proper dataset
    dataset_tasks_list = load_dataset(
        dataset, path_to_datasets, hyperparameters
    )
    output_shape = list(dataset_tasks_list[0].get_train_outputs())[0].shape[0]

    # Build target network
    target_network = prepare_target_network(hyperparameters, output_shape)
    # Build hypernetwork
    no_of_batch_norm_layers = get_number_of_batch_normalization_layer(
        target_network
    )
    if not hyperparameters["use_chunks"]:
        hypernetwork = HMLP(
            target_network.param_shapes[no_of_batch_norm_layers:],
            uncond_in_size=0,
            cond_in_size=hyperparameters["embedding_sizes"][0],
            activation_fn=hyperparameters["activation_function"],
            layers=hyperparameters["hypernetworks_hidden_layers"][0],
            num_cond_embs=hyperparameters["number_of_tasks"],
        ).to(hyperparameters["device"])
    else:
        raise NotImplementedError
    # Load weights
    hnet_weights = load_pickle_file(
        f"{path_to_model}hypernetwork_"
        f'after_{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )
    target_weights_before_masking = load_pickle_file(
        f"{path_to_model}target_network_after_"
        f'{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )
    # Check whether the number of target weights is exactly the same like
    # the loaded weights
    for prepared, loaded in zip(
        [hypernetwork, target_network],
        [hnet_weights, target_weights_before_masking],
    ):
        no_of_loaded_weights = 0
        for item in loaded:
            no_of_loaded_weights += item.shape.numel()
        assert prepared.num_params == no_of_loaded_weights
    return {
        "list_of_CL_tasks": dataset_tasks_list,
        "hypernetwork": hypernetwork,
        "hypernetwork_weights": hnet_weights,
        "target_network": target_network,
        "target_network_weights": target_weights_before_masking,
        "no_of_batch_norm_layers": no_of_batch_norm_layers,
        "hyperparameters": hyperparameters,
    }


def calculate_hypernetwork_output(
    target_network,
    hyperparameters,
    path_to_stored_networks,
    no_of_task_for_loading,
    no_of_task_for_evaluation,
    forward_transfer=False,
):
    no_of_batch_norm_layers = get_number_of_batch_normalization_layer(
        target_network
    )
    if not hyperparameters["use_chunks"]:
        hypernetwork = HMLP(
            target_network.param_shapes[no_of_batch_norm_layers:],
            uncond_in_size=0,
            cond_in_size=hyperparameters["embedding_sizes"][0],
            activation_fn=hyperparameters["activation_function"],
            layers=hyperparameters["hypernetworks_hidden_layers"][0],
            num_cond_embs=hyperparameters["number_of_tasks"],
        ).to(hyperparameters["device"])
        random_hypernetwork = deepcopy(hypernetwork)
    hnet_weights = load_pickle_file(
        f"{path_to_stored_networks}hypernetwork_"
        f"after_{no_of_task_for_loading}_task.pt"
    )
    if forward_transfer:
        assert (no_of_task_for_loading + 1) == no_of_task_for_evaluation
        no_of_task_for_evaluation = no_of_task_for_loading
        # Also embedding from the 'no_of_task_for_loading' will be loaded
        # because embedding from the foregoing task is built randomly
        # (not from zeros!)
    hypernetwork_output = hypernetwork.forward(
        cond_id=no_of_task_for_evaluation, weights=hnet_weights
    )
    random_hypernetwork_output = random_hypernetwork.forward(
        cond_id=no_of_task_for_evaluation
    )
    return random_hypernetwork_output, hypernetwork_output


def load_and_evaluate_networks(
    path_to_datasets,
    path_to_stored_networks,
    dataset,
    tasks_for_loading,
    tasks_for_evaluation,
    seed,
):
    """
    *tasks_for_loading* list of tasks after which
                        network states will be loaded
    *tasks_for_evaluation* list of tasks for evaluation
                           for loaded network in corresponding
                           positions of *tasks_for_loading* list
    *seed* integer / None: has to be exactly the same like
           during model training
    """
    assert len(tasks_for_loading) == len(tasks_for_evaluation)
    hyperparameters = set_hyperparameters(
        dataset,
        grid_search=False,
    )
    # Set seed before drawing the dataset
    if seed is not None:
        set_seed(seed)
    # Load proper dataset
    dataset_tasks_list = load_dataset(
        dataset, path_to_datasets, hyperparameters
    )

    # Build target network
    output_shape = list(dataset_tasks_list[0].get_train_outputs())[0].shape[0]

    results = []
    for no, (loading_task, evaluation_task) in enumerate(
        zip(tasks_for_loading, tasks_for_evaluation)
    ):
        set_seed(no)
        # To generate a new random network
        target_network = prepare_target_network(hyperparameters, output_shape)
        # Store randomly generated weights
        random_model = deepcopy(target_network.weights)
        if evaluation_task == (loading_task + 1):
            forward_transfer = True
        else:
            forward_transfer = False
        # Build hypernetwork
        (
            random_hypernetwork_output,
            hypernetwork_output,
        ) = calculate_hypernetwork_output(
            target_network,
            hyperparameters,
            path_to_stored_networks,
            loading_task,
            evaluation_task,
            forward_transfer=forward_transfer,
        )

        # Prepare sparsed mask and apply it to the target network
        masks = prepare_network_sparsity(
            hypernetwork_output, hyperparameters["sparsity_parameters"]
        )
        random_masks = prepare_network_sparsity(
            random_hypernetwork_output, hyperparameters["sparsity_parameters"]
        )
        random_weights = apply_mask_to_weights_of_network(
            random_model, random_masks
        )
        # Load and prepare target network
        loaded_target_model = load_pickle_file(
            f"{path_to_stored_networks}target_network_after_{loading_task}_task.pt"
        )
        target_weights = apply_mask_to_weights_of_network(
            loaded_target_model, masks
        )

        # During testing a random network on the i-th task we have to compare
        # it with the network trained on the (i-1)-th task
        parameters = {
            "device": hyperparameters["device"],
            "use_batch_norm_memory": False,
            "number_of_task": evaluation_task,
        }
        currently_tested_task = dataset_tasks_list[evaluation_task]
        accuracies = list(
            map(
                lambda x: calculate_accuracy(
                    currently_tested_task,
                    target_network,
                    x,
                    parameters=parameters,
                    evaluation_dataset="test",
                ).item(),
                (target_weights, random_weights),
            )
        )
        results.append(
            [loading_task, evaluation_task, accuracies[0], accuracies[1]]
        )
    dataframe = pd.DataFrame(
        results,
        columns=[
            "loaded_task",
            "evaluated_task",
            "loaded_accuracy",
            "random_net_accuracy",
        ],
    )
    return dataframe


def calculate_backward_transfer(dataframe):
    """
    Calculate backward transfer based on dataframe with results
    containing columns: 'loaded_task', 'evaluated_task',
    'loaded_accuracy' and 'random_net_accuracy'.
    ---
    BWT = 1/(N-1) * sum_{i=1}^{N-1} A_{N,i} - A_{i,i}
    where N is the number of tasks, A_{i,j} is the result
    for the network trained on the i-th task and tested
    on the j-th task.

    Returns a float with backward transfer result.
    """
    backward_transfer = 0
    number_of_last_task = int(dataframe.max()["loaded_task"])
    # Indeed, number_of_last_task represents the number of tasks - 1
    # due to the numeration starting from 0
    for i in range(number_of_last_task + 1):
        trained_on_last_task = dataframe.loc[
            (dataframe["loaded_task"] == number_of_last_task)
            & (dataframe["evaluated_task"] == i)
        ]["loaded_accuracy"].values[0]
        trained_on_the_same_task = dataframe.loc[
            (dataframe["loaded_task"] == i) & (dataframe["evaluated_task"] == i)
        ]["loaded_accuracy"].values[0]
        backward_transfer += trained_on_last_task - trained_on_the_same_task
    backward_transfer /= number_of_last_task
    return backward_transfer


def calculate_forward_transfer(dataframe):
    """
    Calculate forward transfer based on dataframe with results
    containing columns: 'loaded_task', 'evaluated_task',
    'loaded_accuracy' and 'random_net_accuracy'.
    ---
    FWT = 1/(N-1) * sum_{i=1}^{N-1} A_{i-1,i} - R_{i}
    where N is the number of tasks, A_{i,j} is the result
    for the network trained on the i-th task and tested
    on the j-th task and R_{i} is the result for a random
    network evaluated on the i-th task.

    Returns a float with forward transfer result.
    """
    forward_transfer = 0
    number_of_tasks = int(dataframe.max()["loaded_task"] + 1)
    for i in range(1, number_of_tasks):
        extracted_result = dataframe.loc[
            (dataframe["loaded_task"] == (i - 1))
            & (dataframe["evaluated_task"] == i)
        ]
        trained_on_previous_task = extracted_result["loaded_accuracy"].values[0]
        random_network_result = extracted_result["random_net_accuracy"].values[
            0
        ]
        forward_transfer += trained_on_previous_task - random_network_result
    forward_transfer /= number_of_tasks - 1
    return forward_transfer


def calculate_FWT_BWT_different_files(paths, forward=True):
    """
    Calculate mean forward and (or) backward transfer with corresponding
    sample standard deviations based on results saved in .csv files

    Argument:
    ---------
      *paths* (list) contains path to the results files
      *forward* (optional Boolean) defines whether forward transfer will
                be calculated
    Returns:
    --------
      *FWTs* (list of floats) contains consecutive forward transfer values
             or an empty list (if forward is False)
      *BWTs* (list of floats) contains consecutive backward transfer values
    """
    FWTs, BWTs = [], []
    for path in paths:
        dataframe = pd.read_csv(path, sep=";", index_col=0)
        if forward:
            FWTs.append(calculate_forward_transfer(dataframe))
        BWTs.append(calculate_backward_transfer(dataframe))
    if forward:
        print(
            f"Mean forward transfer: {np.mean(FWTs)}, "
            f"population standard deviation: {np.std(FWTs)}"
        )
    print(
        f"Mean backward transfer: {np.mean(BWTs)}, "
        f"population standard deviation: {np.std(BWTs)}"
    )
    return FWTs, BWTs


def evaluate_target_network(
    target_network, network_input, weights, target_network_type, condition=None
):
    """
       *condition* (optional int) the number of the currently tested task
                   for batch normalization

    Returns logits or logits and features (in case of ResNetF)
    """
    if target_network_type == "ResNet":
        assert condition is not None
    if target_network_type == "ResNet":
        # Only ResNet needs information about the currently tested task
        return target_network.forward(
            network_input, weights=weights, condition=condition
        )
    elif target_network_type == "ResNetF":
        # ResNetF returns not only the logits values but also features
        # representation
        logits, features = target_network.forward(
            network_input, weights=weights, condition=condition
        )
        return logits, features
    else:
        return target_network.forward(network_input, weights=weights)


def get_network_logits_for_all_inputs_all_tasks(
    path_to_stored_networks,
    dataset,
    path_to_datasets,
    number_of_model,
    seed,
    sanity_check=True,
):
    """
    Calculate the network output (more specifically, the last layer of the network, before
    the final prediction) for all continual learning tasks and all elements of consecutive
    test sets.

    Returns vectors with output categorized according to the ground truth classes.
    """
    path_to_model = f"{path_to_stored_networks}{number_of_model}/"
    hyperparameters = set_hyperparameters(dataset, grid_search=False)
    # Set seed before drawing the dataset
    if seed is not None:
        set_seed(seed)
    # Load proper dataset
    dataset_tasks_list = load_dataset(
        dataset, path_to_datasets, hyperparameters
    )
    output_shape = list(dataset_tasks_list[0].get_train_outputs())[0].shape[0]
    results_masked, results_without_masks, gt_tasks = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )

    # Build target network
    target_network = prepare_target_network(hyperparameters, output_shape)
    # Build hypernetwork
    no_of_batch_norm_layers = get_number_of_batch_normalization_layer(
        target_network
    )
    if not hyperparameters["use_chunks"]:
        hypernetwork = HMLP(
            target_network.param_shapes[no_of_batch_norm_layers:],
            uncond_in_size=0,
            cond_in_size=hyperparameters["embedding_sizes"][0],
            activation_fn=hyperparameters["activation_function"],
            layers=hyperparameters["hypernetworks_hidden_layers"][0],
            num_cond_embs=hyperparameters["number_of_tasks"],
        ).to(hyperparameters["device"])
    hnet_weights = load_pickle_file(
        f"{path_to_model}hypernetwork_"
        f'after_{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )
    target_weights_without_masking = load_pickle_file(
        f'{path_to_model}target_network_after_{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )
    for task in range(hyperparameters["number_of_tasks"]):
        target_loaded_weights = deepcopy(target_weights_without_masking)
        hypernetwork_output = hypernetwork.forward(
            cond_id=task, weights=hnet_weights
        )
        masks = prepare_network_sparsity(
            hypernetwork_output, hyperparameters["sparsity_parameters"]
        )
        target_masked_weights = apply_mask_to_weights_of_network(
            target_loaded_weights, masks
        )
        currently_tested_task = dataset_tasks_list[task]
        target_network.eval()
        with torch.no_grad():
            input_data = currently_tested_task.get_test_inputs()
            output_data = currently_tested_task.get_test_outputs()
            test_input = currently_tested_task.input_to_torch_tensor(
                input_data, hyperparameters["device"], mode="inference"
            )
            test_output = currently_tested_task.output_to_torch_tensor(
                output_data, hyperparameters["device"], mode="inference"
            )
            gt_classes = test_output.max(dim=1)[1]
            if dataset == "SplitMNIST":
                gt_classes = [x + 2 * task for x in gt_classes]
            target_network_type = hyperparameters["target_network"]
            logits_masked = evaluate_target_network(
                target_network,
                test_input,
                target_masked_weights,
                target_network_type,
                condition=task,
            )
            logits_pure_target = evaluate_target_network(
                target_network,
                test_input,
                target_loaded_weights,
                target_network_type,
                condition=task,
            )
            # Store numbers of classes of the considered samples as well
            # as the number of the current task
            for gt_class, out in zip(gt_classes, logits_masked):
                results_masked[gt_class.item()].append(out.tolist())
            for gt_class, out in zip(gt_classes, logits_pure_target):
                results_without_masks[gt_class.item()].append(out.tolist())
            for gt_class in gt_classes:
                gt_tasks[gt_class.item()].append(task)

        if sanity_check:
            accuracies = list(
                map(
                    lambda x: calculate_accuracy(
                        currently_tested_task,
                        target_network,
                        x,
                        parameters={
                            "device": hyperparameters["device"],
                            "use_batch_norm_memory": False,
                            "number_of_task": task,
                        },
                        evaluation_dataset="test",
                    ).item(),
                    (target_masked_weights, target_loaded_weights),
                )
            )
            print(f"task: {task}, accuracies: {accuracies}")
    extracted_features_masked, extracted_features_pure = [], []
    gt_tasks_masked, gt_tasks_pure = [], []
    gt_tasks_general = []
    for cur_class in list(results_masked.keys()):
        gt_tasks_masked.extend([cur_class] * len(results_masked[cur_class]))
        extracted_features_masked.extend(results_masked[cur_class])
        gt_tasks_pure.extend(
            [cur_class] * len(results_without_masks[cur_class])
        )
        extracted_features_pure.extend(results_without_masks[cur_class])
        gt_tasks_general.extend(gt_tasks[cur_class])
    (
        extracted_features_masked,
        extracted_features_pure,
        gt_tasks_masked,
        gt_tasks_pure,
        gt_tasks_general,
    ) = (
        np.asarray(extracted_features_masked),
        np.asarray(extracted_features_pure),
        np.asarray(gt_tasks_masked),
        np.asarray(gt_tasks_pure),
        np.asarray(gt_tasks_general),
    )
    np.savez_compressed(
        f"{dataset}_{number_of_model}_outputs_test",
        features_masked_target=extracted_features_masked,
        features_pure_target=extracted_features_pure,
        gt_classes_masked_target=gt_tasks_masked,
        gt_classes_pure_target=gt_tasks_pure,
        gt_tasks=gt_tasks_general,
    )
    return (
        extracted_features_masked,
        extracted_features_pure,
        gt_tasks_masked,
        gt_tasks_pure,
        gt_tasks_general,
    )


def plot_accuracy_one_setting(
    path_to_stored_networks,
    no_of_models_for_loading,
    suffix,
    dataset_name,
    folder="./Plots/",
):
    """
    Plot average accuracy for the best setting of the selected method
    for a given dataset. On the plot results after the training of models
    for all tasks are compared with the corresponding results just after
    the training of models.

    Arguments:
    ----------
       *path_to_stored_networks*: (string) path to the folder with results
                                  for all models
       *no_of_models_for_loading*: (list) contains names of subfolders
                                   with consecutive models
       *suffix*: (string) name of the file with results; single files
                 are located in consecutive subfolders
       *dataset_name*: (string) name of the currently analyzed dataset
       *folder*: (optional string) name of the folder for saving results
    """
    individual_results_just_after, individual_results_after_all = [], []
    # Load results for all models: results after learning of all tasks
    # as well as just after learning consecutive tasks
    for model in no_of_models_for_loading:
        accuracy_results = pd.read_csv(
            f"{path_to_stored_networks}{model}/{suffix}", sep=";", index_col=0
        )
        just_after_training = accuracy_results.loc[
            accuracy_results["after_learning_of_task"]
            == accuracy_results["tested_task"]
        ]
        after_all_training_sessions = accuracy_results.loc[
            accuracy_results["after_learning_of_task"]
            == accuracy_results.max()["after_learning_of_task"]
        ]
        individual_results_just_after.append(just_after_training)
        individual_results_after_all.append(after_all_training_sessions)
    dataframe_just_after = pd.concat(
        individual_results_just_after, ignore_index=True, axis=0
    )
    dataframe_just_after["after_learning_of_task"] = "just after training"
    dataframe_after_all = pd.concat(
        individual_results_after_all, ignore_index=True, axis=0
    )
    dataframe_after_all[
        "after_learning_of_task"
    ] = "after training of all tasks"
    dataframe = pd.concat(
        [dataframe_just_after, dataframe_after_all], axis=0, ignore_index=True
    )
    dataframe = dataframe.rename(
        columns={"after_learning_of_task": "evaluation"}
    )
    dataframe["tested_task"] += 1
    tasks = individual_results_just_after[0]["tested_task"].values + 1
    ax = sns.relplot(
        data=dataframe,
        x="tested_task",
        y="accuracy",
        kind="line",
        hue="evaluation",
        height=3,
        aspect=1.5,
    )
    # mean and 95% confidence intervals
    if dataset_name in [
        "Permuted MNIST (10 tasks)",
        "Split MNIST",
        "CIFAR-100 (ResNet)",
        "CIFAR-100 (ZenkeNet)",
    ]:
        ax.set(xticks=tasks, xlabel="Number of task", ylabel="Accuracy [%]")
        legend_fontsize = 11
        if dataset_name == "Permuted MNIST (10 tasks)":
            legend_position = "upper right"
            bbox_position = (0.65, 0.95)
        else:
            legend_position = "lower center"
            bbox_position = (0.5, 0.17)
            if dataset_name == "Split MNIST":
                legend_fontsize = 10
    elif dataset_name == "Permuted MNIST (100 tasks)":
        legend_position = "lower right"
        bbox_position = (0.65, 0.2)
        tasks = np.arange(0, 101, step=10)
        tasks[0] = 1
        ax.set(xticks=tasks, xlabel="Number of task", ylabel="Accuracy [%]")
    else:
        raise ValueError("Not implemented dataset!")
    sns.move_legend(
        ax,
        legend_position,
        bbox_to_anchor=bbox_position,
        fontsize=legend_fontsize,
        title="",
    )
    plt.title(f"Results for {dataset_name}", fontsize=11)
    plt.xlabel("Number of task", fontsize=11)
    plt.ylabel("Accuracy [%]", fontsize=11)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        f"{folder}mean_accuracy_best_setting_{dataset_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_single_model_accuracy_one_setting(
    path_to_stored_networks,
    no_of_models_for_loading,
    suffix,
    dataset_name,
    folder="./Plots/",
    legend=True,
):
    """
    Plot accuracy just after training of consecutive tasks and after
    training of all tasks for a single model of the selected method
    for a given dataset.
    This function is especially prepared for TinyImageNet

    Arguments:
    ----------
       *path_to_stored_networks*: (string) path to the folder with results
                                  for all models
       *no_of_models_for_loading*: (list) contains names of subfolders
                                   with consecutive models
       *suffix*: (string) name of the file with results; single files
                 are located in consecutive subfolders
       *dataset_name*: (string) name of the currently analyzed dataset
       *folder*: (optional string) name of the folder for saving results
       *legend*: (optional Boolean value) defines whether a legend should
                 be inserted
    """
    name_to_save = (
        dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
    )
    for model in no_of_models_for_loading:
        accuracy_results = pd.read_csv(
            f"{path_to_stored_networks}{model}/{suffix}", sep=";", index_col=0
        )
        just_after_training = accuracy_results.loc[
            accuracy_results["after_learning_of_task"]
            == accuracy_results["tested_task"]
        ].copy()
        just_after_training.reset_index(inplace=True, drop=True)
        just_after_training.loc[
            :, "after_learning_of_task"
        ] = "just after training"

        after_all_training_sessions = accuracy_results.loc[
            accuracy_results["after_learning_of_task"]
            == accuracy_results.max()["after_learning_of_task"]
        ].copy()
        after_all_training_sessions.reset_index(inplace=True, drop=True)
        after_all_training_sessions.loc[
            :, "after_learning_of_task"
        ] = "after training of all tasks"
        dataframe = pd.concat(
            [just_after_training, after_all_training_sessions],
            axis=0,
            ignore_index=True,
        )
        dataframe = dataframe.rename(
            columns={"after_learning_of_task": "evaluation"}
        )
        dataframe["tested_task"] += 1
        if "ImageNet" in dataset_name:
            tasks = [0, 4, 9, 14, 19, 24, 29, 34, 39]
        else:
            tasks = just_after_training["tested_task"].values + 1
        values = dataframe["accuracy"].values
        plt.figure(figsize=(5.5, 2.5))
        ax = sns.barplot(
            data=dataframe,
            x="tested_task",
            y="accuracy",
            hue="evaluation",
        )
        ax.set(
            xticks=tasks,
            ylim=(np.min(values) - 3,
                  np.max(values) + 3)
        )
        if legend:
            sns.move_legend(
                ax,
                "upper left",
                bbox_to_anchor=(0, 1.3),
                fontsize=10,
                ncol=2,
                title="",
            )
        else:
            ax._legend.remove()
            plt.title(f"Results for {dataset_name}", fontsize=10, pad=20)
        plt.xlabel("Number of task", fontsize=10)
        plt.ylabel("Accuracy [%]", fontsize=10)
        plt.tight_layout()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(
            f"{folder}accuracy_model_{model}_{name_to_save}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def prepare_tSNE_plot(
    features, gt_classes, name, dataset, label="task", title=None
):
    """
    Prepare a t-SNE plot to produce an embedded version of features.

    Arguments:
    ----------
      *features* -
    """
    if dataset == "PermutedMNIST":
        fig, ax = plt.subplots(figsize=(4, 4))
        s = 0.1
        alpha = None
        legend_loc = "best"
        bbox_to_anchor = None
        fontsize = 9
        legend_fontsize = "medium"
        legend_titlefontsize = None
    elif dataset == "SplitMNIST":
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.tick_params(axis="x", labelsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        s = 0.5
        alpha = 0.75
        legend_loc = "center"
        bbox_to_anchor = (1.15, 0.5)
        fontsize = 8
        legend_fontsize = "small"
        legend_titlefontsize = "small"
    # legend position outside for Split
    values = np.unique(gt_classes)
    for i in values:
        plt.scatter(
            features[gt_classes == i, 0],
            features[gt_classes == i, 1],
            label=i,
            rasterized=True,
            s=s,
            alpha=alpha,
        )
    lgnd = plt.legend(
        title=label,
        loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=legend_fontsize,
        title_fontsize=legend_titlefontsize,
        handletextpad=0.1,
    )
    for i in range(values.shape[0]):
        lgnd.legendHandles[i]._sizes = [20]
    plt.title(title, fontsize=fontsize)
    plt.xlabel("t-SNE embedding first dimension", fontsize=fontsize)
    plt.ylabel("t-SNE embedding second dimension", fontsize=fontsize)
    plt.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def test_calculate_transfers():
    """
    Unittest of calculate_backward_transfer() and calculate_forward_transfer()
    """
    test_results_1 = [
        [0, 0, 80, 15],
        [0, 1, 20, 15],
        [0, 2, 13, 16],
        [0, 3, 19, 18],
        [1, 0, 35, 17],
        [1, 1, 85, 10],
        [1, 2, 20, 18],
        [1, 3, 18, 15],
        [2, 0, 30, 12],
        [2, 1, 10, 15],
        [2, 2, 70, 16],
        [2, 3, 25, 17],
        [3, 0, 35, 17],
        [3, 1, 40, 21],
        [3, 2, 25, 15],
        [3, 3, 90, 10],
    ]
    test_dataframe_1 = pd.DataFrame(
        test_results_1,
        columns=[
            "loaded_task",
            "evaluated_task",
            "loaded_accuracy",
            "random_net_accuracy",
        ],
    )
    output_BWT_1 = calculate_backward_transfer(test_dataframe_1)
    gt_BWT_1 = -45
    assert_almost_equal(output_BWT_1, gt_BWT_1)
    output_FWT_1 = calculate_forward_transfer(test_dataframe_1)
    gt_FWT_1 = 5
    assert_almost_equal(output_FWT_1, gt_FWT_1)


if __name__ == "__main__":
    test_calculate_transfers()
    # Calculate forward and backward transfer
    ###################################################################
    # PermutedMNIST / SplitMNIST, HNET:
    method = "HYPERMASK"  # 'HNET' or 'HYPERMASK'
    dataset = "TinyImageNet_ZenkeNet"
    if dataset == "PermutedMNIST":
        prefix = "PMNIST"
        number_of_runs = 5
    elif dataset == "SplitMNIST":
        prefix = "splitMNIST"
        number_of_runs = 5
    elif dataset == "PermutedMNIST100":
        prefix = "P100MNIST"
        number_of_runs = 3
    elif dataset == "TinyImageNet_ResNet":
        prefix = "Tiny_ResNet"
        number_of_runs = 5
    elif dataset == "TinyImageNet_ZenkeNet":
        prefix = "Tiny_ZenkeNet"
        number_of_runs = 5
    path_to_transfer_calculations = "./transfer/"
    forward = False
    paths = []
    for seed in range(1, number_of_runs + 1):
        paths.append(
            f"{path_to_transfer_calculations}{method}_seed_{seed}_{prefix}.csv"
        )
    FWTs, BWTs = calculate_FWT_BWT_different_files(paths, forward=forward)
    ###################################################################

    path_to_datasets = "./Data"
    dataset = "PermutedMNIST"
    path_to_stored_networks = "./"
    for no_of_model in range(5):
        path_for_single_model = f"{path_to_stored_networks}{no_of_model}/"
        # Setup for PermutedMNIST/CIFAR_100
        if dataset in ["PermutedMNIST", "CIFAR_100"]:
            tasks_for_loading = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
            ]
            tasks_for_evaluation = [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
            ]
        else:
            raise NotImplementedError
        results = load_and_evaluate_networks(
            path_to_datasets,
            path_for_single_model,
            dataset,
            tasks_for_loading,
            tasks_for_evaluation,
            seed=no_of_model + 1,
        )
        results.to_csv(
            f"HYPERMASK_seed_{no_of_model + 1}_{dataset}.csv", sep=";"
        )

    ###################################################################
    # Prepare results for the best setting
    # Permuted MNIST, 10 tasks
    no_of_models_for_loading = [0, 1, 2, 3, 4]
    suffix = "results_mask_sparsity_0.csv"
    dataset_name = "Permuted MNIST (10 tasks)"
    # Split MNIST
    no_of_models_for_loading = [0, 1, 2, 3, 4]
    suffix = "results_mask_sparsity_30.csv"
    dataset_name = "Split MNIST"
    # Permuted MNIST, 100 tasks
    no_of_models_for_loading = [0, 1, 2]
    suffix = "results_mask_sparsity_0.csv"
    dataset_name = "Permuted MNIST (100 tasks)"
    # CIFAR, 10 tasks, ResNet
    no_of_models_for_loading = [0, 1, 2, 3, 4]
    suffix = "results_mask_sparsity_0.csv"
    dataset_name = "CIFAR-100 (ResNet)"
    # CIFAR, 10 tasks, ZenkeNet
    no_of_models_for_loading = [0, 1, 2, 3, 4]
    suffix = "results_mask_sparsity_0.csv"
    dataset_name = "CIFAR-100 (ZenkeNet)"
    plot_accuracy_one_setting(
        path_to_stored_networks,
        no_of_models_for_loading,
        suffix,
        dataset_name,
        folder="./Plots/",
    )
    # TinyImageNet, 40 tasks, ResNet
    no_of_models_for_loading = [0, 1, 2, 3, 4]
    suffix = "results_mask_sparsity_0.csv"
    dataset_name = "Tiny ImageNet (ResNet)"
    legend = True
    # TinyImageNet, 40 tasks, ZenkeNet
    no_of_models_for_loading = [0, 1, 2, 3, 4]
    suffix = "results_mask_sparsity_50.csv"
    dataset_name = "Tiny ImageNet (ZenkeNet)"
    legend = True
    plot_single_model_accuracy_one_setting(
        path_to_stored_networks,
        no_of_models_for_loading,
        suffix,
        dataset_name,
        folder="./Plots/",
        legend=legend,
    )
    ###################################################################
    # Prepare tSNE plots
    datasets = [
        "PermutedMNIST",
        "SplitMNIST",
    ]  # 'PermutedMNIST' or 'SplitMNIST'
    # datasets = ['SplitMNIST']
    for dataset in datasets:
        path_to_datasets = "./Data"
        if dataset == "PermutedMNIST":
            number_of_model = 4
            seed = 5
        elif dataset == "SplitMNIST":
            number_of_model = 1
            seed = 2
        sanity_check = True

        (
            features_masked,
            features_pure,
            gt_class_masked,
            gt_class_pure,
            gt_tasks,
        ) = get_network_logits_for_all_inputs_all_tasks(
            path_to_stored_networks,
            dataset,
            path_to_datasets,
            number_of_model,
            seed,
            sanity_check=sanity_check,
        )

        set_seed(seed)
        X_embedded_masked = TSNE(
            n_components=2, random_state=seed
        ).fit_transform(features_masked)
        X_embedded_pure = TSNE(n_components=2, random_state=seed).fit_transform(
            features_pure
        )
        title_masked = "t-SNE result for the masked target model"
        title_pure = (
            "t-SNE result for the pure target model (without hypernetwork)"
        )
        for divide_by_class in [True, False]:
            if divide_by_class:
                gt_for_masked_part = gt_class_masked
                gt_for_pured_part = gt_class_pure
                label = "class"
            else:
                gt_for_masked_part = gt_tasks + 1
                gt_for_pured_part = gt_tasks + 1
                label = "task"
            prepare_tSNE_plot(
                X_embedded_masked,
                gt_for_masked_part,
                f"tSNE_masked_{dataset}_{number_of_model}_divide_by_{label}",
                dataset,
                label=label,
                title=title_masked,
            )
            prepare_tSNE_plot(
                X_embedded_pure,
                gt_for_pured_part,
                f"tSNE_pure_{dataset}_{number_of_model}_divide_by_{label}",
                dataset,
                label=label,
                title=title_pure,
            )

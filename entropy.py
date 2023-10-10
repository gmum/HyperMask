import torch
import pandas as pd
import torch.nn.functional as F
from copy import deepcopy

from datasets import set_hyperparameters
from main import (
    apply_mask_to_weights_of_network,
    get_number_of_batch_normalization_layer,
    prepare_network_sparsity,
    set_seed,
)
from evaluation import (
    evaluate_target_network,
    load_dataset,
    load_pickle_file,
    prepare_target_network,
)
from hypnettorch.hnets import HMLP


def prepare_and_load_weights_for_models(
    path_to_stored_networks, path_to_datasets, number_of_model, dataset, part=0
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
                           'SplitMNIST' or 'CIFAR100'
       *part*: (optional int) important for CIFAR100: 0 for ResNet,
               1 for ZenkeNet

    Returns a dictionary with the following keys:
       *hypernetwork*: an instance of HMLP class
       *hypernetwork_weights*: loaded weights for the hypernetwork
       *target_network*: an instance of MLP or ResNet class
       *target_network_weights*: loaded weights for the target network
       *hyperparameters*: a dictionary with experiment's hyperparameters
    """
    assert dataset in ["PermutedMNIST", "CIFAR100", "SplitMNIST"]
    path_to_model = f"{path_to_stored_networks}{number_of_model}/"
    hyperparameters = set_hyperparameters(dataset, grid_search=False, part=part)
    seed = number_of_model + 1
    set_seed(seed)
    # Load proper dataset
    dataset_tasks_list = load_dataset(dataset, path_to_datasets, hyperparameters)
    output_shape = list(dataset_tasks_list[0].get_train_outputs())[0].shape[0]

    # Build target network
    target_network = prepare_target_network(hyperparameters, output_shape)
    # Build hypernetwork
    no_of_batch_norm_layers = get_number_of_batch_normalization_layer(target_network)
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
    return {
        "list_of_CL_tasks": dataset_tasks_list,
        "hypernetwork": hypernetwork,
        "hypernetwork_weights": hnet_weights,
        "target_network": target_network,
        "target_network_weights": target_weights_before_masking,
        "hyperparameters": hyperparameters,
    }


def get_task_and_class_prediction_based_on_logits(
    number_of_tasks, inferenced_logits_of_all_tasks
):
    """
    Get task prediction for consecutive samples based on entropy values
    of the output classification layer of the target network.

    Arguments:
    ----------
       *number_of_tasks*: (int) number of CL tasks
       *inferenced_logits_of_all_tasks*: shape: (number of tasks,
                            number of samples, number of output heads)

    Returns:
    --------
       *predicted_tasks*: torch.Tensor with the prediction of tasks for
                          consecutive samples
       *predicted_classes*: torch.Tensor with the prediction of classes for
                            consecutive samples.
       Positions of samples in the two above Tensors are the same.
    """
    predicted_classes, predicted_tasks = [], []
    number_of_samples = inferenced_logits_of_all_tasks.shape[1]
    for no_of_sample in range(number_of_samples):
        task_entropies = torch.zeros(number_of_tasks)
        all_task_single_output_sample = inferenced_logits_of_all_tasks[
            :, no_of_sample, :
        ]
        all_task_single_output_sample = F.softmax(all_task_single_output_sample, dim=1)
        # Calculate entropy based on results from all tasks
        for no_of_inferred_task in range(task_entropies.shape[0]):
            task_entropies[no_of_inferred_task] = -1 * torch.sum(
                all_task_single_output_sample[no_of_inferred_task]
                * torch.log(all_task_single_output_sample[no_of_inferred_task])
            )
        selected_task_id = torch.argmin(task_entropies)
        predicted_tasks.append(selected_task_id.item())
        target_output = all_task_single_output_sample[selected_task_id]
        predicted_classes.append(target_output.argmax().item())
    predicted_tasks = torch.Tensor(predicted_tasks)
    predicted_classes = torch.Tensor(predicted_classes)
    return predicted_tasks, predicted_classes


def calculate_entropy_and_predict_classes_automatically(experiment_models):
    """
    Select the target task automatically and calculate accuracy for
    consecutive samples

    Arguments:
    ----------
    *experiment_models*: A dictionary with the following keys:
       *hypernetwork*: an instance of HMLP class
       *hypernetwork_weights*: loaded weights for the hypernetwork
       *target_network*: an instance of MLP or ResNet class
       *target_network_weights*: loaded weights for the target network
       *hyperparameters*: a dictionary with experiment's hyperparameters

    Returns Pandas Dataframe with results for the selected model.
    """
    hypernetwork = experiment_models["hypernetwork"]
    hypernetwork_weights = experiment_models["hypernetwork_weights"]
    target_network = experiment_models["target_network"]
    target_weights = experiment_models["target_network_weights"]
    hyperparameters = experiment_models["hyperparameters"]
    dataset_CL_tasks = experiment_models["list_of_CL_tasks"]
    hypernetwork.eval()
    target_network.eval()

    results = []
    for task in range(hyperparameters["number_of_tasks"]):
        target_loaded_weights = deepcopy(target_weights)
        gt_tasks = []
        # Iteration over real (GT) tasks
        currently_tested_task = dataset_CL_tasks[task]
        input_data = currently_tested_task.get_test_inputs()
        output_data = currently_tested_task.get_test_outputs()
        test_input = currently_tested_task.input_to_torch_tensor(
            input_data, hyperparameters["device"], mode="inference"
        )
        test_output = currently_tested_task.output_to_torch_tensor(
            output_data, hyperparameters["device"], mode="inference"
        )
        gt_classes = test_output.max(dim=1)[1]
        # if dataset == 'SplitMNIST':
        #     gt_classes = [x + 2 * task for x in gt_classes]
        target_network_type = hyperparameters["target_network"]
        gt_tasks.append([task] * output_data.shape[0])

        with torch.no_grad():
            logits_outputs = []
            for inferenced_task in range(hyperparameters["number_of_tasks"]):
                # Try to predict task for all samples from "task"
                hypernetwork_output = hypernetwork.forward(
                    cond_id=inferenced_task, weights=hypernetwork_weights
                )
                masks = prepare_network_sparsity(
                    hypernetwork_output, hyperparameters["sparsity_parameters"]
                )
                target_masked_weights = apply_mask_to_weights_of_network(
                    target_loaded_weights, masks
                )
                logits_masked = evaluate_target_network(
                    target_network,
                    test_input,
                    target_masked_weights,
                    target_network_type,
                    condition=inferenced_task,
                )
                logits_outputs.append(logits_masked)
            all_inferenced_tasks = torch.stack(logits_outputs)
            # Sizes of consecutive dimensions represent:
            # number of tasks x number of samples x number of output heads
        (
            predicted_tasks,
            predicted_classes,
        ) = get_task_and_class_prediction_based_on_logits(
            hyperparameters["number_of_tasks"], all_inferenced_tasks
        )
        task_prediction_accuracy = (
            torch.sum(predicted_tasks == task).float()
            * 100.0
            / predicted_tasks.shape[0]
        ).item()
        sample_prediction_accuracy = (
            torch.sum(predicted_classes == gt_classes.cpu()).float()
            * 100.0
            / gt_classes.shape[0]
        ).item()
        results.append([task, task_prediction_accuracy, sample_prediction_accuracy])
    results = pd.DataFrame(
        results, columns=["task", "task_prediction_acc", "class_prediction_acc"]
    )
    results.to_csv(
        f"{path_to_stored_networks}entropy_statistics_{number_of_model}.csv", sep=";"
    )
    return results


if __name__ == "__main__":
    # Options for *dataset*:
    # 'PermutedMNIST', 'SplitMNIST'. 'CIFAR100'
    dataset = "CIFAR100"
    path_to_stored_networks = "./"
    path_to_datasets = "./Data/"

    if dataset == "PermutedMNIST":
        part = 0
    elif dataset == "SplitMNIST":
        part = 0
    elif dataset == "CIFAR100":
        # ResNet
        part = 0
        # ZenkeNet
        # part = 1
    number_of_models = 5
    summary_of_results = []
    for number_of_model in range(number_of_models):
        experiment_models = prepare_and_load_weights_for_models(
            path_to_stored_networks,
            path_to_datasets,
            number_of_model,
            dataset,
            part=part,
        )
        results = calculate_entropy_and_predict_classes_automatically(experiment_models)
        summary_of_results.append(
            [
                number_of_model,
                results["task_prediction_acc"].mean(),
                results["class_prediction_acc"].mean(),
            ]
        )
    summary_of_results = pd.DataFrame(
        summary_of_results,
        columns=[
            "no_of_model",
            "mean_model_task_prediction_accuracy",
            "mean_model_class_prediction_accuracy",
        ],
    )
    summary_of_results.to_csv(
        f"{path_to_stored_networks}entropy_mean_results.csv", sep=";"
    )

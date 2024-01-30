import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

from evaluation import (
    prepare_and_load_weights_for_models,
)
from FeCAM import (
    extract_test_set_from_all_tasks,
    extract_test_set_from_single_task,
    translate_output_CIFAR_classes,
    get_target_network_representation,
    evaluate_hypermask_with_selected_embedding,
    translate_output_MNIST_classes,
)


def get_task_and_class_prediction_based_on_logits(
    inferenced_logits_of_all_tasks, setup, dataset
):
    """
    Get task prediction for consecutive samples based on entropy values
    of the output classification layer of the target network.

    Arguments:
    ----------
       *inferenced_logits_of_all_tasks*: shape: (number of tasks,
                            number of samples, number of output heads)
       *setup*: (int) defines how many tasks were performed in this
                experiment (in total)
       *dataset*: (str) name of the dataset for proper class translation

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
        task_entropies = torch.zeros(inferenced_logits_of_all_tasks.shape[0])
        all_task_single_output_sample = inferenced_logits_of_all_tasks[
            :, no_of_sample, :
        ]
        # Calculate entropy based on results from all tasks
        for no_of_inferred_task in range(task_entropies.shape[0]):
            softmaxed_inferred_task = F.softmax(
                all_task_single_output_sample[no_of_inferred_task], dim=-1
            )
            task_entropies[no_of_inferred_task] = -1 * torch.sum(
                softmaxed_inferred_task * torch.log(softmaxed_inferred_task)
            )
        selected_task_id = torch.argmin(task_entropies)
        predicted_tasks.append(selected_task_id.item())
        target_output = all_task_single_output_sample[selected_task_id.item()]
        output_relative_class = target_output.argmax().item()
        if dataset == "CIFAR100_FeCAM_setup":
            output_absolute_class = translate_output_CIFAR_classes(
                [output_relative_class], setup, selected_task_id.item()
            )
        elif dataset in ["PermutedMNIST", "SplitMNIST"]:
            mode = "permuted" if dataset == "PermutedMNIST" else "split"
            output_absolute_class = translate_output_MNIST_classes(
                [output_relative_class], selected_task_id.item(), mode=mode
            )
        else:
            raise ValueError("Wrong name of the dataset!")
        predicted_classes.append(output_absolute_class)
    predicted_tasks = torch.tensor(predicted_tasks, dtype=torch.int32)
    predicted_classes = torch.tensor(predicted_classes, dtype=torch.int32)
    return predicted_tasks, predicted_classes


def calculate_entropy_and_predict_classes_with_FeCAM(experiment_models):
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
       *list_of_CL_tasks*: list of objects containing consecutive tasks
       *no_of_batch_norm_layers*: the number of batch normalization layers
                                  in the target network
       *hyperparameters*: a dictionary with experiment's hyperparameters
       *batch_inference_size*: int related to the batch size during inference
       *setup*: int related to the total number of tasks in a given experiment

    List with the following results for the selected model:
       - consecutive task prediction accuracies,
       - consecutive class prediction accuracies based on logits,
       - consecutive class prediction accuracies based on forward propagation,
       - mean task prediction accuracy (and std. dev.),
       - mean class prediction accuracy based on logits (and std. dev.),
       - mean class prediction accuracy based on forward prop. (and std. dev.).
    """
    hypernetwork = experiment_models["hypernetwork"]
    hypernetwork_weights = experiment_models["hypernetwork_weights"]
    target_network = experiment_models["target_network"]
    target_weights = experiment_models["target_network_weights"]
    hyperparameters = experiment_models["hyperparameters"]
    dataset_CL_tasks = experiment_models["list_of_CL_tasks"]
    no_of_batch_norm_layers = experiment_models["no_of_batch_norm_layers"]
    target_network_type = hyperparameters["target_network"]
    hypernetwork.eval()
    target_network.eval()

    task_predictions, class_predictions_logits, class_predictions_forward = (
        [],
        [],
        [],
    )
    for number_of_incremental_tasks in range(1, experiment_models["setup"] + 1):
        target_loaded_weights = deepcopy(target_weights)
        X_test, y_test, gt_tasks_test = extract_test_set_from_all_tasks(
            dataset_CL_tasks,
            number_of_incremental_tasks,
            experiment_models["setup"],
            hyperparameters["device"],
        )
        results_classes, results_tasks, sanity_check_cls = [], [], []
        no_of_batches = (
            X_test.shape[0] // experiment_models["batch_inference_size"]
        )
        if X_test.shape[0] % experiment_models["batch_inference_size"] > 0.0:
            no_of_batches += 1
        for i in range(no_of_batches):
            X_sample = X_test[
                (experiment_models["batch_inference_size"] * i) : (
                    experiment_models["batch_inference_size"] * (i + 1)
                )
            ]
            logits_outputs_for_different_tasks = []
            for inferenced_task in range(number_of_incremental_tasks):
                # Try to predict task for all samples from "task"
                logits_masked = get_target_network_representation(
                    hypernetwork,
                    hypernetwork_weights,
                    target_network,
                    target_loaded_weights,
                    target_network_type,
                    X_sample,
                    hyperparameters["sparsity_parameters"][0],
                    no_of_batch_norm_layers,
                    inferenced_task,
                )
                logits_masked = logits_masked[0]
                logits_outputs_for_different_tasks.append(logits_masked)
            # shape: number of incremental tasks x number of samples x
            # output classification layer shape (number of classes)
            logits_outputs_for_different_tasks = torch.stack(
                logits_outputs_for_different_tasks
            )
            # METHOD 1): Select tasks and classes based on the previously
            # calculated logits values. In this case, it is not necessary
            # to perform another forward propagation through the network
            # because we evaluated network outputs for each embedding.
            (
                predicted_tasks,
                predicted_classes,
            ) = get_task_and_class_prediction_based_on_logits(
                logits_outputs_for_different_tasks,
                experiment_models["setup"],
                experiment_models["hyperparameters"]["dataset"],
            )
            # METHOD 2): Select classes based on another forward propagation
            # with the selected embedding. The results are varying depending
            # on the batch size when the network uses batch normalization.
            sanity = evaluate_hypermask_with_selected_embedding(
                hypernetwork,
                hypernetwork_weights,
                target_network,
                target_weights,
                target_network_type,
                experiment_models["hyperparameters"]["dataset"],
                X_sample,
                predicted_tasks.flatten().numpy(),
                hyperparameters["sparsity_parameters"][0],
                no_of_batch_norm_layers,
                number_of_incremental_tasks,
                experiment_models["setup"],
            )
            sanity_check_cls.append(sanity)
            results_classes.append(predicted_classes.flatten())
            results_tasks.append(predicted_tasks)
        results_classes = torch.cat(results_classes, dim=0).numpy()
        results_tasks = torch.cat(results_tasks, dim=0).numpy()
        sanity_check_cls = np.concatenate(sanity_check_cls)
        task_prediction_accuracy = (
            np.sum(results_tasks == gt_tasks_test)
            * 100.0
            / results_tasks.shape[0]
        ).item()
        task_predictions.append(task_prediction_accuracy)
        sample_prediction_accuracy = (
            np.sum(results_classes == y_test) * 100.0 / results_classes.shape[0]
        ).item()
        class_predictions_logits.append(sample_prediction_accuracy)
        sample_prediction_forward_accuracy = (
            np.sum(sanity_check_cls == y_test) * 100.0 / y_test.shape[0]
        )
        class_predictions_forward.append(sample_prediction_forward_accuracy)
        print(f"After {number_of_incremental_tasks} incremental task.")
        print(f"Task prediction accuracy: {task_prediction_accuracy}")
        print(
            f"Class prediction accuracy, logits: {sample_prediction_accuracy}"
        )
        print(
            "Class prediction accuracy, forward: "
            f"{sample_prediction_forward_accuracy}"
        )
    summary_of_results = [
        task_predictions,
        class_predictions_logits,
        class_predictions_forward,
    ]
    for element in [
        task_predictions,
        class_predictions_logits,
        class_predictions_forward,
    ]:
        summary_of_results.extend((np.mean(element), np.std(element)))
    return summary_of_results


def calculate_entropy_and_predict_classes_separately(experiment_models):
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
       *dataset_CL_tasks*: list of objects containing consecutive tasks

    Returns Pandas Dataframe with results for the selected model.
    """
    hypernetwork = experiment_models["hypernetwork"]
    hypernetwork_weights = experiment_models["hypernetwork_weights"]
    target_network = experiment_models["target_network"]
    target_weights = experiment_models["target_network_weights"]
    hyperparameters = experiment_models["hyperparameters"]
    dataset_CL_tasks = experiment_models["list_of_CL_tasks"]
    dataset_name = experiment_models["hyperparameters"]["dataset"]
    target_network_type = hyperparameters["target_network"]
    saving_folder = hyperparameters["saving_folder"]
    if "no_of_batch_norm_layers" in experiment_models:
        no_of_batch_norm_layers = experiment_models["no_of_batch_norm_layers"]
    else:
        no_of_batch_norm_layers = 0

    hypernetwork.eval()
    target_network.eval()

    results = []
    for task in range(hyperparameters["number_of_tasks"]):
        target_loaded_weights = deepcopy(target_weights)
        X_test, y_test, gt_tasks = extract_test_set_from_single_task(
            dataset_CL_tasks, task, dataset_name, hyperparameters["device"]
        )

        with torch.no_grad():
            logits_outputs_for_different_tasks = []
            for inferenced_task in range(hyperparameters["number_of_tasks"]):
                # Try to predict task for all samples from "task"
                logits_masked = get_target_network_representation(
                    hypernetwork,
                    hypernetwork_weights,
                    target_network,
                    target_loaded_weights,
                    target_network_type,
                    X_test,
                    hyperparameters["sparsity_parameters"][0],
                    no_of_batch_norm_layers,
                    inferenced_task,
                )
                logits_masked = logits_masked[0]
                logits_outputs_for_different_tasks.append(logits_masked)
            all_inferenced_tasks = torch.stack(
                logits_outputs_for_different_tasks
            )
            # Sizes of consecutive dimensions represent:
            # number of tasks x number of samples x number of output heads
        (
            predicted_tasks,
            predicted_classes,
        ) = get_task_and_class_prediction_based_on_logits(
            all_inferenced_tasks,
            hyperparameters["number_of_tasks"],
            dataset_name,
        )
        predicted_classes = predicted_classes.flatten().numpy()
        task_prediction_accuracy = (
            torch.sum(predicted_tasks == task).float()
            * 100.0
            / predicted_tasks.shape[0]
        ).item()
        print(f"task prediction accuracy: {task_prediction_accuracy}")
        sample_prediction_accuracy = (
            np.sum(predicted_classes == y_test) * 100.0 / y_test.shape[0]
        ).item()
        print(f"sample prediction accuracy: {sample_prediction_accuracy}")
        results.append(
            [task, task_prediction_accuracy, sample_prediction_accuracy]
        )
    results = pd.DataFrame(
        results, columns=["task", "task_prediction_acc", "class_prediction_acc"]
    )
    results.to_csv(
        f"{saving_folder}entropy_statistics_{number_of_model}.csv", sep=";"
    )
    return results


if __name__ == "__main__":
    # The results are varying depending on the batch sizes due to the fact
    # that batch normalization is turned on in ResNet. We selected 2000 as
    # the test set size to ensure that it is derived to the network
    # in one piece.
    batch_inference_size = 2000
    setup = 5
    # Options for *dataset*:
    # 'PermutedMNIST', 'SplitMNIST', 'CIFAR100_FeCAM_setup'
    dataset = "SplitMNIST"
    path_to_datasets = "./Data/"

    if dataset in ["PermutedMNIST", "SplitMNIST"]:
        part = 0
    elif dataset == "CIFAR100_FeCAM_setup":
        part = 6
        # ResNet, 5 tasks, 20 classes per each task
    else:
        raise ValueError("This dataset is currenly not implemented!")
    path_to_stored_networks = f"./Models/{dataset}/"
    path_to_save = f"./Results/{dataset}/"
    os.makedirs(path_to_save, exist_ok=True)

    results_summary = []
    numbers_of_models = [i for i in range(5)]
    seeds = [i + 1 for i in range(5)]

    for number_of_model, seed in zip(numbers_of_models, seeds):
        print(f"Calculations for model no: {number_of_model}")
        experiment_models = prepare_and_load_weights_for_models(
            path_to_stored_networks,
            path_to_datasets,
            number_of_model,
            dataset,
            seed=seed,
            part=part,
        )

        if dataset == "CIFAR100_FeCAM_setup":
            experiment_models["batch_inference_size"] = batch_inference_size
            experiment_models["setup"] = setup
            results = calculate_entropy_and_predict_classes_with_FeCAM(
                experiment_models
            )
        else:
            experiment_models["hyperparameters"]["saving_folder"] = path_to_save
            results = calculate_entropy_and_predict_classes_separately(
                experiment_models
            )
        results_summary.append(results)

    if dataset == "CIFAR100_FeCAM_setup":
        column_names = [
            "task_prediction_accuracy",
            "class_prediction_accuracy_from_logits",
            f"class_prediction_accuracy_from_{batch_inference_size}_batches",
            "mean_task_prediction_accuracy",
            "std_dev_task_prediction_accuracy",
            "mean_class_prediction_accuracy_from_logits",
            "std_dev_class_prediction_accuracy_from_logits",
            f"mean_class_prediction_accuracy_from_{batch_inference_size}_batches",
            f"std_dev_class_prediction_accuracy_from_{batch_inference_size}_batches",
        ]
        table_to_save = results_summary
    else:
        data_statistics = []
        for summary in results_summary:
            data_statistics.append(
                [
                    list(summary["task_prediction_acc"].values),
                    list(summary["class_prediction_acc"].values),
                    np.mean(summary["task_prediction_acc"].values),
                    np.std(summary["task_prediction_acc"].values),
                    np.mean(summary["class_prediction_acc"].values),
                    np.std(summary["class_prediction_acc"].values),
                ]
            )
        column_names = [
            "task_prediction_accuracy",
            "class_prediction_accuracy",
            "mean_task_prediction_accuracy",
            "std_dev_task_prediction_accuracy",
            "mean_class_prediction_accuracy",
            "std_dev_class_prediction_accuracy",
        ]
        table_to_save = data_statistics
    dataframe = pd.DataFrame(table_to_save, columns=column_names)
    dataframe.to_csv(
        f"{path_to_save}entropy_mean_results_batch_inference_"
        f"{batch_inference_size}.csv",
        sep=";",
    )

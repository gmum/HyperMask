import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from hypnettorch.mnets import MLP
from hypnettorch.mnets.resnet import ResNet
from hypnettorch.mnets.zenkenet import ZenkeNet
from hypnettorch.hnets import HMLP
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
import hypnettorch.utils.hnet_regularizer as hreg
from torch import nn
from datetime import datetime
from itertools import product
from torchpercentile import Percentile
from copy import deepcopy

from datasets import (
    set_hyperparameters,
    prepare_split_cifar100_tasks,
    prepare_permuted_mnist_tasks,
    prepare_split_mnist_tasks,
)


def set_seed(value):
    """
    Set deterministic results according to the given value
    (including random, numpy and torch libraries)
    """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_row_to_file(filename, elements):
    """
    Append a single row to the given file.

    Parameters
    ----------
    filename: folder and name of file
    elements: elements to saving in filename
    """
    if not filename.endswith(".csv"):
        filename += ".csv"
    filename = filename.replace(".pt", "")
    with open(filename, "a+") as stream:
        np.savetxt(stream, np.array(elements)[np.newaxis], delimiter=";", fmt="%s")


def write_pickle_file(filename, object_to_save):
    torch.save(object_to_save, f"{filename}.pt")


def load_pickle_file(filename):
    return torch.load(filename)


def get_shapes_of_network(model):
    """
    Get shape of all layers of the loaded model.

    Argument:
    ---------
      *model*: an instance of hypnettorch model, e.g. MLP from mnets

    Returns:
    --------
      A list with lists of shapes of consecutive network layers
    """
    shapes_of_model = []
    for layer in model.weights:
        shapes_of_model.append(list(layer.shape))
    return shapes_of_model


def calculate_current_threshold(current_iteration, max_sparsity, no_of_last_iteration):
    """
    Change the value of sparsity according to the current number
    of iteration, the number of iteration for which sparsity
    should achieve the maximum level, and the desired maximum level
    of sparsity.
    """
    assert current_iteration >= 0.0
    assert no_of_last_iteration >= 0.0
    assert 100.0 >= max_sparsity >= 0.0
    if current_iteration >= no_of_last_iteration:
        return max_sparsity
    else:
        coefficient = max_sparsity / no_of_last_iteration
        current_sparsity = coefficient * current_iteration
        return current_sparsity


def calculate_number_of_iterations(number_of_samples, batch_size, number_of_epochs):
    """
    Calculate the total number of iterations based on the number
    of samples, desired batch size and number of training epochs.

    Arguments:
    ----------
      *number_of_samples* (int) a number of individual samples
      *batch_size* (int) a number of samples entering the network
                   at one iteration
      *number_of_epochs* (int) a desired number of training epochs

    Returns:
    --------
      *no_of_iterations_per_epoch* (int) a number of training iterations
                                   per one epoch
      *total_no_of_iterations* (int) a total number of training iterations
    """
    no_of_iterations_per_epoch = int(np.ceil(number_of_samples / batch_size))
    total_no_of_iterations = int(no_of_iterations_per_epoch * number_of_epochs)
    return no_of_iterations_per_epoch, total_no_of_iterations


def get_number_of_batch_normalization_layer(target_network):
    """
    Get a number of batch normalization layer in a given target network.
    Each normalization layer consists of two vectors.

    Arguments:
    ----------
      *target_network* (hypnettorch.mnets instance) a target network for which
                       a mask will be created
    """
    if "batchnorm_layers" in dir(target_network):
        if target_network.batchnorm_layers is None:
            num_of_batch_norm_layers = 0
        else:
            # Each layer contains a vector of means and a vector of
            # standard deviations
            num_of_batch_norm_layers = 2 * len(target_network.batchnorm_layers)
    else:
        num_of_batch_norm_layers = 0
    return num_of_batch_norm_layers


def calculate_accuracy(data, target_network, weights, parameters, evaluation_dataset):
    """
    Calculate accuracy for a given dataset using a selected network
    and a selected set of weights

    Arguments:
    ----------
      *data*: an instance of the dataset (e.g.
              hypnettorch.data.special.permuted_mnist.PermutedMNIST)
              in the case of the PermutedMNIST dataset
      *target_network*: an instance of the network that will be used
                        during calculations (not necessarily with weights)
      *weights*: weights for the *target_network* network
                 (an instance of torch.nn.modules.container.ParameterList)
      *parameters* a dictionary containing the following keys:
        -device- string: 'cuda' or 'cpu', defines in which device calculations
                 will be performed
        -use_batch_norm_memory- Boolean: defines whether stored weights
                                of the batch normalization layer should be used
                                If True then *number_of_task* has to be given
        -number_of_task- int/None: gives an information which task is currently
                         solved. The number must be given when
                         -use_batch_norm_memory- is True
      *evaluation_dataset*: (string) 'validation' or 'test'; defines whether
                            a validation or a test set will be evaluated
    Returns:
    --------
       torch.Tensor defining an accuracy for the selected setting
    """
    assert (
        parameters["use_batch_norm_memory"] and parameters["number_of_task"] is not None
    ) or not parameters["use_batch_norm_memory"]
    assert evaluation_dataset in ["validation", "test"]
    target_network.eval()
    with torch.no_grad():
        # Currently results will be calculated on the validation or test set
        if evaluation_dataset == "validation":
            input_data = data.get_val_inputs()
            output_data = data.get_val_outputs()
        elif evaluation_dataset == "test":
            input_data = data.get_test_inputs()
            output_data = data.get_test_outputs()
        test_input = data.input_to_torch_tensor(
            input_data, parameters["device"], mode="inference"
        )
        test_output = data.output_to_torch_tensor(
            output_data, parameters["device"], mode="inference"
        )
        gt_classes = test_output.max(dim=1)[1]

        if parameters["use_batch_norm_memory"]:
            logits = target_network.forward(
                test_input, weights=weights, condition=parameters["number_of_task"]
            )
        else:
            logits = target_network.forward(test_input, weights=weights)
        predictions = logits.max(dim=1)[1]

        accuracy = (
            torch.sum(gt_classes == predictions, dtype=torch.float32)
            / gt_classes.numel()
        ) * 100.0
    return accuracy


def prepare_network_sparsity(weights, threshold, verbose=False):
    """
    Function for selection of top N% of weights in consecutive
    network layers

    Arguments:
    ----------
       *weights*: list of torch.Tensor objects containing weights
                  of the network
       *threshold*: float from the range (0, 100) for the selection
                    of most important weights
       *verbose*: optional Boolean defining whether additional information
                  should be written

    Returns:
    --------
       *masks*: list of torch.Tensor objects containing binary
                values indicating which weights were selected
                as the most important ones (1) and which ones
                are considered as the insignificant ones
    """
    masks = []
    for i in range(len(weights)):
        revalued_layer = torch.abs(torch.tanh(weights[i]))
        if i < (len(weights) - 1):
            percentile_value = Percentile()(revalued_layer.flatten(), [threshold])
            assert type(percentile_value.item()) == float
            zeros_weights = torch.zeros(
                revalued_layer.shape, device=revalued_layer.device
            )
            mask = torch.where(
                revalued_layer >= percentile_value, revalued_layer, zeros_weights
            )
            masks.append(mask)
        else:
            masks.append(revalued_layer)

        if verbose:
            ratio = torch.sum(masks[-1] > 0.0).item() / torch.numel(masks[-1])
            print(f"The ratio of non-zero elements: {ratio}")

    return masks


def unittest_prepare_network_sparsity():
    """
    Unittest of 'prepare_network_sparsity' function
    """
    test_list_of_tensors = [
        torch.Tensor([[0.1, 0.2, -0.1, -0.2, 0.03], [1, -1, 0.0, 0.45, -0.02]]),
        torch.Tensor([[0.1, 0.2, 0.3, -0.5, 0.6], [0.01, -0.01, 0.5, 0.08, 0.11]]),
    ]
    sparsity = 70
    test_output = prepare_network_sparsity(test_list_of_tensors, sparsity)
    gt_mask = [
        torch.Tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.7616, 0.7616, 0.0000, 0.4219, 0.0000],
            ]
        ),
        torch.Tensor(
            [
                [0.0997, 0.1974, 0.2913, 0.4621, 0.5370],
                [0.0100, 0.0100, 0.4621, 0.0798, 0.1096],
            ]
        ),
    ]
    for i in range(len(gt_mask)):
        torch.allclose(test_output[i], gt_mask[i])
    print("Unittest passed!")


def apply_mask_to_weights_of_network(target_network, masks):
    """
    Multiply each weight of the *target_network* by the value
    defined in the *masks* list

    Arguments:
    ----------
       *target_network*: an instance of the hypnettorch.mnets, i.e.
                         a network which will be a target
                         network in tandem with a hypernetwork
       *masks*: list of torch.Tensor objects containing values
                of masks for the target network

    Returns:
    --------
       A modified weights of the target network
    """
    num_of_batch_norm_layers = get_number_of_batch_normalization_layer(target_network)
    if "weights" in dir(target_network):
        target_network_weights = target_network.weights
    else:
        target_network_weights = target_network

    if type(target_network_weights) == list:
        total_no_of_layers = len(target_network_weights)
    else:
        total_no_of_layers = len([*target_network_weights.parameters()])

    if num_of_batch_norm_layers > 0:
        assert (len(masks) + num_of_batch_norm_layers) == total_no_of_layers
    else:
        assert len(masks) == total_no_of_layers

    masked_weights = []
    if num_of_batch_norm_layers > 0:
        # Append batch normalization layers, if any
        for i in range(num_of_batch_norm_layers):
            masked_weights.append(target_network_weights[i])
    # Batch normalization layers are located at the beginning
    # of the network's parameter list
    for no_of_layer in range(len(masks)):
        assert (
            masks[no_of_layer].shape
            == target_network_weights[no_of_layer + num_of_batch_norm_layers].shape
        )
        if no_of_layer == len(masks) - 1:
            masked_weights.append(
                target_network_weights[no_of_layer + num_of_batch_norm_layers]
            )
        else:
            masked_weights.append(
                target_network_weights[no_of_layer + num_of_batch_norm_layers]
                * masks[no_of_layer]
            )
    return masked_weights


def evaluate_previous_tasks(
    hypernetwork,
    target_network,
    dataframe_results,
    list_of_permutations,
    sparsity_parameter,
    parameters,
):
    """
    Evaluate the target network according to the weights generated
    by the hypernetwork for all previously trained tasks. For instance,
    if current_task_no is equal to 5, then tasks 0, 1, 2, 3, 4 and 5
    will be evaluated

    Arguments:
    ----------
      *hypernetwork* (hypnettorch.hnets module, e.g. mlp_hnet.MLP)
                     a hypernetwork that generates weights for the target
                     network
      *target_network* (hypnettorch.mnets module, e.g. mlp.MLP)
                       a target network that finally will perform
                       classification
      *dataframe_results* (Pandas Dataframe) stores results; contains
                          following columns: 'after_learning_of_task',
                          'tested_task' and 'accuracy'
      *list_of_permutations*: (hypnettorch.data module), e.g. in the case
                              of PermutedMNIST it will be
                              special.permuted_mnist.PermutedMNISTList
      *sparsity_parameter*: (float) defines which percentage of weights
                            of the target network should be left: it will
                            be (100-sparsity_parameter)%
      *parameters* a dictionary containing the following keys:
        -device- string: 'cuda' or 'cpu', defines in which device calculations
                 will be performed
        -use_batch_norm_memory- Boolean: defines whether stored weights
                                of the batch normalization layer should be used
                                If True then *number_of_task* has to be given
        -number_of_task- int/None: gives an information which task is currently
                         solved

    Returns:
    --------
      *dataframe_results* (Pandas Dataframe) a dataframe updated with
                          the calculated results
    """
    # Calculate accuracy for each previously trained task
    # as well as for the last trained task
    hypernetwork.eval()
    target_network.eval()
    for task in range(parameters["number_of_task"] + 1):
        # Target entropy calculation should be included here: hypernetwork has to be inferred
        # for each task (together with the target network) and the task_id with the lowest entropy
        # has to be chosen
        # Arguments of the function: list of permutations, hypernetwork, sparsity, target network
        # output: task id
        currently_tested_task = list_of_permutations[task]
        # Generate weights of the target network
        hypernetwork_weights = hypernetwork.forward(cond_id=task)
        masks = prepare_network_sparsity(hypernetwork_weights, sparsity_parameter)
        target_weights = apply_mask_to_weights_of_network(target_network, masks)
        accuracy = calculate_accuracy(
            currently_tested_task,
            target_network,
            target_weights,
            parameters=parameters,
            evaluation_dataset="test",
        )
        result = {
            "after_learning_of_task": parameters["number_of_task"],
            "tested_task": task,
            "accuracy": accuracy.cpu().item(),
        }
        print(f"Accuracy for task {task}: {accuracy}%.")
        dataframe_results = dataframe_results.append(result, ignore_index=True)
    return dataframe_results


def save_parameters(saving_folder, parameters, name=None):
    """
    Save hyperparameters to the selected file.

    Arguments:
    ----------
      *saving_folder* (string) defines a path to the folder for saving
      *parameters* (dictionary) contains all hyperparameters to saving
      *name* (optional string) name of the file for saving
    """
    if name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"parameters_{current_time}.csv"
    with open(f"{saving_folder}/{name}", "w") as file:
        for key in parameters.keys():
            file.write(f"{key};{parameters[key]}\n")


def plot_heatmap(load_path):
    """
    Plot heatmap presenting results for different learning tasks

    Argument:
    ---------
      *load_path* (string) contains path to the .csv file with
                  results in a dataframe shape, i.e. with columns:
                  'after_learning_of_task', 'tested_task' (both
                  integers) and 'accuracy' (float)
    """
    dataframe = pd.read_csv(load_path, delimiter=";", index_col=0)
    dataframe = dataframe.astype(
        {"after_learning_of_task": "int32", "tested_task": "int32"}
    )
    table = dataframe.pivot("after_learning_of_task", "tested_task", "accuracy")
    sns.heatmap(table, annot=True, fmt=".1f")
    plt.tight_layout()
    plt.savefig(load_path.replace(".csv", ".pdf"), dpi=300)
    plt.close()


def train_single_task(
    hypernetwork,
    target_network,
    criterion,
    parameters,
    dataset_list_of_tasks,
    current_no_of_task,
):
    """
    Train two neural networks: a hypernetwork will generate a sparse
    binary mask and the weights of the target neural network are multiplied by
    this binary mask creating a sparse network. This module operates
    on a single training task with a specific number.

    Arguments:
    ----------
      *hypernetwork*: (hypnettorch.hnets module, e.g. mlp_hnet.MLP)
                      a hypernetwork that generates weights for the target
                      network
      *target_network*: (hypnettorch.mnets module, e.g. mlp.MLP)
                        a target network that finally will perform
                        classification
      *criterion*: (torch.nn module) implements a loss function,
                   e.g. CrossEntropyLoss
      *parameters*: (dictionary) contains necessary hyperparameters
                    describing an experiment
      *dataset_list_of_tasks*: a module containing list of tasks for the CL
                               scenario, e.g. permuted_mnist.PermutedMNISTList
      *current_no_of_task*: (int) specifies the number of currently solving task

    Returns:
    --------
      *hypernetwork*: a modified module of hypernetwork
      *target_network*: a modified module of the target network
    """
    # Optimizer cannot be located outside of this function because after
    # deep copy of the network it needs to be reinitialized
    if parameters["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            [*hypernetwork.parameters(), *target_network.parameters()],
            lr=parameters["learning_rate"],
        )
    elif parameters["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            [*hypernetwork.parameters(), *target_network.parameters()],
            lr=parameters["learning_rate"],
        )
    else:
        raise ValueError("Wrong type of the selected optimizer!")
    if parameters["best_model_selection_method"] == "val_loss":
        # Store temporary best models to keep those with the highest
        # validation accuracy.
        best_hypernetwork = deepcopy(hypernetwork)
        best_target_network = deepcopy(target_network)
        best_val_accuracy = 0.0
    elif parameters["best_model_selection_method"] != "last_model":
        raise ValueError("Wrong value of best_model_selection_method parameter!")
    # Compute targets for the regularization part of loss before starting
    # the training of a current task
    hypernetwork.train()
    target_network.train()
    print(f"task: {current_no_of_task}")
    if current_no_of_task > 0:
        regularization_targets = hreg.get_current_targets(
            current_no_of_task, hypernetwork
        )
        previous_hnet_theta = None
        previous_hnet_embeddings = None
        previous_target_weights = deepcopy(target_network.weights)
    else:
        previous_target_weights = None

    if (parameters["target_network"] == "ResNet") and parameters["use_batch_norm"]:
        use_batch_norm_memory = True
    else:
        use_batch_norm_memory = False
    current_dataset_instance = dataset_list_of_tasks[current_no_of_task]
    # If training through a given number of epochs is desired
    # the number of iterations has to be calculated
    if parameters["number_of_epochs"] is not None:
        (
            no_of_iterations_per_epoch,
            parameters["number_of_iterations"],
        ) = calculate_number_of_iterations(
            current_dataset_instance.num_train_samples,
            parameters["batch_size"],
            parameters["number_of_epochs"],
        )
        # Scheduler can be set only when the number of epochs is given
        if parameters["lr_scheduler"]:
            current_epoch = 0
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "max",
                factor=np.sqrt(0.1),
                patience=5,
                min_lr=0.5e-6,
                cooldown=0,
                verbose=True,
            )
    for iteration in range(parameters["number_of_iterations"]):
        current_batch = current_dataset_instance.next_train_batch(
            parameters["batch_size"]
        )
        tensor_input = current_dataset_instance.input_to_torch_tensor(
            current_batch[0], parameters["device"], mode="train"
        )
        tensor_output = current_dataset_instance.output_to_torch_tensor(
            current_batch[1], parameters["device"], mode="train"
        )
        gt_output = tensor_output.max(dim=1)[1]
        optimizer.zero_grad()
        # Get weights of the hypernetwork and apply binary mask
        # to the target network
        hnet_weights = hypernetwork.forward(cond_id=current_no_of_task)

        current_sparsity_parameter = parameters["sparsity_parameter"]
        if current_no_of_task == 0 and parameters["adaptive_sparsity"]:
            current_sparsity_parameter = calculate_current_threshold(
                iteration + 1,
                parameters["sparsity_parameter"],
                parameters["number_of_iterations"],
            )
        masks = prepare_network_sparsity(hnet_weights, current_sparsity_parameter)
        loss_norm_target_regularizer = 0.0
        if current_no_of_task > 0:
            # Add another regularizer for weights, e.g. according
            # to L1 or L2 norm. The goal is to prevent significant
            # changes in weights between consecutive tasks.
            # ATTENTION! This norm is not calculated for batch
            # normalization layers
            # This norm is applied BEFORE the multiplication by
            # mask from the hypernetwork
            no_of_batch_norm_layers = get_number_of_batch_normalization_layer(
                target_network
            )
            for no_of_layer in range(len(masks)):
                if parameters["norm_regularizer_masking"]:
                    loss_norm_target_regularizer += torch.norm(
                        (
                            target_network.weights[
                                no_of_layer + no_of_batch_norm_layers
                            ]
                            - previous_target_weights[
                                no_of_layer + no_of_batch_norm_layers
                            ]
                        )
                        * masks[no_of_layer],
                        p=parameters["norm"],
                    )
                else:
                    loss_norm_target_regularizer += torch.norm(
                        target_network.weights[no_of_layer + no_of_batch_norm_layers]
                        - previous_target_weights[
                            no_of_layer + no_of_batch_norm_layers
                        ],
                        p=parameters["norm"],
                    )
        target_weights = apply_mask_to_weights_of_network(target_network, masks)
        # Even if batch normalization layers are applied, statistics
        # for the last saved tasks will be applied so there is no need to
        # give 'current_no_of_task' as a value for the 'condition' argument.
        prediction = target_network.forward(tensor_input, weights=target_weights)
        loss_current_task = criterion(prediction, gt_output)
        loss_regularization = 0.0
        if current_no_of_task > 0:
            loss_regularization = hreg.calc_fix_target_reg(
                hypernetwork,
                current_no_of_task,
                targets=regularization_targets,
                mnet=target_network,
                prev_theta=previous_hnet_theta,
                prev_task_embs=previous_hnet_embeddings,
                inds_of_out_heads=None,
                batch_size=-1,
            )
        append_row_to_file(
            f'{parameters["saving_folder"]}regularization_loss.csv',
            f"{current_no_of_task};{iteration};"
            f"{loss_regularization};{loss_norm_target_regularizer}",
        )
        loss = (
            loss_current_task
            + parameters["beta"] * loss_regularization / max(1, current_no_of_task)
            + parameters["lambda"] * loss_norm_target_regularizer
        )
        loss.backward()
        optimizer.step()

        if parameters["number_of_epochs"] is None:
            condition = (iteration % 100 == 0) or (
                iteration == (parameters["number_of_iterations"] - 1)
            )
        else:
            condition = (
                (iteration % 100 == 0)
                or (iteration == (parameters["number_of_iterations"] - 1))
                or (((iteration + 1) % no_of_iterations_per_epoch) == 0)
            )

        if condition:
            if parameters["number_of_epochs"] is not None:
                current_epoch = (iteration + 1) // no_of_iterations_per_epoch
                print(f"Current epoch: {current_epoch}")
            accuracy = calculate_accuracy(
                current_dataset_instance,
                target_network,
                target_weights,
                parameters={
                    "device": parameters["device"],
                    "use_batch_norm_memory": use_batch_norm_memory,
                    "number_of_task": current_no_of_task,
                },
                evaluation_dataset="validation",
            )
            print(
                f"Task {current_no_of_task}, iteration: {iteration + 1},"
                f" loss: {loss.item()}, validation accuracy: {accuracy}"
            )
            # If the accuracy on the validation dataset is higher
            # than previously
            if parameters["best_model_selection_method"] == "val_loss":
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    best_hypernetwork = deepcopy(hypernetwork)
                    best_target_network = deepcopy(target_network)
            if parameters["save_masks"]:
                filename = f"mask_task_{current_no_of_task}_" f"iteration_{iteration}_"
                write_pickle_file(f'{parameters["saving_folder"]}/{filename}', masks)
            if (
                parameters["number_of_epochs"] is not None
                and parameters["lr_scheduler"]
                and (((iteration + 1) % no_of_iterations_per_epoch) == 0)
            ):
                print("Finishing the current epoch")
                plateau_scheduler.step(accuracy)

    if parameters["best_model_selection_method"] == "val_loss":
        return best_hypernetwork, best_target_network
    else:
        return hypernetwork, target_network


def build_multiple_task_experiment(dataset_list_of_tasks, parameters, use_chunks=False):
    """
    Create a continual learning experiment with multiple tasks
    for a given dataset.

    Arguments:
    ----------
      *dataset_list_of_tasks*: a module containing list of tasks for the CL
                               scenario, e.g. permuted_mnist.PermutedMNISTList
      *parameters*: (dictionary) contains necessary hyperparameters
                    describing an experiment
      *use_chunks*: (Boolean value) optional argument, defines whether
                    a hypernetwork should generate weights in chunks or not

    Returns:
    --------
      *hypernetwork*: (hypnettorch.hnets module, e.g. mlp_hnet.MLP)
                      a hypernetwork that generates weights for the target
                      network
      *target_network*: (hypnettorch.mnets module, e.g. mlp.MLP)
                        a target network that finally will perform
                        classification
      *dataframe*: (Pandas Dataframe) contains results from consecutive
                   evaluations for all previous tasks
    """
    output_shape = list(dataset_list_of_tasks[0].get_train_outputs())[0].shape[0]
    # Create a target network which will be multilayer perceptron
    # or ResNet/ZenkeNet with internal weights
    if parameters["target_network"] == "MLP":
        target_network = MLP(
            n_in=parameters["input_shape"],
            n_out=output_shape,
            hidden_layers=parameters["target_hidden_layers"],
            use_bias=parameters["use_bias"],
            no_weights=False,
        ).to(parameters["device"])
    elif parameters["target_network"] == "ResNet":
        target_network = ResNet(
            in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
            use_bias=parameters["use_bias"],
            num_classes=output_shape,
            n=parameters["resnet_number_of_layer_groups"],
            k=parameters["resnet_widening_factor"],
            no_weights=False,
            use_batch_norm=parameters["use_batch_norm"],
            bn_track_stats=False,
        ).to(parameters["device"])
    elif parameters["target_network"] == "ZenkeNet":
        target_network = ZenkeNet(
            in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
            num_classes=output_shape,
            arch="cifar",
            no_weights=False,
        ).to(parameters["device"])
    # Create a hypernetwork based on the shape of the target network
    no_of_batch_norm_layers = get_number_of_batch_normalization_layer(target_network)
    if not use_chunks:
        hypernetwork = HMLP(
            target_network.param_shapes[no_of_batch_norm_layers:],
            uncond_in_size=0,
            cond_in_size=parameters["embedding_size"],
            activation_fn=parameters["activation_function"],
            layers=parameters["hypernetwork_hidden_layers"],
            num_cond_embs=parameters["number_of_tasks"],
        ).to(parameters["device"])
    else:
        hypernetwork = ChunkedHMLP(
            target_shapes=target_network.param_shapes[no_of_batch_norm_layers:],
            chunk_size=parameters["chunk_size"],
            chunk_emb_size=parameters["chunk_emb_size"],
            cond_in_size=parameters["embedding_size"],
            activation_fn=parameters["activation_function"],
            layers=parameters["hypernetwork_hidden_layers"],
            num_cond_embs=parameters["number_of_tasks"],
        ).to(parameters["device"])

    criterion = nn.CrossEntropyLoss()
    dataframe = pd.DataFrame(
        columns=["after_learning_of_task", "tested_task", "accuracy"]
    )

    if (parameters["target_network"] == "ResNet") and parameters["use_batch_norm"]:
        use_batch_norm_memory = True
    else:
        use_batch_norm_memory = False
    hypernetwork.train()
    target_network.train()
    for no_of_task in range(parameters["number_of_tasks"]):
        hypernetwork, target_network = train_single_task(
            hypernetwork,
            target_network,
            criterion,
            parameters,
            dataset_list_of_tasks,
            no_of_task,
        )

        if no_of_task == (parameters["number_of_tasks"] - 1):
            # Save current state of networks
            write_pickle_file(
                f'{parameters["saving_folder"]}/'
                f"hypernetwork_after_{no_of_task}_task",
                hypernetwork.weights,
            )
            write_pickle_file(
                f'{parameters["saving_folder"]}/'
                f"target_network_after_{no_of_task}_task",
                target_network.weights,
            )
        dataframe = evaluate_previous_tasks(
            hypernetwork,
            target_network,
            dataframe,
            dataset_list_of_tasks,
            parameters["sparsity_parameter"],
            parameters={
                "device": parameters["device"],
                "use_batch_norm_memory": use_batch_norm_memory,
                "number_of_task": no_of_task,
            },
        )
        dataframe = dataframe.astype(
            {"after_learning_of_task": "int", "tested_task": "int"}
        )
        dataframe.to_csv(
            f'{parameters["saving_folder"]}/'
            f'results_{parameters["name_suffix"]}.csv',
            sep=";",
        )
    return hypernetwork, target_network, dataframe


def main_running_experiments(path_to_datasets, parameters, dataset):
    """
    Perform a series of experiments based on the hyperparameters.

    Arguments:
    ----------
      *path_to_datasets*: (str) path to files with datasets
      *parameters*: (dict) contains multiple experiment hyperparameters
      *dataset*: (str) dataset for calculation: PermutedMNIST,
                 CIFAR100 or SplitMNIST

    Returns learned hypernetwork, target network and a dataframe
    with single results.
    """
    if dataset == "PermutedMNIST":
        dataset_tasks_list = prepare_permuted_mnist_tasks(
            path_to_datasets,
            parameters["input_shape"],
            parameters["number_of_tasks"],
            parameters["padding"],
            parameters["no_of_validation_samples"],
        )
    elif dataset == "CIFAR100":
        dataset_tasks_list = prepare_split_cifar100_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
        )
    elif dataset == "SplitMNIST":
        dataset_tasks_list = prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=parameters["no_of_validation_samples"],
            use_augmentation=parameters["augmentation"],
            number_of_tasks=parameters["number_of_tasks"],
        )
    else:
        raise ValueError("Wrong name of the dataset!")

    hypernetwork, target_network, dataframe = build_multiple_task_experiment(
        dataset_tasks_list, parameters, use_chunks=parameters["use_chunks"]
    )
    # Calculate statistics of grid search results
    no_of_last_task = parameters["number_of_tasks"] - 1
    accuracies = dataframe.loc[dataframe["after_learning_of_task"] == no_of_last_task][
        "accuracy"
    ].values
    row_with_results = (
        f"{dataset_tasks_list[0].get_identifier()};"
        f'{parameters["augmentation"]};'
        f'{parameters["embedding_size"]};'
        f'{parameters["seed"]};'
        f'{str(parameters["hypernetwork_hidden_layers"]).replace(" ", "")};'
        f'{parameters["use_chunks"]};{parameters["chunk_emb_size"]};'
        f'{parameters["target_network"]};'
        f'{str(parameters["target_hidden_layers"]).replace(" ", "")};'
        f'{parameters["resnet_number_of_layer_groups"]};'
        f'{parameters["resnet_widening_factor"]};'
        f'{parameters["norm_regularizer_masking"]};'
        f'{parameters["best_model_selection_method"]};'
        f'{parameters["optimizer"]};'
        f'{parameters["activation_function"]};'
        f'{parameters["learning_rate"]};{parameters["batch_size"]};'
        f'{parameters["beta"]};{parameters["sparsity_parameter"]};'
        f'{parameters["norm"]};{parameters["lambda"]};'
        f"{np.mean(accuracies)};{np.std(accuracies)}"
    )
    append_row_to_file(
        f'{parameters["grid_search_folder"]}'
        f'{parameters["summary_results_filename"]}.csv',
        row_with_results,
    )

    load_path = (
        f'{parameters["saving_folder"]}/' f'results_{parameters["name_suffix"]}.csv'
    )
    plot_heatmap(load_path)
    return hypernetwork, target_network, dataframe


if __name__ == "__main__":
    unittest_prepare_network_sparsity()
    path_to_datasets = "./Data"
    dataset = "CIFAR100"  # 'PermutedMNIST', 'CIFAR100', 'SplitMNIST'
    part = 0
    create_grid_search = False
    if create_grid_search:
        summary_results_filename = "grid_search_results"
    else:
        summary_results_filename = "summary_results"
    hyperparameters = set_hyperparameters(
        dataset, grid_search=create_grid_search, part=part
    )
    header = (
        "dataset_name;augmentation;embedding_size;seed;hypernetwork_hidden_layers;"
        "use_chunks;chunk_emb_size;target_network;target_hidden_layers;"
        "layer_groups;widening;norm_regularizer_masking;final_model;optimizer;"
        "hypernet_activation_function;learning_rate;batch_size;beta;"
        "sparsity;norm;lambda;mean_accuracy;std_accuracy"
    )
    append_row_to_file(
        f'{hyperparameters["saving_folder"]}{summary_results_filename}.csv', header
    )

    for no, elements in enumerate(
        product(
            hyperparameters["embedding_sizes"],
            hyperparameters["learning_rates"],
            hyperparameters["betas"],
            hyperparameters["hypernetworks_hidden_layers"],
            hyperparameters["sparsity_parameters"],
            hyperparameters["lambdas"],
            hyperparameters["batch_sizes"],
            hyperparameters["norm_regularizer_masking_opts"],
            hyperparameters["seed"],
        )
    ):
        embedding_size = elements[0]
        learning_rate = elements[1]
        beta = elements[2]
        hypernetwork_hidden_layers = elements[3]
        sparsity_parameter = elements[4]
        lambda_par = elements[5]
        batch_size = elements[6]
        norm_regularizer_masking = elements[7]
        # Of course, seed is not optimized but it is easier to prepare experiments
        # for multiple seeds in such a way
        seed = elements[8]

        parameters = {
            "input_shape": hyperparameters["shape"],
            "augmentation": hyperparameters["augmentation"],
            "number_of_tasks": hyperparameters["number_of_tasks"],
            "seed": seed,
            "hypernetwork_hidden_layers": hypernetwork_hidden_layers,
            "activation_function": hyperparameters["activation_function"],
            "norm_regularizer_masking": norm_regularizer_masking,
            "use_chunks": hyperparameters["use_chunks"],
            "chunk_size": hyperparameters["chunk_size"],
            "chunk_emb_size": hyperparameters["chunk_emb_size"],
            "target_network": hyperparameters["target_network"],
            "target_hidden_layers": hyperparameters["target_hidden_layers"],
            "resnet_number_of_layer_groups": hyperparameters[
                "resnet_number_of_layer_groups"
            ],
            "resnet_widening_factor": hyperparameters["resnet_widening_factor"],
            "adaptive_sparsity": hyperparameters["adaptive_sparsity"],
            "sparsity_parameter": sparsity_parameter,
            "learning_rate": learning_rate,
            "best_model_selection_method": hyperparameters[
                "best_model_selection_method"
            ],
            "lr_scheduler": hyperparameters["lr_scheduler"],
            "batch_size": batch_size,
            "number_of_epochs": hyperparameters["number_of_epochs"],
            "number_of_iterations": hyperparameters["number_of_iterations"],
            "no_of_validation_samples": hyperparameters["no_of_validation_samples"],
            "embedding_size": embedding_size,
            "norm": hyperparameters["norm"],
            "lambda": lambda_par,
            "optimizer": hyperparameters["optimizer"],
            "beta": beta,
            "padding": hyperparameters["padding"],
            "use_bias": hyperparameters["use_bias"],
            "use_batch_norm": hyperparameters["use_batch_norm"],
            "device": hyperparameters["device"],
            "saving_folder": f'{hyperparameters["saving_folder"]}{no}/',
            "grid_search_folder": hyperparameters["saving_folder"],
            "save_masks": hyperparameters["save_consecutive_masks"],
            "name_suffix": f"mask_sparsity_{sparsity_parameter}",
            "summary_results_filename": summary_results_filename,
        }

        os.makedirs(parameters["saving_folder"], exist_ok=True)
        # start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_parameters(
            parameters["saving_folder"],
            parameters,
            name=f'parameters_{parameters["name_suffix"]}.csv',
        )

        # Important! Seed is set before the preparation of the dataset!
        if seed is not None:
            set_seed(seed)

        hypernetwork, target_network, dataframe = main_running_experiments(
            path_to_datasets, parameters, dataset
        )

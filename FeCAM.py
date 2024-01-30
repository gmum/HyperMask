import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import deepcopy

from main import (
    apply_mask_to_weights_of_network,
    prepare_network_sparsity,
)
from evaluation import (
    evaluate_target_network,
    prepare_and_load_weights_for_models,
)
from numpy.testing import assert_array_equal, assert_array_almost_equal


##########################################################################
# Functions from FeCAM, reformatted and with ensuring convergence of SVD
# in _mahalanobis():
# https://github.com/dipamgoswami/FeCAM/blob/main/FeCAM_vit_cifar100.py


def shrink_cov(cov):
    diag_mean = np.mean(np.diagonal(cov))
    off_diag = np.copy(cov)
    np.fill_diagonal(off_diag, 0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag * mask).sum() / mask.sum()
    iden = np.eye(cov.shape[0])
    alpha1 = 1
    alpha2 = 1
    cov_ = (
        cov
        + (alpha1 * diag_mean * iden)
        + (alpha2 * off_diag_mean * (1 - iden))
    )
    return cov_


def normalize_cov(cov):
    sd = np.sqrt(np.diagonal(cov))  # standard deviations of the variables
    cov = cov / (np.matmul(np.expand_dims(sd, 1), np.expand_dims(sd, 0)))
    return cov


def _mahalanobis(dist, cov):
    # To ensure that SVD will converge
    cov = cov + 1e-8
    inv_covmat = np.linalg.pinv(cov)
    # pseudo-inverse of an invertible matrix is same as its inverse
    left_term = np.matmul(dist, inv_covmat)
    mahal = np.matmul(left_term, dist.T)
    return np.diagonal(mahal, 0)


##########################################################################


def translate_output_CIFAR_classes(labels, setup, task):
    """
    Translate labels of form {0, 1, ..., N-1} to the real labels
    of CIFAR100 dataset.

    Arguments:
    ----------
       *labels*: (Numpy array | list) contains labels of the form {0, 1, ..., N-1}
                 where N is the the number of classes in a single task
       *setup*: (int) defines how many tasks were created in this
                training session
       *task*: (int) number of the currently calculated task
    Returns:
    --------
       A numpy array of the same shape like *labels* but with proper
       class labels
    """
    assert setup in [5, 6, 11, 21]
    # 5 tasks: 20 classes in each task
    # 6 tasks: 50 initial classes + 5 incremental tasks per 10 classes
    # 11 tasks: 50 initial classes + 10 incremental tasks per 5 classes
    # 21 tasks: 40 initial classes + 20 incremental tasks per 3 classes
    class_orders = [
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
        94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
        84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
        69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
        17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
        1, 28, 6, 46, 62, 82, 53, 9, 31, 75,
        38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
        40, 30, 23, 85, 2, 95, 56, 48, 71, 64,
        98, 13, 99, 7, 34, 55, 54, 26, 35, 39
    ]
    if setup in [6, 11]:
        no_of_initial_cls = 50
    elif setup == 21:
        no_of_initial_cls = 40
    else:
        no_of_initial_cls = 20
    if task == 0:
        currently_used_classes = class_orders[:no_of_initial_cls]
    else:
        if setup == 6:
            no_of_incremental_cls = 10
        elif setup == 11:
            no_of_incremental_cls = 5
        elif setup == 21:
            no_of_incremental_cls = 3
        else:
            no_of_incremental_cls = 20
        currently_used_classes = class_orders[
            (no_of_initial_cls + no_of_incremental_cls * (task - 1)) : (
                no_of_initial_cls + no_of_incremental_cls * task
            )
        ]
    y_translated = np.array([currently_used_classes[i] for i in labels])
    return y_translated


def unittest_translate_output_CIFAR_classes():
    """
    Unittest of translate_output_CIFAR_classes() function.
    """
    # 21 tasks
    labels = [i for i in range(40)]
    test_1 = translate_output_CIFAR_classes(labels, 21, 0)
    gt_1 = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
            94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
            84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
            69, 66, 77, 47, 79, 93, 29, 50, 57, 83]
    assert (test_1 == gt_1).all()
    labels = [i for i in range(3)]
    test_2 = translate_output_CIFAR_classes(labels, 21, 1)
    gt_2 = [17, 81, 41]
    assert (test_2 == gt_2).all()
    test_3 = translate_output_CIFAR_classes(labels, 21, 20)
    gt_3 = [26, 35, 39]
    assert (test_3 == gt_3).all()
    # 11 tasks
    labels = [i for i in range(50)]
    test_4 = translate_output_CIFAR_classes(labels, 11, 0)
    gt_4 = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
            94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
            84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
            69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
            17, 81, 41, 12, 37, 59, 25, 20, 80, 73]
    assert (test_4 == gt_4).all()
    labels = [i for i in range(5)]
    test_5 = translate_output_CIFAR_classes(labels, 11, 2)
    gt_5 = [82, 53, 9, 31, 75]
    assert (test_5 == gt_5).all()
    # 6 tasks
    labels = [i for i in range(50)]
    test_6 = translate_output_CIFAR_classes(labels, 6, 0)
    assert (test_6 == gt_4).all()
    labels = [i for i in range(10)]
    test_7 = translate_output_CIFAR_classes(labels, 6, 4)
    gt_7 = [40, 30, 23, 85, 2, 95, 56, 48, 71, 64]
    assert (test_7 == gt_7).all()
    # 5 tasks
    labels = [i for i in range(20)]
    test_8 = translate_output_CIFAR_classes(labels, 5, 0)
    gt_8 = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
            94, 92, 10, 72, 49, 78, 61, 14, 8, 86]
    assert (test_8 == gt_8).all()
    test_9 = translate_output_CIFAR_classes(labels, 5, 3)
    gt_9 = [38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
            60, 19, 70, 90, 89, 43, 5, 42, 65, 76]
    assert (test_9 == gt_9).all()


def translate_output_MNIST_classes(relative_labels, task, mode):
    """
    Translate relative labels of form {0, 1} to the real labels
    of Split MNIST dataset.

    Arguments:
    ----------
       *labels*: (Numpy array | list) contains labels of the form
       *task*: (int) number of the currently calculated task,
               starting from 0
       *mode*: (str) "permuted" or "split", depending on the desired
               dataset
    """
    assert mode in ["permuted", "split"]
    if mode == "permuted":
        total_no_of_classes = 100
        no_of_classes_per_task = 10
        # Even if the classifier indicates '0' but from the wrong task
        # it has to get a penalty. Therefore, in Permuted MNIST there
        # are 100 unique classes.
    elif mode == "split":
        total_no_of_classes = 10
        no_of_classes_per_task = 2
    class_orders = [i for i in range(total_no_of_classes)]
    currently_used_classes = class_orders[
        (no_of_classes_per_task * task) : (no_of_classes_per_task * (task + 1))
    ]
    y_translated = np.array(
        [currently_used_classes[i] for i in relative_labels]
    )
    return y_translated


def unittest_translate_output_MNIST_classes():
    """
    Unittest of translate_output_MNIST_classes() function.
    """
    labels = [0, 0, 1, 0]
    test_1 = translate_output_MNIST_classes(labels, 3, "split")
    gt_1 = np.array([6, 6, 7, 6])
    assert_array_equal(test_1, gt_1)

    labels = [0, 1, 1, 0]
    test_2 = translate_output_MNIST_classes(labels, 0, "split")
    gt_2 = np.array([0, 1, 1, 0])
    assert_array_equal(test_2, gt_2)

    labels = [0, 5, 7, 0, 8, 9]
    test_3 = translate_output_MNIST_classes(labels, 0, "permuted")
    gt_3 = np.array([0, 5, 7, 0, 8, 9])
    assert_array_equal(test_3, gt_3)

    test_4 = translate_output_MNIST_classes(labels, 5, "permuted")
    gt_4 = np.array([50, 55, 57, 50, 58, 59])
    assert_array_equal(test_4, gt_4)


def get_target_network_representation(
    hypernetwork,
    hypernetwork_weights,
    target_network,
    target_weights,
    target_network_type,
    input_data,
    sparsity,
    no_of_batch_norm_layers,
    task,
):
    """
    Calculate the output classification layer of the target network,
    having a hypernetwork with its weights, and a target network with
    its weights, as well as the number of the considered task.
    Currently, only ResNetF is handled.

    Arguments:
    ----------
       *hypernetwork*: an instance of HMLP class
       *hypernetwork_weights*: loaded weights for the hypernetwork
       *target_network*: an instance of MLP or ResNet class
       *target_weights*: loaded weights for the target network
       *target_network_type*: str representing the target network architecture
       *input_data*: torch.Tensor with input data for the network
       *sparsity*: int representing the percentage of weights for remove
       *no_of_batch_norm_layers*: the number of batch normalization layers
                                  in the target network
       *task*: int representing the considered task; the corresponding
               embedding and batch normalization statistics will be used

    Returns:
    --------
       A list containing torch.Tensor (or tensors) representing values
       from the output classification layer
    """
    hypernetwork.eval()
    target_network.eval()
    with torch.no_grad():
        hypernetwork_output = hypernetwork.forward(
            cond_id=task, weights=hypernetwork_weights
        )
        masks = prepare_network_sparsity(hypernetwork_output, sparsity)
        target_masked_weights = apply_mask_to_weights_of_network(
            target_weights,
            masks,
            num_of_batch_norm_layers=no_of_batch_norm_layers,
        )
        if target_network_type in ["ResNetF", "MLP_FeCAM"]:
            logits_masked, features = evaluate_target_network(
                target_network,
                input_data,
                target_masked_weights,
                target_network_type,
                condition=task,
            )
            logits_masked = logits_masked.detach().cpu()
            features = features.detach().cpu()
        else:
            logits_masked = evaluate_target_network(
                target_network,
                input_data,
                target_masked_weights,
                target_network_type,
                condition=task,
            )
            logits_masked = logits_masked.detach().cpu()
    if target_network_type in ["ResNetF", "MLP_FeCAM"]:
        return [logits_masked, features]
    else:
        return [logits_masked]


def extract_training_set_from_all_tasks(
    hypernetwork,
    hypernetwork_weights,
    target_network,
    target_weights,
    target_network_type,
    dataset_CL_tasks,
    dataset_name,
    number_of_incremental_tasks,
    total_number_of_tasks,
    sparsity_parameter,
    batch_train_set_size,
    no_of_batch_norm_layers,
    device,
    tukeyed=False,
):
    """
    To prepare covariance matrices and class prototypes it is necessary
    to extract full training sets from consecutive tasks considered
    in a given scenario.
    This function extracts training data samples and corresponding labels
    in terms of real number of classes, not the relative ones (being the
    classification layer output), as well as the task IDs.

    Arguments:
    ----------
        *hypernetwork* (hypnettorch.hnets.mlp_hnet.HMLP) a hypernetwork object
        *hypernetwork_weights* (torch.nn.modules.containter.ParameterList)
                               contains weights for *hypernetwork*
        *target_network* a target network object; currently only ResNetF is
                         supported
        *target_weights* (torch.nn.modules.container.ParameterList) contains
                         weights of the target network
        *target_network_type* (str) defines name of the target architecture
        *dataset_CL_tasks* (list of datasets) list of objects storing training
                           and test samples from consecutive tasks
        *dataset_name*: (str) name of the dataset for proper class translation
        *number_of_incremental_tasks* (int) the number of consecutive tasks
                                      from which the training sets will be
                                      extracted
        *total_number_of_tasks* (int) the number of all tasks in a given
                                experiment
        *sparsity_parameter* (int) defines a percentage of weights for removing
                             in consecutive layers
        *batch_train_set_size* (int) defines size of single batch for
                               the inference of training data
        *no_of_batch_norm_layers* (int) defines the number of batch
                                  normalization layer in the target network
        *device*: (str) 'cpu' or 'cuda', defines the equipment for computations
        *tukeyed*: (Boolean value) optional, defines whether Tukey
                   transformation should be applied after forward propagation
                   through network, default: False

    Returns:
    --------
        *X*: (Numpy array) contains all samples,
             shapes: (number of samples, size of the features)
        *y*: (Numpy array) contains labels for samples from *X*
             shapes: (number of samples,)
        *tasks_gt*: (Numpy array) contains task ID for samples from *X*
                    shapes: (number of samples,)
        *dict_classes_per_task* (dictionary) defines labels for consecutive
                                tasks
    """
    X, y, tasks_gt = [], [], []
    dict_classes_per_task = {}
    for task in range(number_of_incremental_tasks):
        target_loaded_weights = deepcopy(target_weights)
        currently_evaluated_task = dataset_CL_tasks[task]
        #### Potentially FUNCTIOn 1a):
        # Create a dictionary which labels are present in consecutive tasks
        # Get real classes used in the current task
        whole_output = currently_evaluated_task.output_to_torch_tensor(
            currently_evaluated_task.get_train_outputs(),
            device,
            mode="inference",
        )
        whole_output = whole_output.max(dim=1)[1].cpu().detach().numpy()
        if dataset_name == "CIFAR100_FeCAM_setup":
            whole_output = translate_output_CIFAR_classes(
                whole_output, total_number_of_tasks, task
            )
        elif dataset_name in ["PermutedMNIST", "SplitMNIST"]:
            mode = "permuted" if dataset == "PermutedMNIST" else "split"
            whole_output = translate_output_MNIST_classes(
                whole_output, task, mode=mode
            )
        dict_classes_per_task[task] = list(np.unique(whole_output))
        ####
        for (
            batch_size,
            samples,
            labels,
        ) in currently_evaluated_task.train_iterator(batch_train_set_size):
            train_input = currently_evaluated_task.input_to_torch_tensor(
                samples, device, mode="inference"
            )
            train_output = currently_evaluated_task.output_to_torch_tensor(
                labels, device, mode="inference"
            )
            gt_classes = train_output.max(dim=1)[1].cpu().detach().numpy()
            # Translate relative outputs (labels) to the general labels
            if dataset_name == "CIFAR100_FeCAM_setup":
                gt_classes = translate_output_CIFAR_classes(
                    gt_classes, total_number_of_tasks, task
                )
            elif dataset in ["PermutedMNIST", "SplitMNIST"]:
                gt_classes = translate_output_MNIST_classes(
                    gt_classes, task, mode=mode
                )
            y.append(gt_classes)
            current_task_gt = np.zeros_like(gt_classes) + task
            tasks_gt.append(current_task_gt)

            # We know which class we are calculating during the training
            # process, therefore we can select proper task
            features = get_target_network_representation(
                hypernetwork,
                hypernetwork_weights,
                target_network,
                target_loaded_weights,
                target_network_type,
                train_input,
                sparsity_parameter,
                no_of_batch_norm_layers,
                task,
            )
            features = features[0] if len(features) == 1 else features[1]
            if tukeyed:
                features = np.power(features, 2)
            X.append(features)
    X, y, tasks_gt = (
        np.concatenate(X),
        np.concatenate(y),
        np.concatenate(tasks_gt),
    )
    return X, y, tasks_gt, dict_classes_per_task


def extract_test_set_from_single_task(
    dataset_CL_tasks, no_of_task, dataset, device
):
    """
    Extract test samples dedicated for a selected task
    and change relative output classes into absolute classes.

    Arguments:
    ----------
       *dataset_CL_tasks*: list of objects containing consecutive tasks
       *no_of_task*: (int) represents number of the currently analyzed task
       *dataset*: (str) defines name of the dataset used: 'PermutedMNIST',
                  'SplitMNIST' or 'CIFAR100_FeCAM_setup'
       *device*: (str) defines whether CPU or GPU will be used

    Returns:
    --------
       *X_test*: (torch.Tensor) represents input samples
       *gt_classes*: (Numpy array) represents absolute classes for *X_test*
       *gt_tasks*: (list) represents number of task for corresponding samples
    """
    tested_task = dataset_CL_tasks[no_of_task]
    input_data = tested_task.get_test_inputs()
    output_data = tested_task.get_test_outputs()
    X_test = tested_task.input_to_torch_tensor(
        input_data, device, mode="inference"
    )
    test_output = tested_task.output_to_torch_tensor(
        output_data, device, mode="inference"
    )
    gt_classes = test_output.max(dim=1)[1]
    if dataset == "CIFAR100_FeCAM_setup":
        # Currently there is an assumption that only setup with
        # 5 tasks will be used
        gt_classes = translate_output_CIFAR_classes(
            gt_classes, setup=5, task=no_of_task
        )
    elif dataset in ["PermutedMNIST", "SplitMNIST"]:
        mode = "permuted" if dataset == "PermutedMNIST" else "split"
        gt_classes = translate_output_MNIST_classes(
            gt_classes, task=no_of_task, mode=mode
        )
    else:
        raise ValueError("Wrong name of the dataset!")
    gt_tasks = [no_of_task for _ in range(output_data.shape[0])]
    return X_test, gt_classes, gt_tasks


def create_covariance_matrices_and_prototypes(
    X,
    y,
    unique_objects,
    no_shrinkage=2,
    normalize_covariance=True,
    tukeyed_samples=False,
    tukeyed_prototypes=False,
):
    """
    For each class create its covariance matrix and prototype.

    Arguments:
    ----------
       *X*: (Numpy array) contains all samples,
            shapes: (number of samples, size of the features)
       *y*: (Numpy array) contains labels for samples from *X*
            shapes: (number of samples,)
       *unique_objects*: (Numpy array or list) contains labels of all
                         classes/tasks occurring in this experiment
       *no_shrinkage*: (int) defines how many times the shrinkage of covariance
                       matrix should be applied, default: 2
       *normalize_covariance*: (Boolean value) optional, defines whether
                               a covariance matrix normalization should be
                               applied, default: True
       *tukeyed_samples*: (Boolean value) optional, defines whether Tukey
                          transformation should be applied for samples
                          before covariance matrices creation, default: False
       *tukeyed_prototypes*: (Boolean value) optional, defines whether Tukey
                             transformation should be applied for class
                             prototypes, default: False

    Returns:
    --------
       *covariance_matrices* (dict) contains covariance matrices for
                             all classes
       *prototypes* (dict) contains class prototypes
    """
    X = deepcopy(X)
    covariance_matrices, prototypes = {}, {}
    # For the verification whether each index was selected exactly once
    sanity_check_classes = np.zeros_like(y)
    for class_no in unique_objects:
        image_class_mask = y == class_no
        selected_indices = np.nonzero(image_class_mask)
        prototypes[class_no] = np.mean(X[selected_indices], axis=0)
        if tukeyed_prototypes:
            prototypes[class_no] = np.power(prototypes[class_no], 2)
        sanity_check_classes[selected_indices] += 1
        if tukeyed_samples:
            X[selected_indices] = np.power(X[selected_indices], 2)
        covariance = np.cov(X[selected_indices].T)
        for i in range(no_shrinkage):
            covariance = shrink_cov(covariance)
        if normalize_covariance:
            covariance = normalize_cov(covariance)
        covariance_matrices[class_no] = covariance
    desired_sanity_check = np.ones_like(y)
    assert_array_almost_equal(sanity_check_classes, desired_sanity_check)
    return covariance_matrices, prototypes


def extract_test_set_from_all_tasks(
    dataset_CL_tasks, number_of_incremental_tasks, total_number_of_tasks, device
):
    """
    Create a test set containing samples from all the considered tasks
    with corresponding labels (without forward propagation through network)
    and information about task.

    Arguments:
    ----------
       *dataset_CL_tasks* (list of datasets) list of objects storing training
                           and test samples from consecutive tasks
       *number_of_incremental_tasks* (int) the number of consecutive tasks
                                      from which the test sets will be
                                      extracted
       *total_number_of_tasks* (int) the number of all tasks in a given
                               experiment
       *device*: (str) 'cpu' or 'cuda', defines the equipment for computations

    Returns:
    --------
       *X_test* (torch Tensor) contains samples from the test set,
                shape: (number of samples, number of image features [e.g. 3072
                for CIFAR-100])
       *y_test* (Numpy array) contains labels for corresponding samples
                from *X_test* (number of samples, )
       *tasks_test* (Numpy array) contains information about task for
                    corresponding samples from *X_test* (number of samples, )
    """

    test_input_data, test_output_data, test_ID_tasks = [], [], []
    for t in range(number_of_incremental_tasks):
        tested_task = dataset_CL_tasks[t]
        input_test_data = tested_task.get_test_inputs()
        output_test_data = tested_task.get_test_outputs()
        test_input = tested_task.input_to_torch_tensor(
            input_test_data, device, mode="inference"
        )
        test_output = tested_task.output_to_torch_tensor(
            output_test_data, device, mode="inference"
        )
        gt_classes = test_output.max(dim=1)[1].cpu().detach().numpy()
        gt_classes = translate_output_CIFAR_classes(
            gt_classes, total_number_of_tasks, t
        )
        test_input_data.append(test_input)
        test_output_data.append(gt_classes)
        current_task_gt = np.zeros_like(gt_classes) + t
        test_ID_tasks.append(current_task_gt)
    X_test = torch.cat(test_input_data)
    y_test, tasks_test = np.concatenate(test_output_data), np.concatenate(
        test_ID_tasks
    )
    assert X_test.shape[0] == y_test.shape[0] == tasks_test.shape[0]
    return X_test, y_test, tasks_test


def calculate_mahalanobis_distances_between_all_samples_and_classes(
    hypernetwork,
    hypernetwork_weights,
    target_network,
    target_weights,
    target_network_type,
    X_sample,
    covariance_matrices,
    prototypes,
    dict_classes_per_task,
    sparsity_parameter,
    no_of_batch_norm_layers,
    number_of_incremental_tasks,
    normalize=True,
    tukeyed=False,
):
    """
    Calculate output of the target network for each hypernetwork
    embedding. For each embedding select the minimum distance across
    various classes, in terms of the Mahalanobis distance.

    Arguments:
    ----------
        *hypernetwork* (hypnettorch.hnets.mlp_hnet.HMLP) a hypernetwork object
        *hypernetwork_weights* (torch.nn.modules.containter.ParameterList)
                               contains weights for *hypernetwork*
        *target_network* a target network object; currently only ResNetF is
                         supported
        *target_weights* (torch.nn.modules.container.ParameterList) contains
                         weights of the target network
        *target_network_type* (str) defines name of the target architecture
        *X_sample* (torch Tensor) defines a batch of samples for inference
                   (shape: batch inference size, number of features [e.g.
                   3072 for CIFAR-100])
        *covariance_matrices* (dict of Numpy arrays) contains covariance
                              matrices per each class
        *prototypes* (dict of Numpy arrays) contains prototype per each class
        *dict_classes_per_task* (dictionary) defines labels for consecutive
                                tasks
        *sparsity_parameter* (int) defines a percentage of weights for removing
                             in consecutive layers
        *no_of_batch_norm_layers* (int) defines the number of batch
                                  normalization layer in the target network
        *number_of_incremental_tasks* (int) the number of consecutive tasks
                                      from which the test sets will be
                                      extracted
        *normalize*: (Boolean value) optional, defines whether the length
                    of each network output vector and class prototype
                    should be equal to one, default: True
        *tukeyed*: (Boolean value) optional, defines whether Tukey
                   transformation should be applied after forward propagation
                   through network, default: False

    Returns:
    --------
        *task_distances*: (Numpy array) defines nearest distances between
                          a given sample and a class which has the smallest
                          distance for a given task
        *task_class_names*: (Numpy array) stores labels for corresponding
                            tasks and samples from *task_distances*
    """
    task_distances, task_class_names = [], []
    for t in range(number_of_incremental_tasks):
        possible_classes = dict_classes_per_task[t]
        prediction = get_target_network_representation(
            hypernetwork,
            hypernetwork_weights,
            target_network,
            target_weights,
            target_network_type,
            X_sample,
            sparsity_parameter,
            no_of_batch_norm_layers,
            task=t,
        )
        prediction = prediction[0] if len(prediction) == 1 else prediction[1]
        if tukeyed:
            prediction = np.power(prediction, 2)
        if normalize:
            prediction = F.normalize(prediction, dim=-1).numpy()
        # Calculate distances between each prototype and the target's
        # representation
        different_class_distances = []
        for current_class in possible_classes:
            if normalize:
                difference = (
                    prediction
                    - F.normalize(
                        torch.from_numpy(prototypes[current_class]), dim=-1
                    ).numpy()
                )
            else:
                difference = prediction - prototypes[current_class]
            distances = _mahalanobis(
                difference, covariance_matrices[current_class]
            )
            different_class_distances.append(distances)
        different_class_distances = np.array(different_class_distances)
        distance_minimum_per_task = np.min(different_class_distances, axis=0)
        label_minimum_per_task = np.argmin(different_class_distances, axis=0)
        # Take the class corresponding to the minimum distance found
        class_minimum_per_task = np.take(
            possible_classes, label_minimum_per_task
        )
        task_distances.append(distance_minimum_per_task)
        task_class_names.append(class_minimum_per_task)
    task_distances = np.array(task_distances)
    task_class_names = np.array(task_class_names)
    assert task_distances.shape == task_class_names.shape
    return task_distances, task_class_names


def select_nearest_classes_for_samples(task_distances, task_class_names):
    """
    Select a class proper to given samples, according to the FeCAM approach

    Arguments:
    ----------
        *task_distances*: (Numpy array of floats) defines nearest distances
                          between a given sample and a class which has
                          the smallest distance for a given task
                          (shape: number of tasks, number of samples)
        *task_class_names*: (Numpy array of integers) stores classes that
                            have the lowest distances between them and
                            consecutive samples; one class per each task
                            (shape: number of tasks, number of samples)

    Returns:
    --------
        *nearest_classes_for_batch*: (Numpy array) contains labels of classes
                                     that have the lowest distance between
                                     the class prototype and given samples
                                     (shape: number of samples,)
    """
    indices_of_min_task_distances = np.argmin(task_distances, axis=0)
    indices_of_nearest_classes = np.vstack(
        [indices_of_min_task_distances, np.arange(task_distances.shape[1])]
    )
    nearest_classes_for_batch = task_class_names[
        indices_of_nearest_classes[0], indices_of_nearest_classes[1]
    ]
    return nearest_classes_for_batch


def unittest_select_nearest_classes_for_samples():
    """
    Unittest of select_nearest_classes_for_samples() function.
    """
    # 4 tasks, 3 samples
    task_distances_1 = np.array(
        [
            [0.12, 0.25, 0.001],
            [0.09, 0.01, 0.8],
            [0.07, 0.9, 0.002],
            [0.04, 1.1, 0.5],
        ]
    )
    task_class_names_1 = np.array(
        [[11, 12, 68], [24, 23, 34], [35, 58, 40], [47, 71, 51]]
    )
    gt_1 = np.array([47, 23, 68])
    result_1 = select_nearest_classes_for_samples(
        task_distances_1, task_class_names_1
    )
    assert_array_equal(gt_1, result_1)


def main_hypermask_with_fecam(
    experiment_models,
    no_of_covariance_shrinkage=2,
    normalize_covariance=True,
    normalize_features=True,
    apply_tukey=False,
):
    """
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
        *batch_train_set_size*: int related to the batch size during
                                the extraction of training points
        *batch_inference_size*: int related to the batch size during inference
        *setup*: int related to the total number of tasks in a given experiment
        *mode*: 'str': 'class' or 'task' defines whether FeCAM should be used
                for sample class selection using the representation from
                HyperMask or whether FeCAM should be used just for task
                selection and HyperMask as the final predictor on the embedding
                selected by FeCAM
        *accuracy_report_method*: 'str': 'HNET' or 'FeCAM'. In the first case,
                only the model after all tasks is considered as well as samples
                from consecutive tasks separately. For 'FeCAM' models after
                consecutive tasks are evaluated. Also, during the evaluation
                of i-th task, test sets from the previous tasks
                {0, 1, ..., i-1} are also included.
    *no_of_covariance_shrinkage*: (int) defines how many times covariance
                                  shrinking should be performed,
                                  optional: default 2
    *normalize_covariance*: (Boolean value) defines whether normalization
                            of covariance matrices should be performed,
                            optional: default True
    *normalize_features*: (Boolean value) defines whether normalization
                          of features extracted by the network as well as
                          prototype normalization should be performed
                          otpional: default True
    *apply_tukey*: (Boolean value) defines whether Tukey transformation
                   will be performed, optional: default False

    Returns:
    --------
        *summary_of_results*: (list) contains the following elements:
           - consecutive class prediction accuracies,
           - consecutive task prediction accuracies (in 'task' mode),
           - mean class prediction accuracy (with std. dev.)
           - mean task prediction accuracy (with std. dev.), only in 'task'
           mode.
    """
    hypernetwork = experiment_models["hypernetwork"]
    hypernetwork_weights = experiment_models["hypernetwork_weights"]
    target_network = experiment_models["target_network"]
    target_weights = experiment_models["target_network_weights"]
    hyperparameters = experiment_models["hyperparameters"]
    dataset_CL_tasks = experiment_models["list_of_CL_tasks"]
    no_of_batch_norm_layers = experiment_models["no_of_batch_norm_layers"]
    target_network_type = hyperparameters["target_network"]
    batch_train_set_size = experiment_models["batch_train_set_size"]
    batch_inference_size = experiment_models["batch_inference_size"]
    dataset_name = experiment_models["hyperparameters"]["dataset"]
    setup = experiment_models["setup"]
    mode = experiment_models["mode"]

    if dataset_name == "CIFAR100_FeCAM_setup":
        accuracy_report_method = "FeCAM"
    elif dataset_name in ["PermutedMNIST", "SplitMNIST"]:
        accuracy_report_method = "HNET"
    else:
        raise ValueError("This dataset is currently not implemented!")

    hypernetwork.eval()
    target_network.eval()

    if target_network_type == "ResNet":
        target_network_type == "ResNetF"

    class_predictions = []
    if mode == "task":
        task_predictions = []

    if accuracy_report_method == "FeCAM":
        considered_tasks = range(1, setup + 1)
    elif accuracy_report_method == "HNET":
        considered_tasks = [setup]
    for number_of_incremental_tasks in considered_tasks:
        # --FeCAM evaluation method--
        # Iteration over tasks - incremental learning scenario:
        # During the evaluation of the i-th task, tasks {0, 1, 2, ..., i}
        # are evaluated.
        # --HNET evaluation method--
        # Only last model is considered. During the evaluation of the i-th
        # task only its test set is evaluated.
        # Prepare training set containing samples and corresponding labels
        (
            X,
            y,
            tasks_gt,
            dict_classes_per_task,
        ) = extract_training_set_from_all_tasks(
            hypernetwork,
            hypernetwork_weights,
            target_network,
            target_weights,
            target_network_type,
            dataset_CL_tasks,
            dataset_name,
            number_of_incremental_tasks,
            setup,
            hyperparameters["sparsity_parameters"][0],
            batch_train_set_size,
            no_of_batch_norm_layers,
            hyperparameters["device"],
            tukeyed=apply_tukey,
        )
        unique_objects = np.concatenate(list(dict_classes_per_task.values()))
        # For each class occurring in the training set calculate a covariance
        # matrix and a class prototype
        (
            covariance_matrices,
            prototypes,
        ) = create_covariance_matrices_and_prototypes(
            X,
            y,
            unique_objects,
            no_shrinkage=no_of_covariance_shrinkage,
            normalize_covariance=normalize_covariance,
            tukeyed_samples=apply_tukey,
            tukeyed_prototypes=apply_tukey,
        )
        # Load the test set
        X_tests, y_tests, gt_tasks_tests = [], [], []
        if accuracy_report_method == "FeCAM":
            X_test, y_test, gt_tasks_test = extract_test_set_from_all_tasks(
                dataset_CL_tasks,
                number_of_incremental_tasks,
                setup,
                hyperparameters["device"],
            )
            X_tests.append(X_test)
            y_tests.append(y_test)
            gt_tasks_tests.append(gt_tasks_test)
        elif accuracy_report_method == "HNET":
            for current_no_of_task in range(number_of_incremental_tasks):
                (
                    X_test,
                    y_test,
                    gt_tasks_test,
                ) = extract_test_set_from_single_task(
                    dataset_CL_tasks,
                    current_no_of_task,
                    dataset_name,
                    hyperparameters["device"],
                )
                X_tests.append(X_test)
                y_tests.append(y_test)
                gt_tasks_tests.append(gt_tasks_test)
        for no, (X_test, y_test, gt_tasks_test) in enumerate(
            zip(X_tests, y_tests, gt_tasks_tests)
        ):
            no_of_batches = X_test.shape[0] // batch_inference_size
            if X_test.shape[0] % batch_inference_size > 0.0:
                no_of_batches += 1
            results_nearest_classes = []
            if mode == "task":
                results_nearest_tasks = []
            for i in range(no_of_batches):
                X_sample = X_test[
                    (batch_inference_size * i) : (
                        batch_inference_size * (i + 1)
                    )
                ]
                # Calculate outputs of the classification layer using all embeddings
                (
                    task_distances,
                    task_class_names,
                ) = calculate_mahalanobis_distances_between_all_samples_and_classes(
                    hypernetwork,
                    hypernetwork_weights,
                    target_network,
                    target_weights,
                    target_network_type,
                    X_sample,
                    covariance_matrices,
                    prototypes,
                    dict_classes_per_task,
                    hyperparameters["sparsity_parameters"][0],
                    no_of_batch_norm_layers,
                    number_of_incremental_tasks,
                    normalize=normalize_features,
                    tukeyed=apply_tukey,
                )
                # Finally, directly select the class with the lowest
                # distance to the target's output (minimum of the minima) [mode: class]
                # OR get predictions from HyperMask [mode: task].
                if mode == "class":
                    current_nearest_objects = (
                        select_nearest_classes_for_samples(
                            task_distances, task_class_names
                        )
                    )
                elif mode == "task":
                    selected_tasks = np.argmin(task_distances, axis=0)
                    results_nearest_tasks.append(selected_tasks)
                    current_nearest_objects = (
                        evaluate_hypermask_with_selected_embedding(
                            hypernetwork,
                            hypernetwork_weights,
                            target_network,
                            target_weights,
                            target_network_type,
                            dataset_name,
                            X_sample,
                            selected_tasks,
                            hyperparameters["sparsity_parameters"][0],
                            no_of_batch_norm_layers,
                            number_of_incremental_tasks,
                            setup,
                        )
                    )
                results_nearest_classes.append(current_nearest_objects)
            if accuracy_report_method == "FeCAM":
                print(f"After {number_of_incremental_tasks} incremental task")
            elif accuracy_report_method == "HNET":
                print(
                    f"Evaluation of {no} task from the "
                    f"{number_of_incremental_tasks} model"
                )
            results_nearest_classes = np.concatenate(results_nearest_classes)
            sample_prediction_accuracy = (
                np.sum(results_nearest_classes == y_test)
                * 100.0
                / y_test.shape[0]
            )
            class_predictions.append(sample_prediction_accuracy)
            print(f"Class selection accuracy: {sample_prediction_accuracy}")
            if mode == "task":
                results_nearest_tasks = np.concatenate(results_nearest_tasks)
                task_selection_accuracy = (
                    np.sum(results_nearest_tasks == gt_tasks_test)
                    * 100.0
                    / gt_tasks_test.shape[0]
                )
                task_predictions.append(task_selection_accuracy)
                print(f"Task selection accuracy: {task_selection_accuracy}")
    summary_of_results = [class_predictions]
    if mode == "task":
        summary_of_results.append(task_predictions)
    for element in deepcopy(summary_of_results):
        summary_of_results.extend([np.mean(element), np.std(element)])
    return summary_of_results


def evaluate_hypermask_with_selected_embedding(
    hypernetwork,
    hypernetwork_weights,
    target_network,
    target_weights,
    target_network_type,
    dataset_name,
    X_sample,
    current_nearest_tasks,
    sparsity_parameter,
    no_of_batch_norm_layers,
    number_of_incremental_tasks,
    total_number_of_tasks,
):
    """
    Get the output of HyperMask for each sample after the selection
    of embeddings by another method, e.g. FeCAM.

    Arguments:
    ----------
        *hypernetwork* (hypnettorch.hnets.mlp_hnet.HMLP) a hypernetwork object
        *hypernetwork_weights* (torch.nn.modules.containter.ParameterList)
                               contains weights for *hypernetwork*
        *target_network* a target network object; currently only ResNetF is
                         supported
        *target_weights* (torch.nn.modules.container.ParameterList) contains
                         weights of the target network
        *target_network_type* (str) defines name of the target architecture
        *dataset_name*: (str) defines name of the dataset used:
                        'PermutedMNIST', 'SplitMNIST' or 'CIFAR100_FeCAM_setup'
        *X_sample* (torch Tensor) defines a batch of samples for inference
                   (shape: batch inference size, number of features [e.g.
                   3072 for CIFAR-100])
        *current_nearest_tasks* (Numpy arrays of ints) select tasks that
                                  have been selected by another method,
                                  e.g. FeCAM
        *sparsity_parameter* (int) defines a percentage of weights for removing
                             in consecutive layers
        *no_of_batch_norm_layers* (int) defines the number of batch
                                  normalization layer in the target network
        *number_of_incremental_tasks* (int) the number of consecutive tasks
                                      from which the test sets will be
                                      extracted
        *total_number_of_tasks* (int) the number of all tasks in a given
                                experiment

    Returns:
    --------
        *hypermask_results* (Numpy array of ints) stores classes selected
                            by HyperMask after choosing the embedding by
                            another method, e.g. FeCAM
    """
    # It is necessary to run the network with embedding selected for
    # consecutive samples. Therefore, we have to prepare masks with indices
    # for different embeddings.
    hypermask_results = np.zeros(X_sample.shape[0], dtype=int)
    for j in range(number_of_incremental_tasks):
        cur_task_indices = np.argwhere(current_nearest_tasks == j).reshape(-1)
        sample_for_this_task = X_sample[cur_task_indices]
        logits = get_target_network_representation(
            hypernetwork,
            hypernetwork_weights,
            target_network,
            target_weights,
            target_network_type,
            sample_for_this_task,
            sparsity_parameter,
            no_of_batch_norm_layers,
            j,
        )
        logits = logits[0]
        output_classes = logits.max(dim=1)[1].cpu().detach().numpy()

        if dataset_name == "CIFAR100_FeCAM_setup":
            output_classes = translate_output_CIFAR_classes(
                output_classes, total_number_of_tasks, j
            )
        elif dataset_name in ["PermutedMNIST", "SplitMNIST"]:
            mode = "permuted" if dataset_name == "PermutedMNIST" else "split"
            output_classes = translate_output_MNIST_classes(
                output_classes, task=j, mode=mode
            )
        hypermask_results[cur_task_indices] = output_classes
    return hypermask_results


if __name__ == "__main__":
    unittest_translate_output_CIFAR_classes()
    unittest_translate_output_MNIST_classes()
    unittest_select_nearest_classes_for_samples()
    dataset = "CIFAR100_FeCAM_setup"
    mode = "class"
    path_to_datasets = "./Data/"

    if dataset == "PermutedMNIST":
        part = 0
        setup = 10
    elif dataset == "SplitMNIST":
        part = 0
        setup = 5
    elif dataset == "CIFAR100_FeCAM_setup":
        part = 6
        # ResNet, 5 tasks, 20 classes per each task
        setup = 5
    else:
        raise ValueError("This dataset is currenly not implemented!")
    path_to_stored_networks = f"./Models/{dataset}/"
    path_to_save = f"./Results/{dataset}/"
    os.makedirs(path_to_save, exist_ok=True)
    batch_train_set_size = 2000
    batch_inference_size = 2000
    # The results are varying depending on the batch sizes due to the fact
    # that batch normalization is turned on in ResNet. We selected 2000 as
    # the test set size to ensure that it is derived to the network
    # in one piece.
    no_of_covariance_shrinkage = 2
    normalize_covariance = True
    normalize_features = True
    apply_tukey = False

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
            fecam_validation=True,
        )
        experiment_models["batch_train_set_size"] = batch_train_set_size
        experiment_models["batch_inference_size"] = batch_inference_size
        experiment_models["setup"] = setup
        experiment_models["mode"] = mode

        results = main_hypermask_with_fecam(
            experiment_models,
            no_of_covariance_shrinkage=no_of_covariance_shrinkage,
            normalize_covariance=normalize_covariance,
            normalize_features=normalize_features,
            apply_tukey=apply_tukey,
        )
        results_summary.append(results)

    if mode == "class":
        column_names = [
            "class_prediction_accuracy",
            "mean_class_prediction_accuracy",
            "std_dev_class_prediction_accuracy",
        ]
    elif mode == "task":
        column_names = [
            f"class_prediction_accuracy_from_{batch_inference_size}_batches",
            "task_prediction_accuracy",
            "mean_class_prediction_accuracy",
            "std_dev_class_prediction_accuracy",
            "mean_task_prediction_accuracy",
            "std_dev_task_prediction_accuracy",
        ]
    dataframe = pd.DataFrame(results_summary, columns=column_names)
    dataframe.to_csv(
        f"{path_to_save}HyperMask_with_FeCAM_mean_results.csv", sep=";"
    )

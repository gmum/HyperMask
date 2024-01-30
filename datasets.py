import os
import numpy as np
import torch
from hypnettorch.data.special import permuted_mnist
from hypnettorch.data.special.split_cifar import SplitCIFAR100Data
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from TinyImageNet import TinyImageNet
from CIFAR100_FeCAM import SplitCIFAR100Data_FeCAM


def generate_random_permutations(
    shape_of_data_instance, number_of_permutations
):
    """
    Prepare a list of random permutations of the selected shape
    for continual learning tasks.

    Arguments:
    ----------
      *shape_of_data_instance*: a number defining shape of the dataset
      *number_of_permutations*: int, a number of permutations that will
                                be prepared; it corresponds to the total
                                number of tasks
      *seed*: int, optional argument, default: None
              if one would get deterministic results
    """
    list_of_permutations = []
    for _ in range(number_of_permutations):
        list_of_permutations.append(
            np.random.permutation(shape_of_data_instance)
        )
    return list_of_permutations


def prepare_split_cifar100_tasks(
    datasets_folder, validation_size, use_augmentation, use_cutout=False
):
    """
    Prepare a list of 10 tasks with 10 classes per each task.
    i-th task, where i in {0, 1, ..., 9} will store samples
    from classes {10*i, 10*i + 1, ..., 10*i + 9}.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which CIFAR-100
                         is stored / will be downloaded
      *validation_size*: (int) The number of validation samples
      *use_augmentation*: (Boolean) potentially applies
                          a data augmentation method from
                          hypnettorch
      *use_cutout*: (optional Boolean) in the positive case it applies
                    'apply_cutout' option form 'torch_input_transforms'.
    """
    handlers = []
    for i in range(0, 100, 10):
        handlers.append(
            SplitCIFAR100Data(
                datasets_folder,
                use_one_hot=True,
                validation_size=validation_size,
                use_data_augmentation=use_augmentation,
                use_cutout=use_cutout,
                labels=range(i, i + 10),
            )
        )
    return handlers


def prepare_split_cifar100_tasks_aka_FeCAM(
    datasets_folder,
    number_of_tasks,
    no_of_validation_samples_per_class,
    use_augmentation,
    use_cutout=False,
):
    """
    Prepare a list of 5, 10 or 20 incremental tasks with 20, 10 or 5 classes,
    respectively, per each task. Furthermore, the first task contains
    a higher number of classes, i.e. 50 or 40. Therefore, in these cases,
    the total number of tasks is equal to 6, 11 or 21.
    Also, there is a possibility of 5 tasks with 20 classes per each.
    The order of classes is the same like in FeCAM, also the scenarios
    are constructed in such a way to enable a fair comparison with FeCAM

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which CIFAR-100
                         is stored / will be downloaded
      *number_of_tasks* (int) Defines how many continual learning tasks
                        will be created. Possible options: 6, 11 or 21
      *no_of_validation_samples_per_class*: (int) The number of validation
                                            samples in a single class
      *use_augmentation*: (Boolean) potentially applies
                          a data augmentation method from
                          hypnettorch
      *use_cutout*: (optional Boolean) in the positive case it applies
                    'apply_cutout' option form 'torch_input_transforms'.
    """
    # FeCAM considered four scenarios: 5, 10 and 20 incremental tasks
    # and 5 tasks with the equal number of classes
    assert number_of_tasks in [5, 6, 11, 21]
    # The order of image classes in the case of FeCAM was not 0-10, 11-20, etc.,
    # but it was chosen randomly by the authors, and was at follows:
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
    # Incremental tasks from Table I, FeCAM
    if number_of_tasks == 6:
        numbers_of_classes_per_tasks = [50]
        numbers_of_classes_per_tasks.extend([10 for i in range(5)])
    elif number_of_tasks == 11:
        numbers_of_classes_per_tasks = [50]
        numbers_of_classes_per_tasks.extend([5 for i in range(10)])
    elif number_of_tasks == 21:
        numbers_of_classes_per_tasks = [40]
        numbers_of_classes_per_tasks.extend([3 for i in range(20)])
    # Tasks with the equal number of elements, Table V, FeCAM
    elif number_of_tasks == 5:
        numbers_of_classes_per_tasks = [20 for i in range(5)]

    handlers = []
    for i in range(len(numbers_of_classes_per_tasks)):
        current_number_of_tasks = numbers_of_classes_per_tasks[i]
        validation_size = (
            no_of_validation_samples_per_class * current_number_of_tasks
        )
        handlers.append(
            SplitCIFAR100Data_FeCAM(
                datasets_folder,
                use_one_hot=True,
                validation_size=validation_size,
                use_data_augmentation=use_augmentation,
                use_cutout=use_cutout,
                labels=class_orders[
                    (i * current_number_of_tasks) : (
                        (i + 1) * current_number_of_tasks
                    )
                ],
            )
        )
    return handlers


def prepare_tinyimagenet_tasks(
    datasets_folder, seed, validation_size=250, number_of_tasks=40
):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the TinyImageNet dataset according to the WSN setup.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which TinyImageNet
                         is stored / will be downloaded
      *seed*: (int) Necessary for the preparation of random permutation
              of the order of classes in consecutive tasks.
      *validation_size*: (optional int) defines the number of validation
                         samples in each task, by default it is 250 like
                         in the case of WSN
      *number_of_tasks*: (optional int) defines the number of continual
                         learning tasks (by default: 40)

    Returns a list of TinyImageNet objects.
    """
    # Set randomly the order of classes
    rng = np.random.default_rng(seed)
    class_permutation = rng.permutation(200)
    # 40 classification tasks with 5 classes in each
    handlers = []
    for i in range(0, 5 * number_of_tasks, 5):
        current_labels = class_permutation[i : (i + 5)]
        print(f"Order of classes in the current task: {current_labels}")
        handlers.append(
            TinyImageNet(
                data_path=datasets_folder,
                validation_size=validation_size,
                use_one_hot=True,
                labels=current_labels,
            )
        )
    return handlers


def prepare_permuted_mnist_tasks(
    datasets_folder, input_shape, number_of_tasks, padding, validation_size
):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the PermutedMNIST dataset.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *input_shape*: (int) a number defining shape of the dataset
      *validation_size*: (int) The number of validation samples

    Returns a list of PermutedMNIST objects.
    """
    permutations = generate_random_permutations(input_shape, number_of_tasks)
    return permuted_mnist.PermutedMNISTList(
        permutations,
        datasets_folder,
        use_one_hot=True,
        padding=padding,
        validation_size=validation_size,
    )


def prepare_split_mnist_tasks(
    datasets_folder, validation_size, use_augmentation, number_of_tasks=5
):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the SplitMNIST dataset. By default, it should be
    5 task containing consecutive pairs of classes:
    [0, 1], [2, 3], [4, 5], [6, 7] and [8, 9].

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *validation_size*: (int) The number of validation samples
      *use_augmentation*: (bool) defines whether dataset augmentation
                          will be applied
      *number_of_tasks* (int) a number defining the number of learning
                        tasks, by default 5.

    Returns a list of SplitMNIST objects.
    """
    return get_split_mnist_handlers(
        datasets_folder,
        use_one_hot=True,
        validation_size=validation_size,
        num_classes_per_task=2,
        num_tasks=number_of_tasks,
        use_torch_augmentation=use_augmentation,
    )


def set_hyperparameters(dataset, grid_search=False, part=0):
    """
    Set hyperparameters of the experiments, both in the case of grid search
    optimization and a single network run.

    Arguments:
    ----------
      *dataset*: 'PermutedMNIST', 'SplitMNIST' or 'CIFAR100'
      *grid_search*: (Boolean optional) defines whether a hyperparameter
                     optimization should be performed or hyperparameters
                     for just a single run have to be returned
      *part* (only for SplitMNIST or CIFAR100!) selects a subset
             of hyperparameters for optimization (by default 0)

    Returns a dictionary with necessary hyperparameters.
    """
    if dataset == "PermutedMNIST":
        if grid_search:
            hyperparams = {
                "embedding_sizes": [24],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "norm_regularizer_masking_opts": [True, False],
                "betas": [0.001, 0.0005, 0.005],
                "hypernetworks_hidden_layers": [[100, 100]],
                "sparsity_parameters": [0],
                "lambdas": [0.001, 0.0005],
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/grid_search/permuted_mnist/",
                # not for optimization, just for multiple cases
                "seed": [1, 2, 3, 4, 5],
            }

        else:
            # Best hyperparameters
            hyperparams = {
                "seed": [1, 2, 3, 4, 5],
                "embedding_sizes": [24],
                "sparsity_parameters": [0],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.0005],
                "lambdas": [0.001],
                "norm_regularizer_masking_opts": [True],
                "hypernetworks_hidden_layers": [[100, 100]],
                "best_model_selection_method": "last_model",
                "saving_folder": "./Results/permuted_mnist_best_hyperparams/",
            }

        # Both in the grid search and individual runs
        hyperparams["lr_scheduler"] = False
        hyperparams["number_of_iterations"] = 5000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 5000
        hyperparams["no_of_validation_samples_per_class"] = 500
        hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["target_network"] = "MLP"
        hyperparams["resnet_number_of_layer_groups"] = None
        hyperparams["resnet_widening_factor"] = None
        hyperparams["optimizer"] = "adam"
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 8
        hyperparams["use_chunks"] = False
        hyperparams["adaptive_sparsity"] = True
        hyperparams["use_batch_norm"] = False
        # Directly related to the MNIST dataset
        hyperparams["padding"] = 2
        hyperparams["shape"] = (28 + 2 * hyperparams["padding"]) ** 2
        hyperparams["number_of_tasks"] = 10
        hyperparams["augmentation"] = False

    elif dataset == "CIFAR100":
        if grid_search:
            hyperparams = {
                "seed": [5],
                "sparsity_parameters": [0],
                "embedding_sizes": [48],
                "betas": [0.01, 0.05, 0.1, 1],
                "lambdas": [0.01, 0.1, 1],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": True,
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
                hyperparams["use_chunks"] = False
                hyperparams["use_batch_norm"] = False
            else:
                raise ValueError(f"Wrong argument: {part}!")
            hyperparams[
                "saving_folder"
            ] = f"./Results/grid_search/CIFAR_100_part_{part}/"

        else:
            # Best hyperparameters for ResNet
            hyperparams = {
                "seed": [1, 2, 3, 4, 5],
                "embedding_sizes": [48],
                "sparsity_parameters": [0],
                "betas": [0.01],
                "lambdas": [1],
                "batch_sizes": [32],
                "learning_rates": [0.001],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[100]],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
            }
            if part == 0:
                # ResNet
                pass
            elif part == 1:
                # ZenkeNet
                hyperparams["lambdas"] = [0.01]
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
            else:
                raise ValueError(f"Wrong argument: {part}!")
            hyperparams[
                "saving_folder"
            ] = f"./Results/CIFAR_100_best_hyperparams_part_{part}/"
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 500
        hyperparams["no_of_validation_samples_per_class"] = 50
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 10
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["adaptive_sparsity"] = True
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    elif dataset == "TinyImageNet":
        if grid_search:
            hyperparams = {
                "seed": [5],
                "sparsity_parameters": [0, 30],
                "embedding_sizes": [48],
                "betas": [0.01, 0.1],
                "lambdas": [0.01, 0.1],
                "learning_rates": [0.001],
                "batch_sizes": [16],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[10, 10], [100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True,
                "saving_folder": f"./Results/TinyImageNet_grid_search_part_{part}/",
            }
            if part == 0:
                pass
            elif part in [1, 2, 3]:
                # ZenkeNet
                hyperparams["seed"] = [6]
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
                hyperparams["use_batch_norm"] = False
                hyperparams["hypernetworks_hidden_layers"] = [[100, 100]]
                hyperparams["learning_rates"] = [0.001]
                if part == 1:
                    hyperparams["betas"] = [0.01, 0.1, 1.0]
                    hyperparams["lambdas"] = [0.01, 0.1, 1.0]
                    hyperparams["sparsity_parameters"] = [0, 50, 70]
                    hyperparams["embedding_sizes"] = [96]
                    hyperparams["learning_rates"] = [0.005]
                elif part == 2:
                    hyperparams["embedding_sizes"] = [128]
                    hyperparams["betas"] = [0.01, 0.1, 1.0]
                    hyperparams["lambdas"] = [0.01, 0.1, 1.0]
                    hyperparams["sparsity_parameters"] = [0, 30, 50]
                elif part == 3:
                    hyperparams["embedding_sizes"] = [192]
                    hyperparams["sparsity_parameters"] = [0, 50]
                    hyperparams["betas"] = [0.001, 0.01, 0.1, 1]
                    hyperparams["lambdas"] = [0.001, 0.01, 0.1, 1]
            else:
                raise ValueError(f"Wrong argument: {part}!")
            hyperparams[
                "saving_folder"
            ] = f"./Results/Tiny_Zenke_grid_search_part_{part}/"
        else:
            # ResNet
            hyperparams = {
                "seed": [5, 6, 7, 8, 9],
                "embedding_sizes": [96],
                "sparsity_parameters": [0],
                "betas": [1],
                "lambdas": [0.1],
                "batch_sizes": [16],
                "learning_rates": [0.0001],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[100, 100]],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 10,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
                "saving_folder": "./Results/TinyImageNet/ResNet_best_hyperparams/",
            }
            if part == 0:
                pass
            # ZenkeNet
            elif part == 1:
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
                hyperparams["use_batch_norm"] = False
                hyperparams["sparsity_parameters"] = [50]
                hyperparams["betas"] = [0.01]
                hyperparams["lambdas"] = [1.0]
                hyperparams["learning_rates"] = [0.001]
                hyperparams[
                    "saving_folder"
                ] = f"./Results/TinyImageNet/ZenkeNet_best_hyperparams/"
            else:
                raise ValueError(f"Wrong argument: {part}!")
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 250
        hyperparams["no_of_validation_samples_per_class"] = 50
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 64
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 12288
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 40
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["adaptive_sparsity"] = True
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    elif dataset == "SplitMNIST":
        if grid_search:
            hyperparams = {
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "norm_regularizer_masking_opts": [False],
                "betas": [0.001],
                "hypernetworks_hidden_layers": [[25, 25]],
                "sparsity_parameters": [0],
                "lambdas": [0.001],
                # Seed is not for optimization but for ensuring multiple results
                "seed": [1, 2, 3, 4, 5],
                "best_model_selection_method": "last_model",
                "embedding_sizes": [128],
                "augmentation": True,
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams["embedding_sizes"] = [96]
                hyperparams["hypernetworks_hidden_layers"] = [[50, 50]]
                hyperparams["betas"] = [0.01]
                hyperparams["sparsity_parameters"] = [30]
                hyperparams["lambdas"] = [0.0001]
            elif part == 2:
                hyperparams["sparsity_parameters"] = [30]
                hyperparams["norm_regularizer_masking_opts"] = [True]
            else:
                raise ValueError("Not implemented subset of hyperparameters!")

            hyperparams["saving_folder"] = "./Results/grid_search/split_mnist/"

        else:
            # Best hyperparameters
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [128],
                "sparsity_parameters": [30],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.001],
                "lambdas": [0.001],
                "norm_regularizer_masking_opts": [True],
                "hypernetworks_hidden_layers": [[25, 25]],
                "augmentation": True,
                "best_model_selection_method": "last_model",
                "saving_folder": "./Results/split_mnist_test/",
            }
        hyperparams["lr_scheduler"] = False
        hyperparams["target_network"] = "MLP"
        hyperparams["resnet_number_of_layer_groups"] = None
        hyperparams["resnet_widening_factor"] = None
        hyperparams["optimizer"] = "adam"
        hyperparams["number_of_iterations"] = 2000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 1000
        hyperparams["no_of_validation_samples_per_class"] = 500
        hyperparams["target_hidden_layers"] = [400, 400]
        hyperparams["shape"] = 28**2
        hyperparams["number_of_tasks"] = 5
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 96
        hyperparams["use_chunks"] = False
        hyperparams["use_batch_norm"] = False
        hyperparams["adaptive_sparsity"] = True
        hyperparams["padding"] = None

    elif dataset == "CIFAR100_FeCAM_setup":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "sparsity_parameters": [0],
                "betas": [0.01],
                "lambdas": [1],
                "batch_sizes": [32],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[100]],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams["embedding_sizes"] = [24, 48, 96]
                hyperparams["learning_rates"] = [0.0001, 0.001, 0.01]
                hyperparams["hypernetworks_hidden_layers"] = [[100], [200]]
                hyperparams["number_of_tasks"] = 5
            hyperparams[
                "saving_folder"
            ] = f"./Results/CIFAR_100_FeCAM_setup_part_{part}/"
        else:
            # Best hyperparameters for ResNet
            hyperparams = {
                "seed": [1, 2, 3, 4, 5],
                "embedding_sizes": [48],
                "sparsity_parameters": [0],
                "betas": [0.01],
                "lambdas": [1],
                "batch_sizes": [32],
                "learning_rates": [0.0001],
                "norm_regularizer_masking_opts": [False],
                "hypernetworks_hidden_layers": [[200]],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True,
            }
            # FeCAM considered three incremental scenarios: with 6, 11 and 21 tasks
            # ResNet - parts 0, 1 and 2
            # ZenkeNet - parts 3, 4 and 5
            # Also, one scenario with equal number of classes: ResNet - part 6
            if part in [0, 3]:
                hyperparams["number_of_tasks"] = 6
            elif part in [1, 4]:
                hyperparams["number_of_tasks"] = 11
            elif part in [2, 5]:
                hyperparams["number_of_tasks"] = 21
            elif part in [6, 7]:
                hyperparams["number_of_tasks"] = 5
            if part in [3, 4, 5, 7]:
                hyperparams["lambdas"] = [0.01]
                hyperparams["target_network"] = "ZenkeNet"
                hyperparams["resnet_number_of_layer_groups"] = None
                hyperparams["resnet_widening_factor"] = None
            if part not in [0, 1, 2, 3, 4, 5, 6, 7]:
                raise ValueError(f"Wrong argument: {part}!")
            hyperparams[
                "saving_folder"
            ] = f"./Results/CIFAR_100_FeCAM_part_{part}/"
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples_per_class"] = 50
        if hyperparams["target_network"] in ["ResNet", "ResNetF", "ZenkeNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["adaptive_sparsity"] = True
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    else:
        raise ValueError("This dataset is not implemented!")

    # General hyperparameters
    hyperparams["activation_function"] = torch.nn.ELU()
    hyperparams["norm"] = 1  # L1 norm
    hyperparams["use_bias"] = True
    hyperparams["save_consecutive_masks"] = False
    hyperparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams["dataset"] = dataset
    os.makedirs(hyperparams["saving_folder"], exist_ok=True)
    return hyperparams


if __name__ == "__main__":
    datasets_folder = "./Data"
    os.makedirs(datasets_folder, exist_ok=True)
    validation_size = 500
    use_data_augmentation = False
    use_cutout = False

    split_cifar100_list = prepare_split_cifar100_tasks(
        datasets_folder, validation_size, use_data_augmentation, use_cutout
    )

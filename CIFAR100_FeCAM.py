# Modification of hypnettorch file
# (https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/data/cifar100_data.html#CIFAR100Data)
# licensed under the Apache License, Version 2.0
#
# HyperMask with FeCAM needed some modifications during loading of CIFAR-100.


import os
import random
import numpy as np
import time
import torch
import _pickle as pickle
import urllib.request
import tarfile
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image, ImageEnhance, ImageOps
from hypnettorch.data.dataset import Dataset
from hypnettorch.data.cifar10_data import CIFAR10Data
from hypnettorch.data.special.split_cifar import _transform_split_outputs


class CIFAR100Data(Dataset):
    """An instance of the class shall represent the CIFAR-100 dataset.

    Args:
        data_path (str): Where should the dataset be read from? If not
            existing, the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.
        use_data_augmentation (bool): Note, this option currently only applies
            to input batches that are transformed using the class member
            :meth:`input_to_torch_tensor` (hence, **only available for
            PyTorch**, so far).

            Note:
                If activated, the statistics of test samples are changed as
                a normalization is applied (identical to the of class
                :class:`data.cifar10_data.CIFAR10Data`).
        validation_size (int): The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        use_cutout (bool): Whether option ``apply_cutout`` should be set of
            method :meth:`torch_input_transforms`. We use cutouts of size
            ``8 x 8`` as recommended
            `here <https://arxiv.org/pdf/1708.04552.pdf>`__.

            Note:
                Only applies if ``use_data_augmentation`` is set.
    """

    _DOWNLOAD_PATH = "https://www.cs.toronto.edu/~kriz/"
    _DOWNLOAD_FILE = "cifar-100-python.tar.gz"
    _EXTRACTED_FOLDER = "cifar-100-python"

    _TRAIN_BATCH_FN = "train"
    _TEST_BATCH_FN = "test"
    _META_DATA_FN = "meta"

    def __init__(
        self,
        data_path,
        use_one_hot=False,
        use_data_augmentation=False,
        validation_size=5000,
        use_cutout=False,
    ):
        super().__init__()

        start = time.time()

        print("Reading CIFAR-100 dataset ...")

        if not os.path.exists(data_path):
            print('Creating directory "%s" ...' % (data_path))
            os.makedirs(data_path)

        extracted_data_dir = os.path.join(
            data_path, CIFAR100Data._EXTRACTED_FOLDER
        )

        archive_fn = os.path.join(data_path, CIFAR100Data._DOWNLOAD_FILE)

        if not os.path.exists(extracted_data_dir):
            print("Downloading dataset ...")
            urllib.request.urlretrieve(
                CIFAR100Data._DOWNLOAD_PATH + CIFAR100Data._DOWNLOAD_FILE,
                archive_fn,
            )

            # Extract downloaded dataset.
            tar = tarfile.open(archive_fn, "r:gz")
            tar.extractall(path=data_path)
            tar.close()

            os.remove(archive_fn)

        train_batch_fn = os.path.join(
            extracted_data_dir, CIFAR100Data._TRAIN_BATCH_FN
        )
        test_batch_fn = os.path.join(
            extracted_data_dir, CIFAR100Data._TEST_BATCH_FN
        )
        meta_fn = os.path.join(extracted_data_dir, CIFAR100Data._META_DATA_FN)

        assert (
            os.path.exists(train_batch_fn)
            and os.path.exists(test_batch_fn)
            and os.path.exists(meta_fn)
        )

        self._data["classification"] = True
        self._data["sequence"] = False
        self._data["num_classes"] = 100
        self._data["is_one_hot"] = use_one_hot

        self._data["in_shape"] = [32, 32, 3]
        self._data["out_shape"] = [100 if use_one_hot else 1]

        # Fill the remaining _data fields with the information read from
        # the downloaded files.
        self._read_meta(meta_fn)
        self._read_batches(train_batch_fn, test_batch_fn, validation_size)

        # Initialize PyTorch data augmentation.
        self._augment_inputs = False
        if use_data_augmentation:
            self._augment_inputs = True
            self._torch_input_transforms()
        end = time.time()
        print("Elapsed time to read dataset: %f sec" % (end - start))

    def _read_meta(self, filename):
        """Read the meta data file.

        This method will add an additional field to the _data attribute named
        "cifar100". This dictionary will be filled with two members:
            * "fine_label_names": The names of the associated categorical class
                labels.
            * "coarse_label_names": The names of the 20 coarse labels that are
                associated to each sample.

        Args:
            filename: The path to the meta data file.
        """
        with open(filename, "rb") as f:
            meta_data = pickle.load(f, encoding="UTF-8")

        self._data["cifar100"] = dict()

        self._data["cifar100"]["fine_label_names"] = meta_data[
            "fine_label_names"
        ]
        self._data["cifar100"]["coarse_label_names"] = meta_data[
            "coarse_label_names"
        ]

    def _read_batches(self, train_fn, test_fn, validation_size):
        """Read training and testing batch from files.

        The method fills the remaining mandatory fields of the _data attribute,
        that have not been set yet in the constructor.

        The images are converted to match the output shape (32, 32, 3) and
        scaled to have values between 0 and 1. For labels, the correct encoding
        is enforced.

        Args:
            train_fn: Filepath of the train batch.
            test_fn: Filepath of the test batch.
            validation_size: Number of validation samples.
        """
        # Read test batch.
        with open(test_fn, "rb") as f:
            test_batch = pickle.load(f, encoding="bytes")

        # Note, that we ignore the two keys: "batch_label", "coarse_labels" and
        # "filenames".
        test_labels = np.array(test_batch["fine_labels".encode()])
        test_samples = test_batch["data".encode()]

        # Read test batch.
        with open(train_fn, "rb") as f:
            train_batch = pickle.load(f, encoding="bytes")

        train_labels = np.array(train_batch["fine_labels".encode()])
        train_samples = train_batch["data".encode()]

        if validation_size > 0:
            if validation_size >= train_labels.shape[0]:
                raise ValueError(
                    "Validation set must contain less than %d "
                    % (train_labels.shape[0])
                    + "samples!"
                )
            val_inds = np.arange(validation_size)
            train_inds = np.arange(validation_size, train_labels.size)

        else:
            train_inds = np.arange(train_labels.size)

        test_inds = np.arange(
            train_labels.size, train_labels.size + test_labels.size
        )

        labels = np.concatenate([train_labels, test_labels])
        labels = np.reshape(labels, (-1, 1))

        images = np.concatenate([train_samples, test_samples], axis=0)

        # Note, images are currently encoded in a way, that there shape
        # corresponds to (3, 32, 32). For consistency reasons, we would like to
        # change that to (32, 32, 3).
        images = np.reshape(images, (-1, 3, 32, 32))
        images = np.rollaxis(images, 1, 4)
        images = np.reshape(images, (-1, 32 * 32 * 3))
        # Scale images into a range between 0 and 1.
        images = images / 255

        self._data["in_data"] = images
        self._data["train_inds"] = train_inds
        self._data["test_inds"] = test_inds
        if validation_size > 0:
            self._data["val_inds"] = val_inds

        if self._data["is_one_hot"]:
            labels = self._to_one_hot(labels)

        self._data["out_data"] = labels

    def get_identifier(self):
        """Returns the name of the dataset."""
        return "CIFAR-100"

    def input_to_torch_tensor(
        self,
        x,
        device,
        mode="inference",
        force_no_preprocessing=False,
        sample_ids=None,
    ):
        """This method can be used to map the internal numpy arrays to PyTorch
        tensors.

        Note, this method has been overwritten from the base class.

        The input images are preprocessed if data augmentation is enabled.
        Preprocessing involves normalization and (for training mode) random
        perturbations.

        Args:
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        if self._augment_inputs and not force_no_preprocessing:
            if mode == "inference":
                transform = self._test_transform
            elif mode == "train":
                transform = self._train_transform
            else:
                raise ValueError(
                    '"%s" not a valid value for argument "mode".' % mode
                )

            return CIFAR10Data.torch_augment_images(x, device, transform)

        else:
            return Dataset.input_to_torch_tensor(
                self,
                x,
                device,
                mode=mode,
                force_no_preprocessing=force_no_preprocessing,
                sample_ids=sample_ids,
            )

    def _plot_sample(
        self,
        fig,
        inner_grid,
        num_inner_plots,
        ind,
        inputs,
        outputs=None,
        predictions=None,
    ):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("CIFAR-100 Sample")
        else:
            assert np.size(outputs) == 1
            label = np.asscalar(outputs)
            label_name = self._data["cifar100"]["fine_label_names"][label]

            if predictions is None:
                ax.set_title(
                    "Label of shown sample:\n%s (%d)" % (label_name, label)
                )
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                pred_label_name = self._data["cifar100"]["fine_label_names"][
                    pred_label
                ]

                ax.set_title(
                    "Label of shown sample:\n%s (%d)" % (label_name, label)
                    + "\nPrediction: %s (%d)" % (pred_label_name, pred_label)
                )

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(inputs, self.in_shape)))
        fig.add_subplot(ax)

        if num_inner_plots == 2:
            ax = plt.Subplot(fig, inner_grid[1])
            ax.set_title("Predictions")
            bars = ax.bar(range(self.num_classes), np.squeeze(predictions))
            ax.set_xticks(range(self.num_classes))
            if outputs is not None:
                bars[int(label)].set_color("r")
            fig.add_subplot(ax)

    def _plot_config(self, inputs, outputs=None, predictions=None):
        """Re-Implementation of method
        :meth:`data.dataset.Dataset._plot_config`.

        This method has been overriden to ensure, that there are 2 subplots,
        in case the predictions are given.
        """
        plot_configs = super()._plot_config(
            inputs, outputs=outputs, predictions=predictions
        )

        if (
            predictions is not None
            and np.shape(predictions)[1] == self.num_classes
        ):
            plot_configs["outer_hspace"] = 0.6
            plot_configs["inner_hspace"] = 0.4
            plot_configs["num_inner_rows"] = 2
            # plot_configs['num_inner_cols'] = 1
            plot_configs["num_inner_plots"] = 2

        return plot_configs

    def _torch_input_transforms(self):
        normalize = transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        )

        self._train_transform = transforms.Compose(
            [
                transforms.ToPILImage("RGB"),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                CIFAR10Policy(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self._test_transform = transforms.Compose(
            [
                transforms.ToPILImage("RGB"),
                transforms.ToTensor(),
                normalize,
            ]
        )


class SplitCIFAR100Data_FeCAM(CIFAR100Data):
    """An instance of the class shall represent a single SplitCIFAR-100 task.

    Args:
        data_path: Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size: The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        use_data_augmentation (optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor`
            (hence, **only available for PyTorch**).
            Note, we are using the same data augmentation pipeline as for
            CIFAR-10.
        use_cutout (bool): See docstring of class
            :class:`data.cifar10_data.CIFAR10Data`.
        labels: The labels that should be part of this task.
        full_out_dim: Choose the original CIFAR instead of the the new
            task output dimension. This option will affect the attributes
            :attr:`data.dataset.Dataset.num_classes` and
            :attr:`data.dataset.Dataset.out_shape`.
    """

    # Note, we build the validation set below!
    def __init__(
        self,
        data_path,
        use_one_hot=False,
        validation_size=1000,
        use_data_augmentation=False,
        use_cutout=False,
        labels=range(0, 10),
        full_out_dim=False,
    ):
        super().__init__(
            data_path,
            use_one_hot=use_one_hot,
            validation_size=0,
            use_data_augmentation=use_data_augmentation,
            use_cutout=use_cutout,
        )

        _split_cifar_100_fecam_object(
            self,
            data_path,
            use_one_hot,
            validation_size,
            use_data_augmentation,
            labels,
            full_out_dim,
        )

    def transform_outputs(self, outputs):
        """Transform the outputs from the 100D CIFAR100 dataset into proper
        labels based on the constructor argument ``labels``.

        See :meth:`data.special.split_mnist.SplitMNIST.transform_outputs` for
        more information.

        Args:
            outputs: 2D numpy array of outputs.

        Returns:
            2D numpy array of transformed outputs.
        """
        return _transform_split_outputs(self, outputs)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return "SplitCIFAR100"


def _split_cifar_100_fecam_object(
    data,
    data_path,
    use_one_hot,
    validation_size,
    use_data_augmentation,
    labels,
    full_out_dim,
):
    """Extract a subset of labels from a CIFAR-100 dataset.

    The constructors of classes :class:`SplitCIFAR10Data` and
    :class:`SplitCIFAR100Data_FeCAM` are essentially identical. Therefore, the code
    is realized in this function.

    Args:
        data: The data handler (which is a full CIFAR-100 dataset,
            which will be modified).
        (....): See docstring of class :class:`SplitCIFAR10Data`.
    """
    assert isinstance(data, SplitCIFAR100Data_FeCAM)
    data._full_out_dim = full_out_dim

    if isinstance(labels, range):
        labels = list(labels)
    assert (
        np.all(np.array(labels) >= 0)
        and np.all(np.array(labels) < data.num_classes)
        and len(labels) == len(np.unique(labels))
    )
    K = len(labels)

    data._labels = labels

    train_ins = data.get_train_inputs()
    test_ins = data.get_test_inputs()

    train_outs = data.get_train_outputs()
    test_outs = data.get_test_outputs()

    # Get labels.
    if data.is_one_hot:
        train_labels = data._to_one_hot(train_outs, reverse=True)
        test_labels = data._to_one_hot(test_outs, reverse=True)
    else:
        train_labels = train_outs
        test_labels = test_outs

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    train_mask = train_labels == labels[0]
    test_mask = test_labels == labels[0]
    for k in range(1, K):
        train_mask = np.logical_or(train_mask, train_labels == labels[k])
        test_mask = np.logical_or(test_mask, test_labels == labels[k])

    train_ins = train_ins[train_mask, :]
    test_ins = test_ins[test_mask, :]

    train_outs = train_outs[train_mask, :]
    test_outs = test_outs[test_mask, :]

    if validation_size > 0:
        if validation_size >= train_outs.shape[0]:
            raise ValueError(
                "Validation set must contain less than %d "
                % (train_outs.shape[0])
                + "samples!"
            )
        val_inds = np.arange(validation_size)
        train_inds = np.arange(validation_size, train_outs.shape[0])

    else:
        train_inds = np.arange(train_outs.shape[0])

    test_inds = np.arange(
        train_outs.shape[0], train_outs.shape[0] + test_outs.shape[0]
    )

    outputs = np.concatenate([train_outs, test_outs], axis=0)

    if not full_out_dim:
        # Note, the method assumes `full_out_dim` when later called by a
        # user. We just misuse the function to call it inside the
        # constructor.
        data._full_out_dim = True
        outputs = data.transform_outputs(outputs)
        data._full_out_dim = full_out_dim

        # Note, we may also have to adapt the output shape appropriately.
        if data.is_one_hot:
            data._data["out_shape"] = [len(labels)]

        data._data["cifar100"]["fine_label_names"] = [
            data._data["cifar100"]["fine_label_names"][ii] for ii in labels
        ]
        # FIXME I just set it to `None` as I don't know what to do with it
        # right now.
        data._data["cifar100"]["coarse_label_names"] = None

    images = np.concatenate([train_ins, test_ins], axis=0)

    ### Overwrite internal data structure. Only keep desired labels.

    # Note, we continue to pretend to be a 100 class problem, such that
    # the user has easy access to the correct labels and has the original
    # 1-hot encodings.
    if not full_out_dim:
        data._data["num_classes"] = len(labels)
    else:
        # Note, we continue to pretend to be a 10/100 class problem, such that
        # the user has easy access to the correct labels and has the
        # original 1-hot encodings.
        assert data._data["num_classes"] == 100
    data._data["in_data"] = images
    data._data["out_data"] = outputs
    data._data["train_inds"] = train_inds
    data._data["test_inds"] = test_inds
    if validation_size > 0:
        data._data["val_inds"] = val_inds

    n_val = 0
    if validation_size > 0:
        n_val = val_inds.size

    print(
        "Created SplitCIFAR-%d task with labels %s and %d train, %d test "
        % (100, str(labels), train_inds.size, test_inds.size)
        + "and %d val samples." % (n_val)
    )


########################################################
# Classes from FeCAM
########################################################
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC,
            fillcolor=self.fillcolor,
        )


class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC,
            fillcolor=self.fillcolor,
        )


class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor,
        )


class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor,
        )


class Rotate(object):
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(
            rot, Image.new("RGBA", rot.size, (128,) * 4), rot
        ).convert(x.mode)


class Color(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class Posterize(object):
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize(object):
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class Contrast(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class Sharpness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class Brightness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class AutoContrast(object):
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize(object):
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class Invert(object):
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)


class SubPolicy(object):
    def __init__(
        self,
        p1,
        operation1,
        magnitude_idx1,
        p2,
        operation2,
        magnitude_idx2,
        fillcolor=(128, 128, 128),
    ):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert(),
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class CIFAR10Policy(object):
    """Randomly choose one of the best 25 Sub-policies on CIFAR10.

    Example:
    >>> policy = CIFAR10Policy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     CIFAR10Policy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


if __name__ == "__main__":
    pass

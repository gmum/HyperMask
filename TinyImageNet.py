"""
Implementation of TinyImageNet for continual learning tasks.

Parts of the following code come from:
- https://github.com/DennisHanyuanXu/Tiny-ImageNet/blob/master/src/data_prep.py,
- https://github.com/pytorch/vision/issues/6127#issuecomment-1555049003
"""

import os
import numpy as np
import time
import glob
import torch
import urllib.request
from zipfile import ZipFile
from hypnettorch.data.dataset import Dataset
import torchvision.transforms as transforms
from skimage import io
from skimage.color import gray2rgb


class TinyImageNet(Dataset):
    _DOWNLOAD_PATH = "http://cs231n.stanford.edu/"
    _DOWNLOAD_FILE = "tiny-imagenet-200.zip"
    _EXTRACTED_FOLDER = "tiny-imagenet-200"

    def __init__(
        self,
        data_path,
        use_one_hot=False,
        use_data_augmentation=False,
        validation_size=250,
        seed=1,
        labels=[i for i in range(5)],
    ):
        super().__init__()
        self.data_path = data_path
        self._labels = labels
        self._use_one_hot = use_one_hot
        self._seed = seed
        self._validation_size = validation_size
        start = time.time()
        print("Reading TinyImageNet dataset...")

        if not os.path.exists(self.data_path):
            print(f"Creating directory: {self.data_path}")
            os.makedirs(self.data_path, exist_ok=True)

        self.extracted_data_dir = os.path.join(
            self.data_path, TinyImageNet._EXTRACTED_FOLDER
        )
        self._data = dict()
        self._data["tinyimagenet"] = dict()
        # If data has been processed before
        build_from_scratch = True
        if os.path.exists(self.extracted_data_dir):
            build_from_scratch = False

        if build_from_scratch:
            archive_fn = os.path.join(
                self.data_path, TinyImageNet._DOWNLOAD_FILE
            )
            print(archive_fn)

            if not os.path.exists(self.extracted_data_dir):
                print(f"Extracted data dir: {self.extracted_data_dir}")
                print("Downloading dataset...")
                urllib.request.urlretrieve(
                    (
                        f"{TinyImageNet._DOWNLOAD_PATH}"
                        f"{TinyImageNet._DOWNLOAD_FILE}"
                    ),
                    archive_fn,
                )
                zf = ZipFile(archive_fn, "r")
                zf.extractall(path=self.data_path)
                zf.close()
                os.remove(archive_fn)

        self._data["classification"] = True
        self._data["sequence"] = False
        self._data["num_classes"] = 5
        self._data["is_one_hot"] = use_one_hot
        self._data["in_shape"] = [64, 64, 3]
        self._data["out_shape"] = [5 if use_one_hot else 1]

        # Prepare IDs of consecutive classes
        self.ids = {}
        for i, line in enumerate(
            open(f"{self.data_path}/tiny-imagenet-200/wnids.txt", "r")
        ):
            self.ids[line.replace("\n", "")] = i

        # Transform labels to the shape proper for neural networks
        self._translate_labels()
        (
            self.train_data,
            self.train_labels,
        ) = self.prepare_training_test_set_with_labels(mode="train")
        (
            self.test_data,
            self.test_labels,
        ) = self.prepare_training_test_set_with_labels(mode="test")
        if self._use_one_hot:
            self.train_labels = self._to_one_hot(
                self.train_labels, reverse=False
            )
            self.test_labels = self._to_one_hot(self.test_labels, reverse=False)

        # Set names of consecutive classes
        self._read_label_names()
        # Standardization should be performed both in the case of augmented
        # and non-augmented datasets
        (
            self.train_transform,
            self.test_transform,
        ) = self.torch_input_transforms()
        if not use_data_augmentation:
            self.train_transform = self.test_transform
        # Prepare training, validation and test sets
        self._prepare_train_val_test_set()
        # Check whether the dataset was initialized properly
        self._validity_control()
        end = time.time()
        print(f"Elapsed time to read dataset: {end-start} sec.")

    def torch_input_transforms(self):
        """
        Prepare data standarization, and potentially also augmentation,
        for TinyImageNet images.

        Data augmentation is implemented as in
        https://github.com/ihaeyong/WSN/blob/main/dataloader/idataset.py.

        Returns:
        --------
        A tuple containing **train_transform** that applies standarization
        and random image transformations and **test_transform** that applies
        only data standarization.
        """
        test_transform = transforms.Compose(
            [
                transforms.ToPILImage("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        train_transform = transforms.Compose(
            [
                transforms.ToPILImage("RGB"),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        return train_transform, test_transform

    def plot_sample(self):
        pass

    def get_identifier(self):
        return "TinyImageNet"

    def _plot_sample(self):
        pass

    def input_to_torch_tensor(
        self,
        x,
        device,
        mode="inference",
        force_no_preprocessing=False,
        sample_ids=None,
    ):
        """
        Prepare mapping of Numpy arrays to PyTorch tensors.
        This method overwrites the method from the base class.
        The input data are preprocessed (data standarization).

        Arguments:
        ----------
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        if not force_no_preprocessing:
            if mode == "inference":
                transform = self.test_transform
            elif mode == "train":
                transform = self.train_transform
            else:
                raise ValueError(
                    f"{mode} is not a valid value for the" "argument 'mode'."
                )
            return TinyImageNet.torch_preprocess_images(x, device, transform)

        else:
            return Dataset.input_to_torch_tensor(
                self,
                x,
                device,
                mode=mode,
                force_no_preprocessing=force_no_preprocessing,
                sample_ids=sample_ids,
            )

    @staticmethod
    def torch_preprocess_images(x, device, transform, img_shape=[64, 64, 3]):
        """
        Prepare preprocessing of TinyImageNet images with a selected
        PyTorch transformation.

        Arguments:
        ----------
            x (Numpy array): 2D array containing TinyImageNet images.
            device (torch.device or int): PyTorch device on which a final
                                          tensor will be moved
            transform: (torchvision.transforms): a method of data modification

        Returns:
        --------
            (torch.Tensor): The preprocessed images as PyTorch tensor.
        """
        assert len(x.shape) == 2
        # First dimension is related to batch size and second is related
        # to the flattened image.
        x = (x * 255.0).astype("uint8")
        x = x.reshape(-1, *img_shape)
        x = torch.stack([transform(x[i, ...]) for i in range(x.shape[0])]).to(
            device
        )
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, np.prod(img_shape))
        return x

    def prepare_training_test_set_with_labels(self, mode="train"):
        """
        Function implemented on the basis of:
        https://github.com/pytorch/vision/issues/6127#issuecomment-1555049003

        Arguments:
        ----------
           *mode* (optional string) 'train' for the training set or 'test'
                  for the validation set
        """
        assert mode in ["train", "test"]
        data, labels = [], []
        if mode == "train":
            filenames = glob.glob(
                f"{self.data_path}/tiny-imagenet-200/train/*/*/*.JPEG"
            )
        elif mode == "test":
            id_dict_test = {}
            for i, line in enumerate(
                open(
                    f"{self.data_path}/tiny-imagenet-200/val/val_annotations.txt",
                    "r",
                )
            ):
                current_info = line.split("\t")
                img, id = current_info[0], current_info[1]
                id_dict_test[img] = self.ids[id]
            filenames = glob.glob(
                f"{self.data_path}/tiny-imagenet-200/val/images/*.JPEG"
            )
        for file in filenames:
            image = io.imread(file)
            if len(image.shape) == 2:  # gray-scale
                image = gray2rgb(image)
            image = image / 255
            if mode == "train":
                label = self.ids[file.split("/")[-3]]
            elif mode == "test":
                label = id_dict_test[file.split("/")[-1]]
            # Add to the image only cases which are from the desired class
            if label in self._labels:
                label = self.translate_real_label_to_temp_label[label]
                image = image.reshape(np.prod(image.shape))
                data.append(image)
                labels.append([label])
        data = np.vstack(data)
        labels = np.array(labels)
        # 200 classes in TinyImageNet
        assert np.min(np.squeeze(labels)) == np.min(
            list(self.translate_temp_label_to_real_labels.keys())
        )
        assert np.max(np.squeeze(labels)) == np.max(
            list(self.translate_temp_label_to_real_labels.keys())
        )
        return data, labels

    def _read_label_names(self):
        """
        Function implemented on the basis of:
        https://github.com/DennisHanyuanXu/Tiny-ImageNet/blob/master/src/data_prep.py

        Set a dictionary with names of consecutive classes of TinyImageNet.
        """
        class_names = dict()
        loaded_file = open(
            os.path.join(self.extracted_data_dir, "words.txt"), "r"
        )
        names = loaded_file.readlines()
        for current_class in names:
            line = current_class.strip("\n").split("\t")
            if line[0] in self.ids.keys():
                label_ID = self.ids[line[0]]
                class_names[label_ID] = line[1].split(",")[0]
        loaded_file.close()
        self._data["tinyimagenet"]["label_names"] = class_names

    def _prepare_train_val_test_set(self):
        """
        Prepares a stratified selection of the training and validation set.
        Also, prepares a final version of the test set and filles keys
        necessary for further calculations.
        """
        if self._validation_size > 0:
            no_of_classes = len(self._labels)
            if self._validation_size < 1:
                # We assume that the number of samples from each class
                # is exactly the same
                no_of_samples = self.train_data.shape[0] / 5
                self._no_of_val_samples = int(
                    no_of_classes * self._validation_size * no_of_samples
                )
            else:
                self._no_of_val_samples = self._validation_size

            (
                self._data["train_inds"],
                self._data["val_inds"],
            ) = self._select_val_indices(no_of_classes)

        else:
            self.train_labels = self.train_labels.squeeze()
            self._data["train_inds"] = np.arange(0, self.train_labels.shape[0])

        self._data["test_inds"] = np.arange(
            self.train_labels.shape[0],
            self.train_labels.shape[0] + self.test_labels.shape[0],
        )

        self._data["in_data"] = np.concatenate(
            [self.train_data, self.test_data]
        )
        del self.train_data
        del self.test_data

        if not self._use_one_hot:
            self.train_labels = np.expand_dims(self.train_labels, axis=1)
        self._data["out_data"] = np.concatenate(
            [self.train_labels, self.test_labels]
        )
        del self.train_labels
        del self.test_labels

    def _select_val_indices(self, no_of_classes):
        """
        Prepare a selection of train and validation sets with memory saving!
        TinyImageNet is a large dataset, therefore solutions need to be
        memory-efficient.

        Args:
        -----
          *no_of_classes*: (int) number of classes in the dataset

        Returns:
        --------
          *train_indices*: (list) contains indices of elements in the training
                           set
          *test_indices*: (list) contains indices of elements in the test set
        """
        # 40 is the number of tasks
        self._no_of_val_samples_per_class = int(
            self._no_of_val_samples / no_of_classes
        )
        if not self._use_one_hot:
            self.train_labels = self.train_labels.squeeze()
            train_labels_for_val_separation = self.train_labels
        else:
            train_labels_for_val_separation = self._to_one_hot(
                self.train_labels, reverse=True
            ).squeeze()
        unique_classes = np.unique(train_labels_for_val_separation)
        class_positions = {}
        for no_of_class in unique_classes:
            class_positions[no_of_class] = np.argwhere(
                train_labels_for_val_separation == no_of_class
            )

        np.random.seed(self._seed)
        train_indices, val_indices = [], []
        for cur_class in list(class_positions.keys()):
            perm = np.random.permutation(class_positions[cur_class])
            cur_class_val_indices = perm[: self._no_of_val_samples_per_class]
            cur_class_train_indices = perm[self._no_of_val_samples_per_class :]
            train_indices.extend(list(cur_class_train_indices.flatten()))
            val_indices.extend(list(cur_class_val_indices.flatten()))
        return np.array(train_indices), np.array(val_indices)

    def _validity_control(self):
        """
        Control whether the set was prepared according to the desired hyperparams.
        """
        # Test set: 5 classes, 50 samples per class
        assert self._data["test_inds"].shape[0] == 250
        # Total number of samples: 250 in test set (50 * 5)
        # and 500 * 5 in the concatenated training/validation sets.
        if not self._use_one_hot:
            labels_squeezed = self._data["out_data"].squeeze()
        else:
            labels_squeezed = self._to_one_hot(
                self._data["out_data"], reverse=True
            ).squeeze()
        assert labels_squeezed.shape[0] == 2750
        assert self._data["in_data"].shape[0] == 2750
        assert self._data["in_data"].shape[1] == 12288
        # 64 * 64 * 3 = 12288
        # 2250 examples
        # Control test set
        test_labels = labels_squeezed[self._data["test_inds"]]
        temporary_labels = list(self.translate_temp_label_to_real_labels.keys())
        for label in temporary_labels:
            assert np.count_nonzero(test_labels == label) == 50
        # Control validation set
        if self._validation_size > 0:
            assert self._data["val_inds"].shape[0] == self._no_of_val_samples
            val_labels = labels_squeezed[self._data["val_inds"]]
            for label in temporary_labels:
                assert (
                    np.count_nonzero(val_labels == label)
                    == self._no_of_val_samples_per_class
                )
        # Control train set
        train_labels = labels_squeezed[self._data["train_inds"]]
        no_of_train_samples_per_class = train_labels.shape[0] / 5
        for label in temporary_labels:
            assert (
                np.count_nonzero(train_labels == label)
                == no_of_train_samples_per_class
            )

    def _translate_labels(self):
        """
        Due to the fact that TinyImageNet has 200 classes and a subset of classes
        may be scattered across the space of labels and neural networks may be
        less performing for a high-dimensional output vector we have to reduce the
        output size and prepare labels in the form {0, 1, 2, ..., number of classes}.
        However, the inverse process should be possible, i.e. to which original class
        corresponds to current class number.
        """
        sorted_labels = np.sort(self._labels)
        self.translate_temp_label_to_real_labels = dict()
        self.translate_real_label_to_temp_label = dict()
        for i in range(sorted_labels.shape[0]):
            self.translate_temp_label_to_real_labels[i] = sorted_labels[i]
            self.translate_real_label_to_temp_label[sorted_labels[i]] = i

    def translate_temporary_to_real_label(self):
        return self.translate_temp_label_to_real_labels

    def translate_real_to_temporary_label(self):
        return self.translate_real_label_to_temp_label


if __name__ == "__main__":
    tinyimagenet = TinyImageNet(
        data_path="./Data",
        validation_size=250,
        use_one_hot=True,
        labels=[1, 5, 13, 21, 36],
    )

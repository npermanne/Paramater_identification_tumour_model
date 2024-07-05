# Packages
import numpy as np
import pandas as pd
import os
import re
from skimage.metrics import structural_similarity
import math
from enum import Enum
from multiprocessing import Pool
from metrics import Metric

PARAMETERS = [
    "sources",
    "average_healthy_glucose_absorption",
    "average_cancer_glucose_absorption",
    "average_healthy_oxygen_consumption",
    "average_cancer_oxygen_consumption",
    "quiescent_multiplier",
    "critical_multiplier",
    "cell_cycle",
    "radiosensitivity_G1",
    "radiosensitivity_S",
    "radiosensitivity_G2",
    "radiosensitivity_M",
    "radiosensitivity_G0",
    "source_glucose_supply",
    "source_oxygen_supply",
    "glucose_diffuse_rate",
    "oxygen_diffuse_rate",
    "h_cells"
]

DATA_NAME = "image{}_type={}_time={}.npy"


# Allow to crop an image
def crop(image, percentage=0.5):
    destination_size = len(image) * percentage
    start = int(len(image) / 2 - destination_size / 2)
    end = int(len(image) / 2 + destination_size / 2)
    return image[start:end, start:end]


# Find all pair of value that have a specific difference in an array
def find_value_pairs(a: np.array, difference, tol):
    abs_diff_matrix = np.abs(a[:, np.newaxis] - a)
    mask = np.abs(abs_diff_matrix - difference) <= tol
    indices = np.argwhere(mask)
    indices = indices[indices[:, 0] <= indices[:, 1]]
    return a[indices]


class Comparator:
    """
    A class filed with datasets that allows easy iteration and comparison between sample

    Parameters:
        datasets_name (list): List of names of datasets contained in this comparator

    Attributes:
        general_dataset (dataframe): Pandas dataframe containing all the sample from all the dataset

    Methods:
        __len__(): Returns the number of samples in this comparator
        __getitem__(item): Return a specific sample and his parameter
        get_all_value(param, value): Get all value that respect param == value
        get_all_indexes(param, value): Get all indexes that respect param == value
        get_possible_values(param): Get all possible values that param can take
        compare(compare_function, i1, i2, crop_percentage=0.5): Compare two sample at index i1 and i2 with a specific function
        diff(i1, i2, crop_percentage=0.5): Returns the absolute difference between sample i1 and i2
        corr_hist(i1, i2, bins=100, crop_percentage=0.5): Returns the histogram correlation between sample i1 and i2
        ssim(i1, i2, crop_percentage=0.5): Returns the structural similarity index(SSIM) between sample i1 and i2
        mean_absolute_error(i1, i2, crop_percentage=0.5): Returns the mean absolute difference between sample i1 and i2
        root_mean_squared_error(i1, i2, crop_percentage=0.5): Returns the root mean squared difference between sample i1 and i2
        max_absolute_error(i1, i2, crop_percentage=0.5): Returns the max absolute difference between sample i1 and i2
    """

    def __init__(self, *datasets_name, jupyter=False):
        self.general_dataset = None
        self.jupyter = jupyter

        for dataset_name in datasets_name:
            if jupyter:
                path = os.path.join("..", "datasets", dataset_name, "dataset.csv")
            else:
                path = os.path.join("datasets", dataset_name, "dataset.csv")
            df = pd.read_csv(path, index_col=0)
            df.insert(df.shape[1], "origin_folder", [dataset_name for _ in range(len(df))], False)
            df.insert(df.shape[1], "sample_index", range(len(df)), False)

            if self.general_dataset is None:
                self.general_dataset = df
            else:
                self.general_dataset = pd.concat([self.general_dataset, df], ignore_index=True)

    def __len__(self):
        return len(self.general_dataset)

    def get_item(self, idx, draw: int, image_type: str):
        original_folder = self.general_dataset.iloc[idx]["origin_folder"]
        sample_index = self.general_dataset.iloc[idx]["sample_index"]

        if self.jupyter:
            path = os.path.join("..", "datasets", original_folder, DATA_NAME.format(sample_index, image_type, draw))
        else:
            path = os.path.join("datasets", original_folder, DATA_NAME.format(sample_index, image_type, draw))
        ret = np.load(path)

        param = self.general_dataset.iloc[idx].to_dict()
        del param["origin_folder"]
        del param["sample_index"]

        return ret, param

    def get_all_value(self, param, value, draw: int, image_type: str):
        indexes = self.general_dataset.index[self.general_dataset[param] == value].tolist()
        return np.array([self.get_item(i, draw, image_type)[0] for i in indexes])

    def get_all_indexes(self, param, value):
        return np.array(self.general_dataset.index[self.general_dataset[param] == value].tolist())

    def get_possible_values(self, param):
        return np.array(list(set(self.general_dataset[param])))

    def compare(self, compare_function, i1, i2, draw: int, image_type: str, crop_percentage=0.5):
        image1 = crop(self.get_item(i1, draw, image_type)[0], percentage=crop_percentage)
        image2 = crop(self.get_item(i2, draw, image_type)[0], percentage=crop_percentage)
        return compare_function(image1, image2)

    def mean_and_std_difference(self, metric: Metric, draw: int, image_type: str, parameter: str, difference: float, tol: float, process_number: int, iteration: int, crop_percentage=0.5):
        all_possible_values = self.get_possible_values(parameter)
        different_pairs = find_value_pairs(all_possible_values, difference, tol)
        all_indexes_pairs = None
        for i, different_pair in enumerate(different_pairs):
            indexes1 = self.get_all_indexes(parameter, different_pair[0])
            indexes2 = self.get_all_indexes(parameter, different_pair[1])
            v1, v2 = np.meshgrid(indexes1, indexes2)
            pairs = np.stack((v1.flatten(), v2.flatten()), axis=-1)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            pairs = np.sort(pairs, axis=1)
            pairs = np.unique(pairs, axis=0)
            all_indexes_pairs = pairs if i == 0 else np.concatenate([all_indexes_pairs, pairs])

        random_indices = np.random.choice(len(all_indexes_pairs), size=iteration, replace=True)
        all_indexes_pairs = all_indexes_pairs[random_indices]
        global function

        def function(a):
            return self.compare(metric.get_function(), a[0], a[1], draw, image_type, crop_percentage=crop_percentage)

        results = None
        with Pool(processes=process_number) as pool:
            results = pool.map(function, all_indexes_pairs)

        return np.mean(results, axis=0), np.std(results, axis=0)


if __name__ == '__main__':
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(find_value_pairs(a, 0, 0))

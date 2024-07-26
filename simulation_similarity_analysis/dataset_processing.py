# Packages
import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import Isomap
import re
from skimage.metrics import structural_similarity
import math
from enum import Enum
from multiprocessing import Pool
from sklearn.decomposition import PCA
from metrics import SimilarityMetric

# Parameters
DATASETS_FOLDER = "datasets"
DATASET_CSV_NAME = "dataset.csv"
DATA_NAME = "image{}_type={}_time={}.npy"
SIMULATION_SIMILARITY_FOLDER = "simulation_similarity_analysis"

IMG_TYPES = ["cells_types", "cells_densities", "oxygen", "glucose"]
TIMESTEPS = range(350, 1150, 100)


# Find all pair of value that have a specific difference in an array
def find_value_pairs(a: np.array, difference, tol):
    abs_diff_matrix = np.abs(a[:, np.newaxis] - a)
    mask = np.abs(abs_diff_matrix - difference) <= tol
    indices = np.argwhere(mask)
    indices = indices[indices[:, 0] <= indices[:, 1]]
    return a[indices]


class DatasetProcessing:
    """
        A class storing a dataset and allowing a vast amount of operation on it

        Parameters:
            dataset_name (str): The dataset to analyse

        Attributes:
            self.dataset_name(str): The name of the dataset
            self.dataset_path (str): The path to the dataset
            self.dataset (Dataframe): A pandas dataframe containing the dataset
        Methods:
            __len__(): Returns the number of samples in this comparator
            __getitem__(item): Return a specific sample and his parameter
        """

    def __init__(self, dataset_name: str, jupyter=False):
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(DATASETS_FOLDER, self.dataset_name)
        if jupyter:
            self.dataset_path = os.path.join("..", self.dataset_path)
        self.dataset = pd.read_csv(os.path.join(self.dataset_path, DATASET_CSV_NAME), index_col=0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, sample_idx):
        sample_matrices_dico = dict()
        for img_type in IMG_TYPES:
            sample_matrices_dico[img_type] = dict()
            for timestep in TIMESTEPS:
                path_to_sample = DATA_NAME.format(sample_idx, img_type, timestep)
                sample_matrices_dico[img_type][timestep] = np.load(os.path.join(self.dataset_path, path_to_sample))

        sample_matrices_array = np.array([[np.load(os.path.join(self.dataset_path, DATA_NAME.format(sample_idx, img_type, timestep))) for timestep in TIMESTEPS] for img_type in IMG_TYPES])
        return sample_matrices_dico, sample_matrices_array

    def get_sample_matrix(self, sample_idx, img_type, timestep):
        path_to_sample = DATA_NAME.format(sample_idx, img_type, timestep)
        return np.load(os.path.join(self.dataset_path, path_to_sample))

    def get_sample_param(self, sample_idx):
        return self.dataset.iloc[sample_idx].to_dict()

    def get_possible_param_values(self, param):
        return np.array(list(set(self.dataset[param])))

    def get_all_indexes(self, param, value):
        return np.array(self.dataset.index[self.dataset[param] == value].tolist())

    def get_sample_matrix(self, sample_idx, timestep, img_type):
        path_to_sample_matrix = DATA_NAME.format(sample_idx, img_type, timestep)
        sample_matrix = np.load(os.path.join(self.dataset_path, path_to_sample_matrix))
        return sample_matrix

    def pca_per_matrix(self, timestep, img_type):
        os.makedirs(os.path.join(SIMULATION_SIMILARITY_FOLDER, "pca"), exist_ok=True)
        data = np.array([self.get_sample_matrix(i, timestep, img_type).flatten() for i in range(self.__len__())])
        pca = PCA(n_components=2)
        reduction = pca.fit_transform(data)
        explained_variance = pca.explained_variance_ratio_.sum()
        path = os.path.join(SIMULATION_SIMILARITY_FOLDER, "pca", f"pca_{img_type}_{timestep}_vr={explained_variance:.2f}.npy")
        np.save(path, reduction)

    def pca_combined(self):
        os.makedirs(os.path.join(SIMULATION_SIMILARITY_FOLDER, "pca"), exist_ok=True)
        data = np.array([self[i][1].flatten() for i in range(len(self))])
        pca = PCA(n_components=2)
        reduction = pca.fit_transform(data)
        explained_variance = pca.explained_variance_ratio_.sum()
        path = os.path.join(SIMULATION_SIMILARITY_FOLDER, "pca", f"pca_combined_vr={explained_variance:.4f}.npy")
        np.save(path, reduction)

    def isomap_per_matrix(self, timestep, img_type):
        os.makedirs(os.path.join(SIMULATION_SIMILARITY_FOLDER, "isomap"), exist_ok=True)
        data = np.array([self.get_sample_matrix(i, timestep, img_type).flatten() for i in range(self.__len__())])
        isomap = Isomap(n_components=2, n_neighbors=15)
        reduction = isomap.fit_transform(data)
        path = os.path.join(SIMULATION_SIMILARITY_FOLDER, "isomap", f"isomap_{img_type}_{timestep}.npy")
        np.save(path, reduction)

    def isomap_combined(self):
        os.makedirs(os.path.join(SIMULATION_SIMILARITY_FOLDER, "isomap"), exist_ok=True)
        data = np.array([self[i][1].flatten() for i in range(len(self))])
        isomap = Isomap(n_components=2, n_neighbors=15)
        reduction = isomap.fit_transform(data)
        path = os.path.join(SIMULATION_SIMILARITY_FOLDER, "isomap", f"isomap_combined.npy")
        np.save(path, reduction)

    def mutual_information(self, timestep, img_type, parameter):
        os.makedirs(os.path.join(SIMULATION_SIMILARITY_FOLDER, "mi"), exist_ok=True)
        data_matrices = [self.get_sample_matrix(i, timestep, img_type).flatten() for i in range(self.__len__())]
        n = int(np.sqrt(len(data_matrices[0])))
        data_target = np.array([self.get_sample_param(i)[parameter] for i in range(len(self))])
        mutual_information = mutual_info_regression(data_matrices, data_target, discrete_features=(img_type == "cells_types")).reshape(n, n)
        path = os.path.join(SIMULATION_SIMILARITY_FOLDER, "mi", f"mi_{timestep}_{img_type}_{parameter}.npy")
        np.save(path, mutual_information)

    def similarity_between_matrix_per_difference(self, metric: SimilarityMetric, timestep: int, img_type: str, parameter: str, difference: float, tol: float, process_number: int, iteration: int):
        possible_param_values = self.get_possible_param_values(parameter)
        pairs_different_param = find_value_pairs(possible_param_values, difference, tol)
        all_indexes_pairs = None
        for i, different_pair in enumerate(pairs_different_param):
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
            return metric.get_function()(self.get_sample_matrix(a[0], timestep, img_type), self.get_sample_matrix(a[1], timestep, img_type))

        results = None
        with Pool(processes=process_number) as pool:
            results = pool.map(function, all_indexes_pairs)

        os.makedirs(os.path.join(SIMULATION_SIMILARITY_FOLDER, "similarity_between_matrix"), exist_ok=True)
        path_mean = os.path.join(SIMULATION_SIMILARITY_FOLDER, "similarity_between_matrix", f"mean_{metric.__str__()}_{timestep}_{img_type}_{parameter}_{difference}.npy")
        path_std = os.path.join(SIMULATION_SIMILARITY_FOLDER, "similarity_between_matrix", f"std_{metric.__str__()}_{timestep}_{img_type}_{parameter}_{difference}.npy")
        np.load(path_mean, np.mean(results, axis=0))
        np.load(path_std, np.std(results, axis=0))


if __name__ == "__main__":
    dataset_processing = DatasetProcessing("full_no_dose_dataset_start=350_interval=100_ndraw=8_size=(64,64)")


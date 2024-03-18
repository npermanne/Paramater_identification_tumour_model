# Packages
import numpy as np
import pandas as pd
import os
import re
from skimage.metrics import structural_similarity
import math

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


#
# Class filed with datasets that allows easy iteration and comparison between sample
#


# Allow to crop an image
def crop(image, percentage=0.5):
    destination_size = len(image) * percentage
    start = int(len(image) / 2 - destination_size / 2)
    end = int(len(image) / 2 + destination_size / 2)
    return image[start:end, start:end]


class Comparator:
    def __init__(self, *datasets_name):
        self.general_dataset = None

        for dataset_name in datasets_name:
            df = pd.read_csv(os.path.join("..", "datasets", dataset_name, "dataset.csv"), index_col=0)
            df.insert(df.shape[1], "origin_folder", [dataset_name for _ in range(len(df))], False)
            df.insert(df.shape[1], "sample_index", range(len(df)), False)

            if self.general_dataset is None:
                self.general_dataset = df
            else:
                self.general_dataset = pd.concat([self.general_dataset, df], ignore_index=True)

    def __len__(self):
        return len(self.general_dataset)

    def __getitem__(self, item):
        original_folder = self.general_dataset.iloc[item]["origin_folder"]
        sample_index = self.general_dataset.iloc[item]["sample_index"]

        s, i, d, h, w = re.findall(r'\d+', original_folder)
        s, d, i = int(s), int(d), int(i)

        ret = {}
        for draw in range(d):
            ret[s + draw * i] = {}
            for type in ['cells_types', 'cells_densities', 'oxygen', 'glucose']:
                ret[s + draw * i][type] = np.load(
                    os.path.join("..", "datasets", original_folder, DATA_NAME.format(sample_index, type, s + draw * i)))

        param = self.general_dataset.iloc[item].to_dict()
        del param["origin_folder"]
        del param["sample_index"]

        return ret, param

    # Get all value that respect param == value
    def get_all_value(self, param, value):
        indexes = self.general_dataset.index[self.general_dataset[param] == value].tolist()
        return [self.__getitem__(i)[0] for i in indexes]

    # Get all indexes that respect param == value
    def get_all_indexes(self, param, value):
        return self.general_dataset.index[self.general_dataset[param] == value].tolist()

    # Get possibles value of a parameter:
    def get_possible_values(self, param):
        return np.array(list(set(self.general_dataset[param])))

    # Apply function to 2 images (compare them)
    def compare(self, compare_function, i1, i2, crop_percentage=0.5):
        images1 = self.__getitem__(i1)[0]
        images2 = self.__getitem__(i2)[0]

        ret = {}
        for draw in images1.keys():
            ret[draw] = {}
            for image_type in images1[draw].keys():
                image1 = crop(images1[draw][image_type], percentage=crop_percentage)
                image2 = crop(images2[draw][image_type], percentage=crop_percentage)
                ret[draw][image_type] = compare_function(image1, image2)

        return ret

    # Difference between 2 images
    def diff(self, i1, i2, crop_percentage=0.5):
        return self.compare(lambda a, b: np.absolute(a - b), i1, i2, crop_percentage=crop_percentage)

    # Histogram correlation between 2 images
    def corr_hist(self, i1, i2, bins=100, crop_percentage=0.5):
        def corr_hist_function(image1, image2):
            min_value, max_value = min(np.min(image1), np.min(image2)), max(np.max(image1), np.max(image2))
            hist1, bins1 = np.histogram(image1, bins=np.linspace(min_value, max_value + 1, bins))
            hist2, bins2 = np.histogram(image2, bins=np.linspace(min_value, max_value + 1, bins))
            return np.corrcoef(hist1, hist2)[0, 1]

        return self.compare(corr_hist_function, i1, i2, crop_percentage=crop_percentage)

    # Structural Similarity index (SSIM) between 2 images
    def ssim(self, i1, i2, crop_percentage=0.5):
        def ssim_function(image1, image2):
            min_value, max_value = min(np.min(image1), np.min(image2)), max(np.max(image1), np.max(image2))
            return structural_similarity(image1, image2, full=True, data_range=max_value - min_value)[0]

        return self.compare(ssim_function, i1, i2, crop_percentage=crop_percentage)

    # Mean absolute error
    def mean_absolute_error(self, i1, i2, crop_percentage=0.5):
        return self.compare(lambda a, b: np.abs(a - b).mean(), i1, i2, crop_percentage=crop_percentage)

    # Root mean squared error
    def root_mean_squared_error(self, i1, i2, crop_percentage=0.5):
        return self.compare(lambda a, b: math.sqrt(np.square(a - b).mean()), i1, i2, crop_percentage=crop_percentage)

    # Max absolute error
    def max_absolute_error(self, i1, i2, crop_percentage=0.5):
        return self.compare(lambda a, b: np.max(np.absolute(a - b)), i1, i2, crop_percentage=crop_percentage)


def value_that_diff(array, target_diff):
    tab = []
    i, j = 0, 0
    while i != len(array) and j != len(array):
        if array[j] - array[i] == target_diff:
            tab.append((array[i], array[j]))
            i += 1
        elif array[j] - array[i] < target_diff:
            j += 1
        else:
            i += 1

    return tab

if __name__ == '__main__':
    comparator = Comparator(
        "same_value_study_start=350_interval=100_ndraw=8_size=(64,64)",
        "cell_cycle_study_start=350_interval=100_ndraw=8_size=(64,64)")

    print(comparator.get_possible_values("cell_cycle"))

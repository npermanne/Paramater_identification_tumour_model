from torch.utils.data import Dataset
import pandas as pd
import re
import os
import numpy as np

# PATH (should be the same that in model/dataset_generator.py
DATASET_FOLDER_PATH = "datasets"
DATASET_FOLDER_NAME = "{}_start={}_interval={}_ndraw={}_size=({},{})"
DATASET_FILE_NAME = "dataset.csv"
PARAMETER_DATA = os.path.join("simulation", "parameter_data.csv")
DATA_NAME = "image{}_type={}_time={}.npy"
CELLS_TYPES = "cells_types"
CELLS_DENSITIES = "cells_densities"
OXYGEN = "oxygen"
GLUCOSE = "glucose"

# TRAIN, VALIDATION, TEST SET PROPORTION (SHOULD ADD UP TO ONE)
TRAIN_PROPORTION = 0.64
VALIDATION_PROPORTION = 0.16
TEST_PROPORTION = 0.2

# NORMALISATION PARAMETER
LOW_VALUE = 0
HIGH_VALUE = 1


class SimulationDataset(Dataset):
    """
    Custom dataset loader that inherits from torch.utils.data.Dataset

    Parameters:
        datasetType (string): Type of dataset (train, val or test)
        folderName (string): Name of the folder where the dataset is stored
        n_draw (int): Number of draws
        parameters_of_interest (list): List of parameters of interest that we want to predict
        img_types (list): List containing all the image types that we want to use

    Attributes:
        img_types (list): List containing all the image types that we want to use
        n_draw (int): Number of draws
        parameters_of_interest (list): List of parameters of interest that we want to predict
        height (int): height of the image
        width (int): width of the images
        sequence_times (list): List of times corresponding to the different images
        general_path (string): Path to the dataset
        dataframe (Dataframe): Dataframe containing all the sample and their respective parameters
        parameter_data (Dataframe): Dataframe containing the parameter and their range

    Methods:
        __len__(): Return the number of sample in this dataset
        get_height(): Return the height of the image in the sample
        get_width(): Return the width of the image in the sample
        __getitem__(idx): Return the input, the output and the output scaled of the model
    """

    def __init__(self, datasetType, folderName, n_draw, parameters_of_interest, img_types=None):
        if img_types is None:
            img_types = [CELLS_TYPES, CELLS_DENSITIES, OXYGEN, GLUCOSE]
        self.img_types = img_types
        self.n_draw = n_draw
        self.parameters_of_interest = parameters_of_interest
        s, i, d, h, w = re.findall(r'\d+', folderName)
        self.height = int(h)
        self.width = int(w)
        self.sequence_times = list(range(int(s), int(s) + int(d) * int(i), int(i)))[:n_draw]
        self.general_path = os.path.join(DATASET_FOLDER_PATH, folderName)
        self.dataframe = pd.read_csv(os.path.join(self.general_path, DATASET_FILE_NAME))
        self.parameter_data = pd.read_csv(PARAMETER_DATA, index_col=0)

        if datasetType == "train":
            self.start = 0  # Inclusive
            self.end = int(self.dataframe.shape[0] * 0.64)  # Exclusive
        elif datasetType == "val":
            self.start = int(self.dataframe.shape[0] * 0.64)  # Inclusive
            self.end = int(self.dataframe.shape[0] * (0.64 + 0.16))  # Exclusive
        elif datasetType == "test":
            self.start = int(self.dataframe.shape[0] * (0.64 + 0.16))  # Inclusive
            self.end = self.dataframe.shape[0]  # Exclusive
        else:
            raise Exception("Not a valid datasetType")

    def __len__(self):
        return self.end - self.start

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def __getitem__(self, idx):
        idx = self.start + idx
        inputs = np.zeros((self.n_draw, len(self.img_types), self.height, self.width), dtype=np.float32)
        for draw in range(self.n_draw):
            for index_type in range(len(self.img_types)):
                inputs[draw][index_type] = np.load(
                    os.path.join(self.general_path,
                                 DATA_NAME.format(idx, self.img_types[index_type], self.sequence_times[draw])))

        outputs = np.zeros((len(self.parameters_of_interest)), dtype=np.float32)
        outputsScaled = np.zeros((len(self.parameters_of_interest)), dtype=np.float32)
        i = 0
        for parameter in self.parameters_of_interest:
            min_parameter = self.parameter_data.at[parameter, "Minimum"]
            max_parameter = self.parameter_data.at[parameter, "Maximum"]
            param_val = self.dataframe.loc[idx][parameter]
            outputs[i] = param_val
            outputsScaled[i] = (HIGH_VALUE - LOW_VALUE) * (param_val - min_parameter) / (
                        max_parameter - min_parameter) + LOW_VALUE

            i += 1

        return inputs, outputs, outputsScaled


if __name__ == '__main__':
    s = SimulationDataset("train", "basic_corrected_start=350_interval=100_ndraw=8_size=(64,64)", 4, ["cell_cycle"])
    a, b, c = s[3]
    print(b)
    print(c)

from torch.utils.data import Dataset
import pandas as pd
import re
import os
import numpy as np

# PATH (should be the same that in model/dataset_generator.py
DATASET_FOLDER_PATH = "dataset"
DATASET_FOLDER_NAME = "{}_start={}_interval={}_ndraw={}"
DATASET_FILE_NAME = "dataset.csv"
DATA_NAME = "image{}_type={}_time={}.npy"
CELLS_TYPES = "cells_types"
CELLS_DENSITIES = "cells_densities"
OXYGEN = "oxygen"
GLUCOSE = "glucose"

class SimulationDataset(Dataset):
    def __init__(self, folderName, n_draw, parameters_of_interest, cells_type=True, cells_density=True, oxygen=True,
                 glucose=True):
        self.cells_type = cells_type
        self.cells_density = cells_density
        self.oxygen = oxygen
        self.glucose = glucose
        self.parameters_of_interest = parameters_of_interest
        s, i, d = re.findall(r'\d+', folderName)
        self.sequence_times = list(range(int(s), int(s) + int(d) * int(i), int(i)))[:n_draw]
        self.general_path = os.path.join(DATASET_FOLDER_PATH, folderName)
        self.dataframe = pd.read_csv(os.path.join(self.general_path, DATASET_FILE_NAME))

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        inputs = {}
        for sequence_time in self.sequence_times:
            inputs[sequence_time] = {}
            if self.cells_type:
                inputs[sequence_time][CELLS_TYPES] = np.load(
                    os.path.join(self.general_path, DATA_NAME.format(idx, CELLS_TYPES, sequence_time)))
            if self.cells_density:
                inputs[sequence_time][CELLS_DENSITIES] = np.load(
                    os.path.join(self.general_path, DATA_NAME.format(idx, CELLS_DENSITIES, sequence_time)))
            if self.oxygen:
                inputs[sequence_time][OXYGEN] = np.load(
                    os.path.join(self.general_path, DATA_NAME.format(idx, OXYGEN, sequence_time)))
            if self.glucose:
                inputs[sequence_time][GLUCOSE] = np.load(
                    os.path.join(self.general_path, DATA_NAME.format(idx, GLUCOSE, sequence_time)))

        outputs = {}
        for parameter in self.parameters_of_interest:
            outputs[parameter] = self.dataframe.loc[idx][parameter]

        return inputs, outputs


if __name__ == '__main__':
    s = SimulationDataset("basic_start=350_interval=100_ndraw=4", 4, ["cell_cycle_G1"])
    print(s[4])
from torch.utils.data import Dataset
import pandas as pd
import re
import os
import numpy as np

# PATH (should be the same that in model/dataset_generator.py
DATASET_FOLDER_PATH = "datasets"
DATASET_FOLDER_NAME = "{}_start={}_interval={}_ndraw={}_size=({},{})"
DATASET_FILE_NAME = "dataset.csv"
DATA_NAME = "image{}_type={}_time={}.npy"
CELLS_TYPES = "cells_types"
CELLS_DENSITIES = "cells_densities"
OXYGEN = "oxygen"
GLUCOSE = "glucose"


class SimulationDataset(Dataset):
    def __init__(self, folderName, n_draw, parameters_of_interest, img_types=None):
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

    def __len__(self):
        return self.dataframe.shape[0]

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def __getitem__(self, idx):
        inputs = np.zeros((self.n_draw, len(self.img_types), self.height, self.width), dtype=np.float32)
        for draw in range(self.n_draw):
            for index_type in range(len(self.img_types)):
                inputs[draw][index_type] = np.load(
                    os.path.join(self.general_path,
                                 DATA_NAME.format(idx, self.img_types[index_type], self.sequence_times[draw])))

        outputs = np.zeros((len(self.parameters_of_interest)),dtype=np.float32)
        i = 0
        for parameter in self.parameters_of_interest:
            outputs[i] = self.dataframe.loc[idx][parameter]
            i += 1

        return inputs, outputs


if __name__ == '__main__':
    s = SimulationDataset("basic_start=350_interval=100_ndraw=4", 4, ["cell_cycle_G1"])
    print(s[4])

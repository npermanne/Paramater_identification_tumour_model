from simulation import Simulation
from simulation import DEFAULT_PARAMETERS
from enum import Enum
from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
import os

# PATH (should be the same that in networks/dataLoader.py
DATASET_FOLDER_PATH = "datasets"
DATASET_FOLDER_NAME = "{}_start={}_interval={}_ndraw={}_size=({},{})"
DATASET_FILE_NAME = "dataset.csv"
DATA_NAME = "image{}_type={}_time={}.npy"
CELLS_TYPES = "cells_types"
CELLS_DENSITIES = "cells_densities"
OXYGEN = "oxygen"
GLUCOSE = "glucose"


# Class to generate a dataset of simulations images and their respective target parameters
class DatasetGenerator:
    """
    Class to generate a dataset of simulations images and their respective target parameters
    @ Params:
    img_size: size of the image (tuple)
    interval: interval between 2 consecutive draws  (in hours)
    n_draw: number of draw in the total sequence
    parameters: dictionary of parameters
    """

    def __init__(self, img_size, start_draw, interval, n_draw, parameter_data_file, parameters_of_interest, n_samples,
                 name):
        self.img_size = img_size
        self.start_draw = start_draw
        self.interval = interval
        self.n_draw = n_draw
        self.parameter_data = pd.read_csv(parameter_data_file, index_col=0)
        self.parameters_of_interest = parameters_of_interest
        self.n_samples = n_samples
        self.name = DATASET_FOLDER_NAME.format(name, self.start_draw, self.interval, self.n_draw, self.img_size[0],
                                               self.img_size[1])

    def generate_sample(self, parameter, color_type=True):
        sample = {}
        simu = Simulation(self.img_size[0], self.img_size[1], parameter)
        simu.cycle(self.start_draw)
        for i in range(self.n_draw):
            sample[self.start_draw + i * self.interval] = {CELLS_TYPES: simu.get_cells_type(color=color_type),
                                                           CELLS_DENSITIES: simu.get_cells_density(),
                                                           OXYGEN: simu.get_oxygen(), GLUCOSE: simu.get_glucose()}
            simu.cycle(self.interval)
        return sample

    def plot_sample(self, parameter):
        sample = self.generate_sample(parameter)
        plt.axis("off")
        for time, draw in sample.items():
            for type, matrix in draw.items():
                path = os.path.join("pictures",
                                    "simu_{}x{}_t{}_{}.png".format(self.img_size[0], self.img_size[1], time, type))
                plt.imshow(matrix)
                plt.savefig(path, bbox_inches='tight')

    def generate_parameters(self):
        parameters = {}
        for nameRow, row in self.parameter_data.iterrows():
            if nameRow not in self.parameters_of_interest:
                if row["Type"] == "int":
                    parameters[nameRow] = np.full(self.n_samples, int(row["Default Value"]))
                else:
                    parameters[nameRow] = np.full(self.n_samples, float(row["Default Value"]))
            else:
                if row["Type"] == "int":
                    parameters[nameRow] = np.random.randint(low=int(row["Minimum"]), high=int(row["Maximum"])+1, size=self.n_samples)
                else:
                    parameters[nameRow] = np.random.uniform(low=float(row["Minimum"]), high=float(row["Maximum"]), size=self.n_samples)
        return parameters

    def generate_dataset(self):
        general_path = os.path.join(DATASET_FOLDER_PATH, self.name)
        if not os.path.exists(general_path):
            os.makedirs(general_path)

        df = pd.DataFrame(self.generate_parameters())
        df.to_csv(os.path.join(general_path, DATASET_FILE_NAME), index=True)

        times = []
        print("Starting for {}".format(self.n_samples), end='\r')
        for index, rows in df.iterrows():
            start = time.time()
            sample = self.generate_sample(rows, color_type=False)
            for t, draw in sample.items():
                for type, matrix in draw.items():
                    path = os.path.join(general_path, DATA_NAME.format(index, type, t))
                    np.save(path, matrix)
            end = time.time()
            times.append(end - start)
            print("{} out of {} done \t Expected time remaining = {} s".format(index + 1, self.n_samples,
                                                                               (self.n_samples - index - 1) * np.mean(
                                                                                   times)), end='\r')

    def generate_dataset_multi_process(self, process_number: int, chunksize: int):
        general_path = os.path.join(DATASET_FOLDER_PATH, self.name)
        if not os.path.exists(general_path):
            os.makedirs(general_path)

        df = pd.DataFrame(self.generate_parameters())
        df.to_csv(os.path.join(general_path, DATASET_FILE_NAME), index=True)

        global generate

        def generate(i):
            row = df.iloc[i]
            sample = self.generate_sample(row, color_type=False)
            for t, draw in sample.items():
                for type, matrix in draw.items():
                    path = os.path.join(general_path, DATA_NAME.format(i, type, t))
                    np.save(path, matrix)
            print(f"Sample {i} done !")

        pool = Pool(process_number)
        pool.map(generate, range(df.shape[0]))

    def generate_missing(self):
        general_path = os.path.join(DATASET_FOLDER_PATH, self.name)
        df = pd.read_csv(os.path.join(general_path, DATASET_FILE_NAME))

        def generate(i):
            row = df.iloc[i]
            sample = self.generate_sample(row, color_type=False)
            for t, draw in sample.items():
                for type, matrix in draw.items():
                    path = os.path.join(general_path, DATA_NAME.format(i, type, t))
                    np.save(path, matrix)
            print(f"Sample {i} done !")

        generate(384)


if __name__ == '__main__':
    parameter_data_file = os.path.join("simulation", "parameter_data.csv")
    dataset_generator = DatasetGenerator((64, 64), 350, 100, 8, parameter_data_file, [], 100,
                                         "same_value_study")

    # print(dataset_generator.generate_parameters())
    dataset_generator.generate_dataset_multi_process(12, 1)
    # dataset_generator.generate_missing()


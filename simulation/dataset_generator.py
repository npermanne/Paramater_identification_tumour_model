from simulation import Simulation
from simulation import DEFAULT_PARAMETERS
from enum import Enum
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

    def __init__(self, img_size, start_draw, interval, n_draw, parameters, n_samples, name):
        self.img_size = img_size
        self.start_draw = start_draw
        self.interval = interval
        self.n_draw = n_draw
        self.parameters = parameters
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

    def generate_dataset(self):
        general_path = os.path.join(DATASET_FOLDER_PATH, self.name)
        if not os.path.exists(general_path):
            os.makedirs(general_path)

        df = pd.DataFrame(self.parameters)
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


if __name__ == '__main__':
    n_samples = 500
    params = {
        "sources": np.full(n_samples, 100, dtype=int),
        "average_healthy_glucose_absorption": np.full(n_samples, .36),
        "average_cancer_glucose_absorption": np.full(n_samples, .54),
        "average_healthy_oxygen_consumption": np.full(n_samples, 20),
        "average_cancer_oxygen_consumption": np.full(n_samples, 20),
        "quiescent_glucose_level": np.full(n_samples, 0.36 * 2 * 24),
        "quiescent_oxygen_level": np.full(n_samples, 0.54 * 2 * 24),
        "critical_glucose_level": np.full(n_samples, 0.36 * (3 / 4) * 24),
        "critical_oxygen_level": np.full(n_samples, 0.54 * (3 / 4) * 24),
        "cell_cycle_G1": np.random.randint(5, 13, n_samples),
        "cell_cycle_S": np.random.randint(8, 11, n_samples),
        "cell_cycle_G2": np.random.randint(2, 5, n_samples),
        "cell_cycle_M": np.random.randint(1, 3, n_samples),
        "radiosensitivity_G1": np.full(n_samples, 1),
        "radiosensitivity_S": np.full(n_samples, .75),
        "radiosensitivity_G2": np.full(n_samples, 1.25),
        "radiosensitivity_M": np.full(n_samples, 1.25),
        "radiosensitivity_G0": np.full(n_samples, .75),
        "source_glucose_supply": np.full(n_samples, 130),
        "source_oxygen_supply": np.full(n_samples, 4500),
        "glucose_diffuse_rate": np.full(n_samples, 0.2),
        "oxygen_diffuse_rate": np.full(n_samples, 0.2),
        "h_cells": np.full(n_samples, 1000)
    }

    dataset_generator = DatasetGenerator((64, 64), 350, 100, 4, params, n_samples, "basic")
    dataset_generator.generate_dataset()

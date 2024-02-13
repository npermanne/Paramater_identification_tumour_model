from simulation import Simulation
from simulation import DEFAULT_PARAMETERS
from enum import Enum
from matplotlib import pyplot as plt
import numpy as np
import os

# Mean and standard deviation of the parameters of the simulation
PARAMETERS_MEAN_STD = {
    "sources": (100,),
    "average_healthy_glucose_absorption": (.36,),
    "average_cancer_glucose_absorption": (.54,),
    "average_healthy_oxygen_consumption": (20,),
    "average_cancer_oxygen_consumption": (20,),
    "quiescent_glucose_level": (0.36 * 2 * 24,),
    "quiescent_oxygen_level": (0.54 * 2 * 24,),
    "critical_glucose_level": (0.36 * (3 / 4) * 24,),
    "critical_oxygen_level": (0.54 * (3 / 4) * 24,),
    "cell_cycle_G1": (11,),
    "cell_cycle_S": (8,),
    "cell_cycle_G2": (4,),
    "cell_cycle_M": (1,),
    "radiosensitivity_cycle_G1": (1,),
    "radiosensitivity_cycle_S": (.75,),
    "radiosensitivity_cycle_G2": (1.25,),
    "radiosensitivity_cycle_M": (1.25,),
    "radiosensitivity_cycle_G0": (.75,),
    "source_glucose_supply": (130,),
    "source_oxygen_supply": (4500,),
    "glucose_diffuse_rate": (0.2,),
    "oxygen_diffuse_rate": (0.2,),
    "h_cells": (1000,)
}


# Class to generate a dataset of simulations images and their respective target parameters
class DatasetGenerator:
    '''
    Class to generate a dataset of simulations images and their respective target parameters
    @ Params:
    img_size: size of the image (tuple)
    interval: interval between 2 consecutive draws  (in hours)
    n_draw: number of draw in the total sequence
    parameters_considered: set of parameters considered (the other parameters are set to the mean value)
    default_standard_deviation (optional): value of the standard deviation of the parameter if no value is provided)

    '''

    def __init__(self, img_size, start_draw, interval, n_draw, parameters_considered,
                 default_standard_deviation=None):
        self.img_size = img_size
        self.start_draw = start_draw
        self.interval = interval
        self.n_draw = n_draw

    def generate_sample(self, parameters):
        parameters["cell_cycle"] = [parameters["cell_cycle_G1"], parameters["cell_cycle_S"],
                                    parameters["cell_cycle_G2"],
                                    parameters["cell_cycle_M"]]
        parameters["radiosensitivities"] = [parameters["radiosensitivity_cycle_G1"],
                                            parameters["radiosensitivity_cycle_S"],
                                            parameters["radiosensitivity_cycle_G2"],
                                            parameters["radiosensitivity_cycle_M"],
                                            parameters["radiosensitivity_cycle_G0"]]

        sample = {}
        simu = Simulation(self.img_size[0], self.img_size[1], parameters)
        simu.cycle(self.start_draw)
        for i in range(self.n_draw):
            sample[self.start_draw + i * self.interval] = {"cells_types": simu.get_cells_type(),
                                                           "cells_densities": simu.get_cells_density(),
                                                           "oxygen": simu.get_oxygen(), "glucose": simu.get_glucose()}
            simu.cycle(self.interval)
        return sample

    def plot_sample(self, parameters):
        sample = self.generate_sample(parameters)
        plt.axis("off")
        for time, draw in sample.items():
            for type, matrix in draw.items():
                path = os.path.join("pictures",
                                    "simu_{}x{}_t{}_{}.png".format(self.img_size[0], self.img_size[1], time, type))
                plt.imshow(matrix)
                plt.savefig(path,bbox_inches='tight')


if __name__ == '__main__':
    dataset = DatasetGenerator((50, 50), 100, 100, 4, None)
    dataset.plot_sample({keys: value[0] for keys, value in PARAMETERS_MEAN_STD.items()})

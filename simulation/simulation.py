from grid import Grid, get_multiplicator, scale
from cell import HealthyCell, CancerCell, OARCell, Cell
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from enum import Enum
import numpy as np
import random
import time
import os
import math

DEFAULT_PARAMETERS = {
    "sources": 100,
    "average_healthy_glucose_absorption": .36,
    "average_cancer_glucose_absorption": .54,
    "average_healthy_oxygen_consumption": 20,
    "average_cancer_oxygen_consumption": 20,
    "quiescent_multiplier": 2,
    "critical_multiplier": (3 / 4),
    "cell_cycle_G1": 11,
    "cell_cycle_S": 8,
    "cell_cycle_G2": 4,
    "cell_cycle_M": 1,
    "cell_cycle": 24,
    "radiosensitivity_G1": 1,
    "radiosensitivity_S": .75,
    "radiosensitivity_G2": 1.25,
    "radiosensitivity_M": 1.25,
    "radiosensitivity_G0": .75,
    "source_glucose_supply": 130,
    "source_oxygen_supply": 4500,
    "glucose_diffuse_rate": 0.2,
    "oxygen_diffuse_rate": 0.2,
    "h_cells": 1000
}


class Simulation:
    """
    A class implementing the tumour simulation based on cell.py and grid.py

    Parameters:
        x_size (int): the size of the grid along the x-axis.
        y_size (int): the size of the grid along the y-axis.
        params (dict): the initial biological parameters for the simulation.
        treatment_planning (list): the fraction of doses delivered at each time.

    Attributes:
        x_size (int): the size of the grid along the x-axis.
        y_size (int): the size of the grid along the y-axis.
        sources (int): Number of glucose and oxygen sources in the simulation.
        average_healthy_glucose_absorption (float): Average glucose absorption rate for healthy cells.
        average_cancer_glucose_absorption (float): Average glucose absorption rate for cancer cells.
        average_healthy_oxygen_consumption (float): Average oxygen consumption rate for healthy cells.
        average_cancer_oxygen_consumption (float): Average oxygen consumption rate for cancer cells.
        cell_cycle (list): List containing durations of cell cycle stages.
        radiosensitivities (list): List containing radiosensitivities of cell cycle stages.
        quiescent_glucose_level (float): Glucose level for quiescent stage.
        quiescent_oxygen_level (float): Oxygen level for quiescent stage.
        critical_glucose_level (float): Critical glucose level before the cell's death.
        critical_oxygen_level (float): Critical oxygen level before the cell's death.
        source_glucose_supply (float): Glucose supply rate from sources.
        source_oxygen_supply (float): Oxygen supply rate from sources.
        glucose_diffuse_rate (float): Rate of glucose diffusion.
        oxygen_diffuse_rate (float): Rate of oxygen diffusion.
        h_cells (int): Number of initial healthy cells.
        treatment_planning (list or None): List containing treatment planning information.
        hours_passed (int): Hours passed in the simulation.
        grid (Grid): Grid object representing the simulation environment.

    Methods:
        cycle(steps=1): cycle 'steps' hours through the simulation
        get_cells_type(color=True): return a matrix containing the type of cells for each pixel
        get_cells_density(): return a matrix containing the densities of cells for each pixel
        get_glucose(): return a matrix containing the glucose level for each pixel
        get_oxygen(): return a matrix containing the oxygen level for each pixel

    """

    def __init__(self, x_size, y_size, params: dict, treatment_planning=None):
        self.y_size = y_size
        self.x_size = x_size

        self.sources = int(params["sources"])

        self.average_healthy_glucose_absorption = params["average_healthy_glucose_absorption"]
        self.average_cancer_glucose_absorption = params["average_cancer_glucose_absorption"]

        self.average_healthy_oxygen_consumption = params["average_healthy_oxygen_consumption"]
        self.average_cancer_oxygen_consumption = params["average_cancer_oxygen_consumption"]

        if "cell_cycle" in params.keys():
            total_cell_cycles = params["cell_cycle"]
            G1 = int(total_cell_cycles * (11 / 24)) if int(total_cell_cycles * (11 / 24)) > 0 else 1
            S = int(total_cell_cycles * (8 / 24)) if int(total_cell_cycles * (11 / 24)) > 0 else 1
            G2 = int(total_cell_cycles * (4 / 24)) if int(total_cell_cycles * (11 / 24)) > 0 else 1
            M = int(total_cell_cycles - G1 - S - G2) if int(total_cell_cycles - G1 - S - G2) > 0 else 1
            self.cell_cycle = [G1, S, G2, M]
        else:
            self.cell_cycle = [params["cell_cycle_G1"], params["cell_cycle_S"], params["cell_cycle_G2"],
                               params["cell_cycle_M"]]

        self.radiosensitivities = [params["radiosensitivity_G1"], params["radiosensitivity_S"],
                                   params["radiosensitivity_G2"], params["radiosensitivity_M"],
                                   params["radiosensitivity_G0"]]

        self.quiescent_glucose_level = self.average_healthy_glucose_absorption * params["quiescent_multiplier"] * sum(
            self.cell_cycle)
        self.quiescent_oxygen_level = self.average_healthy_oxygen_consumption * params["quiescent_multiplier"] * sum(
            self.cell_cycle)

        self.critical_glucose_level = self.average_healthy_glucose_absorption * params["critical_multiplier"] * sum(
            self.cell_cycle)
        self.critical_oxygen_level = self.average_healthy_oxygen_consumption * params["critical_multiplier"] * sum(
            self.cell_cycle)

        self.source_glucose_supply = params["source_glucose_supply"]
        self.source_oxygen_supply = params["source_oxygen_supply"]

        self.glucose_diffuse_rate = params["glucose_diffuse_rate"]
        self.oxygen_diffuse_rate = params["oxygen_diffuse_rate"]

        self.h_cells = params["h_cells"]

        self.treatment_planning = treatment_planning

        self.hours_passed = 0
        # Init grid
        self.grid = Grid(
            self.x_size,
            self.y_size,
            self.sources,
            self.average_healthy_glucose_absorption,
            self.average_cancer_glucose_absorption,
            self.average_healthy_oxygen_consumption,
            self.average_cancer_oxygen_consumption,
            self.critical_glucose_level,
            self.critical_oxygen_level,
            self.quiescent_oxygen_level,
            self.quiescent_glucose_level,
            self.cell_cycle,
            self.radiosensitivities
        )

        # Init Healthy Cells
        for i in range(self.x_size):
            for j in range(self.y_size):
                if random.random() < self.h_cells / (self.x_size * self.y_size):
                    new_cell = HealthyCell(stage=random.randint(0, 4),
                                           average_healthy_glucose_absorption=self.average_healthy_glucose_absorption,
                                           average_cancer_glucose_absorption=self.average_cancer_glucose_absorption,
                                           average_healthy_oxygen_consumption=self.average_healthy_oxygen_consumption,
                                           average_cancer_oxygen_consumption=self.average_cancer_oxygen_consumption,
                                           critical_glucose_level=self.critical_glucose_level,
                                           critical_oxygen_level=self.critical_oxygen_level,
                                           quiescent_oxygen_level=self.quiescent_oxygen_level,
                                           quiescent_glucose_level=self.quiescent_glucose_level)
                    self.grid.cells[i, j].append(new_cell)

        # Init Cancer Cell
        new_cell = CancerCell(stage=random.randint(0, 3),
                              average_healthy_glucose_absorption=self.average_healthy_glucose_absorption,
                              average_cancer_glucose_absorption=self.average_cancer_glucose_absorption,
                              average_healthy_oxygen_consumption=self.average_healthy_oxygen_consumption,
                              average_cancer_oxygen_consumption=self.average_cancer_oxygen_consumption,
                              critical_glucose_level=self.critical_glucose_level,
                              critical_oxygen_level=self.critical_oxygen_level,
                              quiescent_oxygen_level=self.quiescent_oxygen_level,
                              quiescent_glucose_level=self.quiescent_glucose_level,
                              cell_cycle=self.cell_cycle,
                              radiosensitivities=self.radiosensitivities)

        self.grid.cells[self.x_size // 2, self.y_size // 2].append(new_cell)

        self.grid.count_neighbors()

    def cycle(self, steps=1):
        for _ in range(steps):
            self.grid.fill_source(self.source_glucose_supply, self.source_oxygen_supply)
            self.grid.cycle_cells()
            self.grid.diffuse_glucose(self.glucose_diffuse_rate)
            self.grid.diffuse_oxygen(self.oxygen_diffuse_rate)
            if self.treatment_planning is not None and len(self.treatment_planning) > self.hours_passed and self.treatment_planning[self.hours_passed] != 0:
                self.grid.irradiate(self.treatment_planning[self.hours_passed])
            self.hours_passed += 1
            if self.hours_passed % 24 == 0:
                self.grid.compute_center()

    def get_cells_type(self, color=True):
        if color:
            return [[patch_type_color(self.grid.cells[i][j]) for j in range(self.y_size)] for i in range(self.x_size)]
        else:
            return [[self.grid.cells[i][j].pixel_type() for j in range(self.y_size)] for i in range(self.x_size)]

    def get_cells_density(self):
        return [[len(self.grid.cells[i][j]) for j in range(self.y_size)] for i in range(self.x_size)]

    def get_glucose(self):
        return self.grid.glucose

    def get_oxygen(self):
        return self.grid.oxygen

    def get_total_doses(self):
        return self.grid.total_doses


def patch_type_color(patch):
    if len(patch) == 0:
        return 0, 0, 0
    else:
        return patch[0].cell_color()


def make_gif(parameter, iteration, step, interval, filename, treatment=None):
    simu = Simulation(60, 60, parameter, treatment_planning=treatment)
    fig, axis = plt.subplots(2, 2)
    fig.suptitle("Simulation with {} hours passed".format(0))
    type_plot = axis[0, 0].imshow(simu.get_cells_type())
    axis[0, 0].set_title('Type')
    axis[0, 0].axis('off')

    density_plot = axis[0, 1].imshow(simu.get_cells_density())
    axis[0, 1].set_title('Density')
    axis[0, 1].axis('off')
    colorbar_density = plt.colorbar(density_plot, ax=axis[0, 1])

    glucose_plot = axis[1, 0].imshow(simu.get_glucose())
    axis[1, 0].set_title('Glucose')
    axis[1, 0].axis('off')
    colorbar_glucose = plt.colorbar(glucose_plot, ax=axis[1, 0])

    oxygen_plot = axis[1, 1].imshow(simu.get_oxygen())
    axis[1, 1].set_title('Oxygen')
    axis[1, 1].axis('off')
    colorbar_oxygen = plt.colorbar(oxygen_plot, ax=axis[1, 1])

    def anim_function(frame):
        fig.suptitle("Simulation with {} hours passed".format(simu.hours_passed))
        simu.cycle(step)
        img_density = np.array(simu.get_cells_density())
        img_glucose = simu.get_glucose()
        img_oxygen = simu.get_oxygen()

        type_plot.set_data(simu.get_cells_type())
        density_plot.set_data(img_density)
        glucose_plot.set_data(img_glucose)
        oxygen_plot.set_data(img_oxygen)

        # Update colorbar's mappable
        colorbar_density.mappable.set_clim(vmin=np.min(img_density), vmax=np.max(img_density))
        colorbar_glucose.mappable.set_clim(vmin=np.min(img_glucose), vmax=np.max(img_glucose))
        colorbar_oxygen.mappable.set_clim(vmin=np.min(img_oxygen), vmax=np.max(img_oxygen))

    anim_created = FuncAnimation(fig, anim_function, frames=iteration, interval=interval)
    anim_created.save(filename)


if __name__ == '__main__':
    # dose_hours = list(range(350, 1300, 24))
    # default_treatment = np.zeros(1300)
    # default_treatment[dose_hours] = 2
    #
    # time_planning = []
    # time_not_planning = []
    #
    # for _ in range(10):
    #     start = time.time()
    #     simu = Simulation(64, 64, DEFAULT_PARAMETERS, treatment_planning=default_treatment)
    #     simu.cycle(1200)
    #     end = time.time()
    #     time_planning.append(end - start)
    #
    # for _ in range(10):
    #     start = time.time()
    #     simu = Simulation(64, 64, DEFAULT_PARAMETERS)
    #     simu.cycle(1200)
    #     end = time.time()
    #     time_not_planning.append(end - start)
    #
    # print(f"Mean time taken for the simulation of 1200 hours (with treatment planning): {np.mean(time_planning)}")
    # print(f"Mean time taken for the simulation of 1200 hours (without treatment planning): {np.mean(time_not_planning)}")

    # make_gif(DEFAULT_PARAMETERS, 210, 5, 1, f"normal_oxygen_healthy_consumption.gif")
    # DEFAULT_PARAMETERS["average_healthy_oxygen_consumption"] = 12
    # make_gif(DEFAULT_PARAMETERS, 210, 5, 1, f"lower_oxygen_healthy_consumption.gif")
    # DEFAULT_PARAMETERS["average_healthy_oxygen_consumption"] = 30
    # make_gif(DEFAULT_PARAMETERS, 210, 5, 1, f"higher_oxygen_healthy_consumption.gif")


    doses = np.load(os.path.join("datasets","best_model_treatment_dataset_start=350_interval=100_ndraw=8_size=(64,64)", "treatments.npy"))[0]
    print(doses.shape)
    make_gif(DEFAULT_PARAMETERS, 210,5,1,f"Normal_with_treatment.gif",treatment=doses)

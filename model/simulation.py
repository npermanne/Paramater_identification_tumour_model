from grid import Grid
from cell import HealthyCell, CancerCell, OARCell, Cell
from matplotlib import pyplot as plt
from enum import Enum
import numpy as np
import random

DEFAULT_PARAMETERS = {
    "sources": 100,
    "average_healthy_glucose_absorption": .36,
    "average_cancer_glucose_absorption": .54,
    "average_healthy_oxygen_consumption": 20,
    "average_cancer_oxygen_consumption": 20,
    "quiescent_glucose_level": 0.36 * 2 * 24,
    "quiescent_oxygen_level": 0.54 * 2 * 24,
    "critical_glucose_level": 0.36 * (3 / 4) * 24,
    "critical_oxygen_level": 0.54 * (3 / 4) * 24,
    "cell_cycle_G1": 11,
    "cell_cycle_S": 8,
    "cell_cycle_G2": 4,
    "cell_cycle_M": 1,
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
    def __init__(self, x_size, y_size, params: dict, treatment_planning=None):
        self.y_size = y_size
        self.x_size = x_size

        self.sources = int(params["sources"])

        self.average_healthy_glucose_absorption = params["average_healthy_glucose_absorption"]
        self.average_cancer_glucose_absorption = params["average_cancer_glucose_absorption"]

        self.average_healthy_oxygen_consumption = params["average_healthy_oxygen_consumption"]
        self.average_cancer_oxygen_consumption = params["average_cancer_oxygen_consumption"]

        self.quiescent_glucose_level = params["quiescent_glucose_level"]
        self.quiescent_oxygen_level = params["quiescent_oxygen_level"]

        self.critical_glucose_level = params["critical_glucose_level"]
        self.critical_oxygen_level = params["critical_oxygen_level"]

        self.cell_cycle = [params["cell_cycle_G1"], params["cell_cycle_S"], params["cell_cycle_G2"],
                           params["cell_cycle_M"]]

        self.radiosensitivities = [params["radiosensitivity_G1"], params["radiosensitivity_S"],
                                   params["radiosensitivity_G2"], params["radiosensitivity_M"],
                                   params["radiosensitivity_G0"]]

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
            if self.treatment_planning is not None:
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
        return self.grid.glucose


def patch_type_color(patch):
    if len(patch) == 0:
        return 0, 0, 0
    else:
        return patch[0].cell_color()


if __name__ == '__main__':
    for i in DEFAULT_PARAMETERS.keys():
        print(i)
    iterations = 1200
    STEP = 5
    a = np.zeros(24)
    a[23] = 0
    planning = np.tile(a, iterations // 24)
    simu = Simulation(50, 50, DEFAULT_PARAMETERS, planning)

    plt.ion()
    fig, axis = plt.subplots(2, 2)
    fig.suptitle("Simulation with {} hours passed".format(0))

    type_plot = axis[0, 0].imshow(simu.get_cells_type())
    axis[0, 0].set_title('Type')

    density_plot = axis[0, 1].imshow(simu.get_cells_density())
    axis[0, 1].set_title('Density')
    colorbar_density = plt.colorbar(density_plot)

    glucose_plot = axis[1, 0].imshow(simu.get_glucose())
    axis[1, 0].set_title('Glucose')
    colorbar_glucose = plt.colorbar(glucose_plot)

    oxygen_plot = axis[1, 1].imshow(simu.get_oxygen())
    axis[1, 1].set_title('Oxygen')
    colorbar_oxygen = plt.colorbar(oxygen_plot)

    for i in range(iterations // STEP):
        fig.suptitle("Simulation with {} hours passed".format(simu.hours_passed))
        simu.cycle(STEP)

        type_plot = axis[0, 0].imshow(simu.get_cells_type())
        density_plot = axis[0, 1].imshow(simu.get_cells_density())
        colorbar_density.update_normal(density_plot)
        glucose_plot = axis[1, 0].imshow(simu.get_glucose())
        colorbar_glucose.update_normal(glucose_plot)
        oxygen_plot = axis[1, 1].imshow(simu.get_oxygen())
        colorbar_oxygen.update_normal(oxygen_plot)

        fig.canvas.draw()
        fig.canvas.flush_events()

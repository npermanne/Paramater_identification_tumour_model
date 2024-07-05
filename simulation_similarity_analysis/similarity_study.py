import matplotlib.pyplot as plt
from enum import Enum
from multiprocessing import Pool
from multiprocessing import set_start_method
import time

import numpy as np

from utils import *

SIMILARITY_ANALYSIS_FOLDER = "simulation_similarity_analysis"


class SimilarityAnalysis:
    def __init__(self, folder_name, parameter, tol, *dataset):
        self.comparator = Comparator(*dataset)
        self.parameter = parameter
        self.parameter_range = pd.read_csv(os.path.join("simulation", "parameter_data.csv"), index_col=0).loc[[parameter], ["Minimum", "Maximum"]].values[0]
        self.type = pd.read_csv(os.path.join("simulation", "parameter_data.csv"), index_col=0).loc[[parameter], ["Type"]].values[0][0]
        self.tol = (self.parameter_range[1] - self.parameter_range[0])
        self.path = os.path.join(SIMILARITY_ANALYSIS_FOLDER, f"{folder_name}_analysis")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def image_mean_diff(self, draw, image_type, iteration):
        fig, ax = plt.subplots(5, 5, dpi=500)
        fig.suptitle(f"Evolution of the mean difference between 2 images of types {image_type} and draw {draw}")
        if self.type == 'float':
            differences = np.linspace(0, self.parameter_range[1] - self.parameter_range[0], 25, endpoint=False)
        else:
            differences = np.arange(0, 25, 1)
        datas = [self.comparator.mean_and_std_difference(Metric.IMAGE_ABSOLUTE_DIFFERENCE, draw, image_type, self.parameter, difference, self.tol, 12, iteration)[0] for difference in differences]

        v_min = np.min(datas)
        v_max = np.max(datas)
        for index, difference in enumerate(differences):
            title = f"{difference:2.3f}" if self.type == 'float' else f"{difference}"
            i, j = index // 5, index % 5
            ax[i, j].set_title(title, x=-0.2, y=0.25)
            ax[i, j].imshow(datas[index], vmin=v_min, vmax=v_max)
            ax[i, j].axis('off')
        plt.tight_layout()
        save_path = os.path.join(self.path, f"Evolution of the mean difference between 2 images of types {image_type} at time {draw}.png")
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()

    def metric_mean_diff(self, metric: Metric, draw, image_type, iteration, number_of_measures):
        differences = np.linspace(0, self.parameter_range[1] - self.parameter_range[0], number_of_measures, endpoint=False)
        datas = [self.comparator.mean_and_std_difference(metric, draw, image_type, self.parameter, difference, self.tol, 12, iteration) for difference in differences]
        mean = np.array([data[0] for data in datas])
        std = np.array([data[1] for data in datas])

        plt.plot(differences, mean, color="orange")
        plt.fill_between(differences, mean - std, mean + std, alpha=0.2, color="orange")
        plt.title(f"Evolution of similarity between images of types {image_type} at time {draw}")
        plt.ylabel(f"{metric.__str__()}")
        plt.xlabel(f"Difference in {self.parameter}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, f"Evolution of the {metric.__str__()} between 2 images of types {image_type} at time {draw}.png"))
        plt.close()


if __name__ == '__main__':
    tolerance_per_param = {
        "average_healthy_glucose_absorption": 0,
        "average_cancer_glucose_absorption": 0,
        "average_healthy_oxygen_consumption": 0,
        "average_cancer_oxygen_consumption": 0,
        "cell_cycle": 0
    }
    s = SimilarityAnalysis("cell_cycle_without_treatment", "cell_cycle", 0, "full_dataset_start=350_interval=100_ndraw=8_size=(64,64)")
    #s.image_mean_diff(350, "cells_types", 10)
    s.metric_mean_diff(Metric.CORRELATION_HISTOGRAM, 350, "cells_types", 10, 28)

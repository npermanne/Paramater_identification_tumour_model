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
        self.tol = tol
        self.path = os.path.join(SIMILARITY_ANALYSIS_FOLDER, f"{folder_name}_analysis")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def plot_mean_difference_image_square(self, draw, iteration, differences):
        for images_type in ['cells_types', 'cells_densities', 'oxygen', 'glucose']:
            fig, ax = plt.subplots(5, 5, dpi=500)
            fig.suptitle(f"Evolution of the mean difference between 2 images of types {images_type} and draw {draw}")
            datas = [
                self.comparator.mean_and_std_difference(Metric.IMAGE_ABSOLUTE_DIFFERENCE, draw, images_type,
                                                        self.parameter, difference, self.tol, 12, iteration)[0] for
                difference in differences]

            v_min = np.min(datas)
            v_max = np.max(datas)
            for index, difference in enumerate(differences):
                title = f"{difference:2.3f}" if type(difference) == float else f"{difference}"
                i, j = index // 5, index % 5
                ax[i, j].set_title(title, x=-0.2, y=0.25)
                ax[i, j].imshow(datas[index], vmin=v_min, vmax=v_max)
                ax[i, j].axis('off')
            plt.tight_layout()
            save_path = os.path.join(self.path,
                                     f"Evolution of the mean difference between 2 images of types {images_type}  at time {draw}.png")
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            plt.close()

    def plot_metrics(self, metric: Metric, draw, iteration, images_type, differences):
        datas = [
            self.comparator.mean_and_std_difference(metric, draw, images_type, self.parameter, difference, self.tol, 12,
                                                    iteration) for difference in differences]

        mean = [data[0] for data in datas]
        std = [data[1] for data in datas]
        plt.errorbar(differences, mean, std)
        plt.title(f"Evolution of {metric.__str__()} between 2 images of types {images_type} and draw {draw}")
        save_path = os.path.join(self.path,
                                 f"Evolution of the {metric.__str__()} between 2 images of types {images_type}  at time {draw}.png")
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    print("hello")

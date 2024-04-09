import matplotlib.pyplot as plt
from enum import Enum
from multiprocessing import Pool
from multiprocessing import set_start_method
import time

from utils import *

SIMILARITY_ANALYSIS_FOLDER = "simulation_similarity_analysis"


class SimilarityAnalysis:
    def __init__(self, parameter, tol, *dataset):
        self.comparator = Comparator(*dataset)
        self.parameter = parameter
        self.tol = tol
        self.path = os.path.join(SIMILARITY_ANALYSIS_FOLDER, f"{parameter}_analysis")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def mean_different_image(self, draw, iteration, differences):
        for images_type in ['cells_types', 'cells_densities', 'oxygen', 'glucose']:
            fig, ax = plt.subplots(1, len(differences), dpi=250)
            fig.suptitle(f"Evolution of the mean difference between 2 images of types {images_type} and draw {draw}",
                         y=0.7)
            datas = [
                self.comparator.mean_difference(Metric.IMAGE_ABSOLUTE_DIFFERENCE, draw, images_type, self.parameter,
                                                difference, self.tol, 12, iteration) for difference in differences]
            v_min = np.min(datas)
            v_max = np.max(datas)
            for index, difference in enumerate(differences):
                ax[index].set_title(f"{difference}")
                ax[index].imshow(datas[index], vmin=v_min, vmax=v_max)
                ax[index].axis('off')
            plt.tight_layout()
            save_path = os.path.join(self.path,
                                     f"Evolution of the mean difference between 2 images of types {images_type}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

    def mean_different_metric(self, metric: Metric, iteration, differences):
        fig, ax = plt.subplots(1, 4, dpi=150, figsize=(16, 2))
        fig.suptitle(f"Mean {metric} between 2 images in function of the parameter difference")
        for index, images_type in enumerate(['cells_types', 'cells_densities', 'oxygen', 'glucose']):
            X = differences
            for draw in [350, 550, 750, 950]:
                Y = [self.comparator.mean_difference(metric, draw, images_type, self.parameter, diff, self.tol,
                                                     12, iteration) for diff in X]
                if index == 3:
                    ax[index].plot(X, Y, label=draw)
                else:
                    ax[index].plot(X, Y)
            ax[index].set_title(f"{images_type}")
        fig.tight_layout()
        fig.legend()
        save_path = os.path.join(self.path,
                                 f"Mean of the {metric} between 2 images in function of the parameter difference.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    analysis = SimilarityAnalysis(
        "cell_cycle",
        0,
        "same_study_start=350_interval=100_ndraw=8_size=(64,64)",
        "cell_cycle_start=350_interval=100_ndraw=8_size=(64,64)")

    i = 0
    analysis.mean_different_image(350, 1000, [0, 5, 10, 15, 20])
    i += 1
    print(f"{i}/6", end='\r')
    for metric in [Metric.CORRELATION_HISTOGRAM, Metric.SSIM, Metric.MEAN_ABSOLUTE_ERROR,
                   Metric.ROOT_MEAN_SQUARED_ERROR, Metric.MAX_ABSOLUTE_ERROR]:
        analysis.mean_different_metric(metric, 1000, range(0, 21))
        i += 1
        print(f"{i}/6", end='\r')

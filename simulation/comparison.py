import numpy as np
from matplotlib import pyplot as plt
from simulation import Simulation
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
import os
import pandas as pd
import random

PARAMETER_DATA = pd.read_csv(os.path.join("simulation", "parameter_data.csv"), index_col=0)
MIN_PRED = 0
MAX_PRED = 1


class Comparison:
    def __init__(self, config, step, duration):
        self.comparison_data = pd.read_csv(os.path.join("results", config, "evaluation_data.csv"))
        self.default_param = {index: row["Default Value"] for index, row in PARAMETER_DATA.iterrows()}
        for column in self.comparison_data:
            param_name = column[len(column.split('_')[0]) + 1:]
            min_param = PARAMETER_DATA.loc[param_name]["Minimum"]
            max_param = PARAMETER_DATA.loc[param_name]["Maximum"]
            for i in range(len(self.comparison_data)):
                self.comparison_data.loc[i, column] = min_param + (self.comparison_data.loc[i, column] - MIN_PRED) * (
                        max_param - min_param) / (MAX_PRED - MIN_PRED)
        self.step = step
        self.duration = duration
        self.timesteps = range(0, duration, step)
        self.image_types = ["cells_type", "cells_density", "oxygen", "glucose"]

    def get_2_simulations(self, idx):
        comp_param = dict(self.comparison_data.loc[idx])
        true_param = {key: comp_param.get(f"true_{key}", value) for key, value in self.default_param.items()}
        pred_param = {key: comp_param.get(f"predicted_{key}", value) for key, value in self.default_param.items()}

        global run_simu

        def run_simu(true_or_pred):
            simu = Simulation(64, 64, true_param) if true_or_pred == 'true' else Simulation(64, 64, pred_param)
            images = {image_type: {} for image_type in self.image_types}
            for i in self.timesteps:
                images["cells_type"][i] = simu.get_cells_type(color=False)
                images["cells_density"][i] = simu.get_cells_density()
                images["oxygen"][i] = simu.get_oxygen()
                images["glucose"][i] = simu.get_glucose()
                simu.cycle(self.step)
            return images

        with Pool(processes=10) as pool:
            true_images = pool.map(run_simu, ["true" for _ in range(10)])
            pred_images = pool.map(run_simu, ["pred" for _ in range(10)])

        true_images_arranged = {image_type: {timestep: [0 for _ in range(10)] for timestep in self.timesteps} for
                                image_type
                                in self.image_types}
        pred_images_arranged = {image_type: {timestep: [0 for _ in range(10)] for timestep in self.timesteps} for
                                image_type
                                in self.image_types}

        for i in range(10):
            for timestep in self.timesteps:
                for image_type in self.image_types:
                    true_images_arranged[image_type][timestep][i] = true_images[i][image_type][timestep]
                    pred_images_arranged[image_type][timestep][i] = pred_images[i][image_type][timestep]

        for image_type in self.image_types:
            for timestep in self.timesteps:
                true_images_arranged[image_type][timestep] = np.array(true_images_arranged[image_type][timestep])
                pred_images_arranged[image_type][timestep] = np.array(pred_images_arranged[image_type][timestep])
        return true_images_arranged, pred_images_arranged

    def get_2_mean_simulations(self, idx):

        true_images_arranged, pred_images_arranged = self.get_2_simulations(idx)

        true_images_mean = {
            image_type: {timestep: np.mean(true_images_arranged[image_type][timestep], axis=0) for timestep in
                         self.timesteps} for
            image_type in self.image_types}
        pred_images_mean = {
            image_type: {timestep: np.mean(pred_images_arranged[image_type][timestep], axis=0) for timestep in
                         self.timesteps} for
            image_type in self.image_types}

        return true_images_mean, pred_images_mean

    def get_2_mean_differences(self, idx):
        number_of_diff = 10
        true_images_arranged, pred_images_arranged = self.get_2_simulations(idx)
        inherent_differences = {
            image_type: {timestep: [0 for _ in range(number_of_diff)] for timestep in self.timesteps} for image_type in
            self.image_types}
        prediction_differences = {
            image_type: {timestep: [0 for _ in range(number_of_diff)] for timestep in self.timesteps} for image_type in
            self.image_types}

        for i in range(number_of_diff):
            index = np.random.randint(10, size=4)
            for image_type in self.image_types:
                for timestep in self.timesteps:
                    inherent_differences[image_type][timestep][i] = np.absolute(
                        true_images_arranged[image_type][timestep][index[0]] -
                        true_images_arranged[image_type][timestep][index[1]])
                    prediction_differences[image_type][timestep][i] = np.absolute(
                        true_images_arranged[image_type][timestep][index[2]] -
                        pred_images_arranged[image_type][timestep][index[3]])

        mean_inherent_differences = {
            image_type: {timestep: np.mean(inherent_differences[image_type][timestep], axis=0) for timestep in
                         self.timesteps} for image_type in self.image_types}
        mean_prediction_differences = {
            image_type: {timestep: np.mean(prediction_differences[image_type][timestep], axis=0) for timestep in
                         self.timesteps} for image_type in self.image_types}

        return mean_inherent_differences, mean_prediction_differences

    def make_animated_gif(self, true_simu, pred_simu, interval, column_name, filename):

        fig, axis = plt.subplots(4, 2, figsize=(6, 10))
        plots = [[None for _ in range(len(axis[0]))] for _ in range(len(axis))]
        colorbars = [[None for _ in range(len(axis[0]))] for _ in range(len(axis))]
        vmins = {image_type: min(np.min(true_simu[image_type][0]), np.min(pred_simu[image_type][0])) for
                 image_type in self.image_types}
        vmaxs = {image_type: max(np.max(true_simu[image_type][0]), np.max(pred_simu[image_type][0])) for
                 image_type in self.image_types}
        for i, image_type in enumerate(self.image_types):
            for j in range(2):
                if j == 0:
                    plots[i][j] = axis[i, j].imshow(true_simu[image_type][0], vmin=vmins[image_type],
                                                    vmax=vmaxs[image_type])
                    axis[i, j].axis("off")
                    axis[i, j].set_title(f"{column_name[0]} {image_type}")
                elif j == 1:
                    plots[i][j] = axis[i, j].imshow(pred_simu[image_type][0], vmin=vmins[image_type],
                                                    vmax=vmaxs[image_type])
                    axis[i, j].axis("off")
                    axis[i, j].set_title(f"{column_name[1]} {image_type}")
                colorbars[i][j] = fig.colorbar(plots[i][j])

        def animate(frame):
            timestep = self.timesteps[frame]
            vmins = {image_type: min(np.min(true_simu[image_type][timestep]),
                                     np.min(pred_simu[image_type][timestep])) for image_type in self.image_types}
            vmaxs = {image_type: max(np.max(true_simu[image_type][timestep]),
                                     np.max(pred_simu[image_type][timestep])) for image_type in self.image_types}
            fig.suptitle("Simulation with {} hours passed".format(self.timesteps[frame]))
            for i, image_type in enumerate(self.image_types):
                for j in range(2):
                    if j == 0:
                        plots[i][j].set_data(true_simu[image_type][timestep])
                    elif j == 1:
                        plots[i][j].set_data(true_simu[image_type][timestep])

                    plots[i][j].norm.autoscale([vmins[image_type], vmaxs[image_type]])
                    colorbars[i][j].mappable.set_clim(vmin=vmins[image_type], vmax=vmaxs[image_type])

        anim_created = FuncAnimation(fig, animate, frames=len(self.timesteps), interval=interval)
        anim_created.save(filename)

    def plot_metrics(self, idx, metricsName, metricsFunction, filename):
        number_of_diff = 10
        true_images_arranged, pred_images_arranged = self.get_2_simulations(idx)
        inherent_metrics = {
            image_type: {timestep: np.zeros(10) for timestep in self.timesteps} for image_type in
            self.image_types}
        prediction_metrics = {
            image_type: {timestep: np.zeros(10) for timestep in self.timesteps} for image_type in
            self.image_types}

        for i in range(number_of_diff):
            index = np.random.randint(10, size=4)
            for image_type in self.image_types:
                for timestep in self.timesteps:
                    inherent_metrics[image_type][timestep][i] = metricsFunction(
                        true_images_arranged[image_type][timestep][index[0]],
                        true_images_arranged[image_type][timestep][index[1]])
                    prediction_metrics[image_type][timestep][i] = metricsFunction(
                        true_images_arranged[image_type][timestep][index[2]],
                        pred_images_arranged[image_type][timestep][index[3]])

        inherent_metrics_mean = {
            image_type: {timestep: np.mean(inherent_metrics[image_type][timestep]) for timestep in self.timesteps} for
            image_type in self.image_types}
        prediction_metrics_mean = {
            image_type: {timestep: np.mean(prediction_metrics[image_type][timestep]) for timestep in self.timesteps} for
            image_type in self.image_types}

        inherent_metrics_std = {
            image_type: {timestep: np.std(inherent_metrics[image_type][timestep]) for timestep in self.timesteps} for
            image_type in self.image_types}
        prediction_metrics_std = {
            image_type: {timestep: np.std(prediction_metrics[image_type][timestep]) for timestep in self.timesteps} for
            image_type in self.image_types}

        fig, axis = plt.subplots(4, 2, figsize=(12,16))
        fig.suptitle(f"Inherent and Predicted {metricsName}")
        for i, image_type in enumerate(self.image_types):
            axis[i, 0].plot(self.timesteps, inherent_metrics_mean[image_type].values(), label="Inherent")
            axis[i, 0].plot(self.timesteps, prediction_metrics_mean[image_type].values(), label="Prediction")
            axis[i, 0].set_title(f"Mean metric for {image_type}")
            axis[i, 1].plot(self.timesteps, inherent_metrics_std[image_type].values(), label="Inherent")
            axis[i, 1].plot(self.timesteps, prediction_metrics_std[image_type].values(), label="Prediction")
            axis[i, 1].set_title(f"Std metric for {image_type}")
            axis[i, 0].legend()
            axis[i, 1].legend()
        plt.savefig(filename)


def corr_hist_function(image1, image2):
    bins = 100
    min_value, max_value = min(np.min(image1), np.min(image2)), max(np.max(image1), np.max(image2))
    hist1, bins1 = np.histogram(image1, bins=np.linspace(min_value, max_value + 1, bins))
    hist2, bins2 = np.histogram(image2, bins=np.linspace(min_value, max_value + 1, bins))
    return np.corrcoef(hist1, hist2)[0, 1]

if __name__ == '__main__':
    comparator = Comparison("config3_full", 5, 1200)
    comparator.plot_metrics(3, "Histogram Correlation",corr_hist_function, "histo_corr.png")

import os.path
import random

from networks.model import Network
from multiprocessing import Pool
import pandas as pd
import time


class HyperparameterTuning:
    def __init__(self, tuning_name, task: dict, hyperparameters: dict, n_epoch, early_stopping_patience,
                 early_stopping_delta):
        self.tuning_name = tuning_name
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.n_epoch = n_epoch
        self.task = task
        self.hyperparameters = hyperparameters
        self.results = pd.DataFrame(columns=["Time", "Performance"] + list(self.hyperparameters.keys()))

    def execute_once(self, idx, params):
        config = {"DATASET": self.task, "MODEL": {}, "TRAINING": {}, "NAME": os.path.join(self.tuning_name, f"{idx}")}
        config["MODEL"]["INPUT_LSTM"] = params["INPUT_LSTM"]
        config["MODEL"]["OUTPUT_LSTM"] = params["OUTPUT_LSTM"]
        config["MODEL"]["LSTM_LAYERS"] = params["LSTM_LAYERS"]
        config["MODEL"]["CONV_LAYERS"] = params["CONV_LAYERS"]
        config["TRAINING"]["LEARNING_RATE"] = params["LEARNING_RATE"]
        config["TRAINING"]["BATCH_SIZE"] = params["BATCH_SIZE"]
        config["TRAINING"]["EPOCH"] = self.n_epoch
        config["TRAINING"]["EARLY_STOPPING_DELTA"] = self.early_stopping_delta
        config["TRAINING"]["EARLY_STOPPING_PATIENCE"] = self.early_stopping_patience
        config["TRAINING"]["DEVICE"] = "cpu"

        start = time.time()
        network = Network(config)
        network.train()
        perf = network.evaluate()
        end = time.time()

        params["Performance"] = perf
        params["Time"] = end - start
        self.results.loc[idx] = params

    def random_search_pool(self, iteration, pool_size):
        data = []
        for i in range(iteration):
            random_param = {key: random.choice(self.hyperparameters[key]) for key in self.hyperparameters.keys()}
            data.append((i, random_param))

        global function

        def function(a):
            self.execute_once(a[0], a[1])

        pool = Pool(pool_size)
        pool.map(function, data)

        self.results.to_csv(os.path.join(self.tuning_name, "performances.csv"))

    def random_search(self, iteration):
        for i in range(iteration):
            random_param = {key: random.choice(self.hyperparameters[key]) for key in self.hyperparameters.keys()}
            self.execute_once(i, random_param)

        self.results.to_csv(os.path.join("results", self.tuning_name, "performances.csv"))


if __name__ == '__main__':
    my_task = {
        "FOLDER_NAME": "full_param_study_start=350_interval=100_ndraw=8_size=(64,64)",
        "N_DRAWS": 4,
        "IMG_TYPES": ["cells_types", "cells_densities", "oxygen", "glucose"],
        "PARAMETERS_OF_INTEREST": ["cell_cycle", "average_healthy_glucose_absorption",
                                   "average_cancer_glucose_absorption",
                                   "average_healthy_oxygen_consumption", "average_cancer_oxygen_consumption"]
    }
    my_hyperparameters = {
        "INPUT_LSTM": [1, 10, 100, 200, 500, 1000],
        "OUTPUT_LSTM": [1, 10, 100, 200, 500, 1000],
        "LSTM_LAYERS": [1, 2, 3, 4, 5],
        "CONV_LAYERS": [1, 2, 3, 4],
        "LEARNING_RATE": [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        "BATCH_SIZE": [1, 2, 4, 8, 16, 32, 64]
    }

    my_tuning = HyperparameterTuning("random_search_1", my_task, my_hyperparameters, 50, 10, 0.001)
    my_tuning.random_search(30)

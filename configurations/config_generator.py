import yaml
import os

if __name__ == '__main__':
    model_dico = {
        "CONV_LAYERS": 4,
        "FEED_FORWARD": 200,
        "LSTM_LAYERS": 1,
        "INPUT_LSTM": 200,
        "OUTPUT_LSTM": 200
    }
    training_dico = {
        "BATCH_SIZE": 1,
        "EPOCH": 300,
        "LEARNING_RATE": 0.0001,
        "L2_REGULARIZATION": 0.001,
        "EARLY_STOPPING_MIN_DELTA": 0.001,
        "EARLY_STOPPING_PATIENCE": 10
    }
    for i, dataset in enumerate(["full_dataset_start=350_interval=100_ndraw=8_size=(64,64)", "full_treatment_dataset_start=350_interval=100_ndraw=8_size=(64,64)"]):
        for draw in range(1, 9):
            filename = os.path.join("configurations", f"best_config_{'without' if i == 0 else 'with'}_treatment_{draw}_draw.yaml")
            dataset_dico = {
                "FOLDER_NAME": dataset,
                "IMG_TYPES": ["cells_types", "cells_densities", "oxygen", "glucose"],
                "N_DRAWS": draw,
                "PARAMETERS_OF_INTEREST": [
                    "average_healthy_glucose_absorption",
                    "average_cancer_glucose_absorption",
                    "average_healthy_oxygen_consumption",
                    "average_cancer_oxygen_consumption",
                    "cell_cycle"
                ]
            }
            config = {"DATASET": dataset_dico, "MODEL": model_dico, "TRAINING": training_dico}
            yaml.dump(config, open(filename, "w"))

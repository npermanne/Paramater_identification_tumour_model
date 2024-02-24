import matplotlib.pyplot as plt
from networks.dataLoader import SimulationDataset
import numpy as np
from networks.architecture import Net
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os

# Parameters
RESULTS_FOLDER = "results"


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL
#
######################################################################################

class Network:
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS:
    #     - param (dic): dictionnary containing the parameters defined in the
    #                    configuration (yaml) file
    # --------------------------------------------------------------------------------
    def __init__(self, param):
        ###############################
        # USEFUL VARIABLES
        ###############################
        self.img_types = param["DATASET"]["IMG_TYPES"]
        self.n_draws = param["DATASET"]["N_DRAWS"]
        self.parameter_of_interest = param["DATASET"]["PARAMETERS_OF_INTEREST"]
        self.batch_size = param["TRAINING"]["BATCH_SIZE"]
        self.epochs = param["TRAINING"]["EPOCH"]
        self.name = param["NAME"]

        ###############################
        # LOAD DATASET
        ###############################
        self.train_dataset = SimulationDataset(param["DATASET"]["FOLDER_NAME"], self.n_draws,
                                               self.parameter_of_interest, self.img_types)
        self.val_dataset = SimulationDataset(param["DATASET"]["FOLDER_NAME"], self.n_draws, self.parameter_of_interest,
                                             self.img_types)
        self.test_dataset = SimulationDataset(param["DATASET"]["FOLDER_NAME"], self.n_draws, self.parameter_of_interest,
                                              self.img_types)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        ###############################
        # ARCHITECTURE INITIALIZATION
        ###############################
        self.param_architecture = {
            "N_DRAWS": self.n_draws,
            "N_TYPES": len(self.img_types),
            "INPUT_LSTM": param["MODEL"]["INPUT_LSTM"],
            "OUTPUT_LSTM": param["MODEL"]["INPUT_LSTM"],
            "BATCH_SIZE": self.batch_size,
            "HEIGHT": self.train_dataset.get_height(),
            "WIDTH": self.train_dataset.get_width(),
            "N_PARAMS": len(param["DATASET"]["PARAMETERS_OF_INTEREST"])
        }
        self.network = Net(self.param_architecture).to(param["TRAINING"]["DEVICE"])

        ###############################
        # TRAINING PARAMETERS
        ###############################
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.criterion = nn.MSELoss()

    ###############################
    # LOAD WEIGHTS
    ###############################
    def loadWeights(self):
        self.network.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl'))

    ###############################
    # TRAINING LOOP
    ###############################
    def train(self):
        validation_losses = np.zeros(self.epochs)
        losses = np.zeros(self.epochs)

        # EPOCHS ITERATIONS
        for iter_epoch in range(self.epochs):
            print("Epoch {}/{}".format(iter_epoch, self.epochs))

            # TRAINING
            running_loss = 0
            for iter_train, data in enumerate(self.train_dataloader):
                inputs, outputs = data

                # Forward Pass
                predicted = self.network.forward(inputs)
                loss = self.criterion(predicted, outputs)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Loss data
                running_loss += loss.item()

            losses[iter_epoch] = running_loss / (iter_train + 1)

            # VALIDATION
            running_validation_loss = 0
            with torch.no_grad():
                for iter_val, data in enumerate(self.val_dataloader):
                    validation_inputs, validation_outputs = data

                    # Forward Pass
                    predicted = self.network.forward(validation_inputs)
                    validation_loss = self.criterion(predicted, validation_outputs)

                    # Validation Loss data
                    running_validation_loss += validation_loss.item()

            validation_losses[iter_epoch] = running_loss / (iter_val + 1)

        # CREATE FOLDER FOR SAVING
        folder_path = os.path.join(RESULTS_FOLDER, self.name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path_weight = os.path.join(folder_path, "weight.pkl")
        path_graph = os.path.join(folder_path, "performance.png")

        # SAVE WEIGHT
        torch.save(self.network.state_dict(), path_weight)

        # PLOT PERFORMANCE CURVE
        X = np.arange(self.epochs)
        plt.plot(X, losses, label="Training Loss")
        plt.plot(X, validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path_graph)

    def evaluate(self):
        print("hello")

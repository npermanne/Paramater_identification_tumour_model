import torch.nn as nn
import torch.nn.functional as F


######################################################################################
#
# CLASS DESCRIBING THE ARCHITECTURE
#
######################################################################################

class Net(nn.Module):
    def __init__(self, param):
        super().__init__()

        # Parameters
        self.n_draws = param["N_DRAWS"]
        self.n_types = param["N_TYPES"]
        self.input_LSTM = param["INPUT_LSTM"]
        self.output_LSTM = param["OUTPUT_LSTM"]
        self.lstm_layers = param["LSTM_LAYERS"]
        self.batch_size = param["BATCH_SIZE"]
        self.height = param["HEIGHT"]
        self.width = param["WIDTH"]
        self.n_params = param["N_PARAMS"]

        # --------------------------------------------------------------------------------------
        # CNN

        # input: (n_types, Height, Width)
        self.conv1 = nn.Conv2d(self.n_types, 64, kernel_size=3, padding=1)
        # size: (64, Height, Width)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # size: (64, Height/2, Width/2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # size: (128, Height/2, Width/2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # size: (128, Height/4, Width/4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # size: (256, Height/4, Width/4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # size: (256, Height/8, Width/8)
        # size: (4 x Height x Width)
        self.linear1 = nn.Linear(4 * self.height * self.width, self.input_LSTM)
        # size: (self.input_LSTM)

        # --------------------------------------------------------------------------------------
        # LSTM

        # size: input_LSTM
        self.lstm = nn.LSTM(self.input_LSTM, self.output_LSTM, batch_first=True, num_layers=self.lstm_layers)
        # size: output_LSTM

        # --------------------------------------------------------------------------------------
        # FULLY CONNECTED

        # size: (n_draws, output_LSTM)
        # size: (n_draws x output_LSTM)
        self.linear2 = nn.Linear(self.n_draws * self.output_LSTM, self.n_params)
        # size: (n_params)

    def forward(self, x):
        # (Batch Size, n_draws, n_types, Height,  Width)
        x = x.view(self.batch_size * self.n_draws, self.n_types, self.height, self.width)
        # (Batch Size x n_draws, n_types, Height,  Width)
        x = F.relu(self.conv1(x))
        # (Batch Size x n_draws, 64, Height,  Width)
        x = self.pool1(x)
        # (Batch Size x n_draws, 64, Height/2,  Width/2)
        x = F.relu(self.conv2(x))
        # (Batch Size x n_draws, 128, Height / 2, Width / 2)
        x = self.pool2(x)
        # (Batch Size x n_draws, 128, Height / 4, Width / 4)
        x = F.relu(self.conv3(x))
        # (Batch Size x n_draws, 256, Height / 4, Width / 4)
        x = self.pool3(x)
        # (Batch Size x n_draws, 256, Height / 8, Width / 8)
        x = x.view(self.batch_size, self.n_draws, 4 * self.height * self.width)
        # (Batch Size, n_draws, 4 x Height x Width)
        x = F.relu(self.linear1(x))
        # (Batch Size, n_draws, input_LSTM)
        x, (h, c) = self.lstm(x)
        # (Batch Size, n_draws, output_LSTM)
        x = x.reshape(self.batch_size, self.n_draws * self.output_LSTM)
        # (Batch Size, n_draws * output_LSTM)
        x = F.sigmoid(self.linear2(x))
        # (Batch Size, n_params)
        return x

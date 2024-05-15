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
        self.convolution_layers = param["CONV_LAYERS"]
        self.feed_forward_size = param["FEED_FORWARD"]

        if self.lstm_layers == 0:
            self.input_LSTM = self.output_LSTM

        # --------------------------------------------------------------------------------------
        # CNN

        self.sequential_convolution = nn.Sequential()
        for i in range(self.convolution_layers):
            if i == 0:
                self.sequential_convolution.append(nn.Conv2d(self.n_types, 64, kernel_size=3, padding=1))
            else:
                self.sequential_convolution.append(
                    nn.Conv2d(64 * (2 ** (i - 1)), 64 * (2 ** i), kernel_size=3, padding=1))
            self.sequential_convolution.append(nn.ReLU())
            self.sequential_convolution.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # size: (16/2^(conv_layer-1) x Height x Width)
        self.linear1 = nn.Linear(int(16 / (2 ** (self.convolution_layers - 1))) * self.height * self.width,
                                 self.input_LSTM)
        # size: (input_LSTM)

        # --------------------------------------------------------------------------------------
        # LSTM

        # size: (input_LSTM)
        if self.lstm_layers > 0:
            self.lstm = nn.LSTM(self.input_LSTM, self.output_LSTM, batch_first=True, num_layers=self.lstm_layers)
        # size: (output_LSTM)

        # --------------------------------------------------------------------------------------
        # FULLY CONNECTED

        # size: (n_draws, output_LSTM)
        # size: (n_draws x output_LSTM)
        self.sequential_linear = nn.Sequential()
        if self.feed_forward_size > 0:
            self.sequential_linear.append(nn.Linear(self.n_draws * self.output_LSTM, self.feed_forward_size))
            self.sequential_linear.append(nn.ReLU())
        self.sequential_linear.append(nn.Linear(self.n_draws * self.output_LSTM if self.feed_forward_size == 0 else self.feed_forward_size, self.n_params))
        # size: (n_params)

    def forward(self, x):
        # (Batch Size, n_draws, n_types, Height,  Width)
        self.batch_size = x.shape[0]
        x = x.view(self.batch_size * self.n_draws, self.n_types, self.height, self.width)
        # (Batch Size x n_draws, n_types, Height,  Width)
        x = self.sequential_convolution(x)
        # (Batch Size x n_draws, 64*(2^conv_layer-1) , Height/(2^conv_layer),  Width/(2^conv_layer))
        x = x.view(self.batch_size, self.n_draws,
                   int(16 / (2 ** (self.convolution_layers - 1))) * self.height * self.width)
        # (Batch Size x n_draws, 16/2^(conv_layer-1) x Height x Width)
        x = F.relu(self.linear1(x))
        # (Batch Size, n_draws, input_LSTM)
        if self.lstm_layers > 0:
            x, (h, c) = self.lstm(x)
        # (Batch Size, n_draws, output_LSTM)
        x = x.reshape(self.batch_size, self.n_draws * self.output_LSTM)
        # (Batch Size, n_draws * output_LSTM)
        x = F.sigmoid(self.sequential_linear(x))
        # (Batch Size, n_params)
        return x

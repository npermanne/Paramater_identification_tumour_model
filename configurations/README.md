# Configurations
The configuration files are all `yaml` files that describe the dataset used, the model used and the training parameter.
Below are all the configuration parameter:

- `NAME`: Name of the result folder
- `DATASET`:
  - `FOLDER_NAME`: Name of the dataset folder
  - `N_DRAWS`: Number of draws used in the model (for example 4 => {350 , 450 , 550 , 650})
  - `IMG_TYPES`: List of images types (e.g. cells_types, cells_densities, oxygen, glucose)
  - `PARAMETERS_OF_INTEREST`: List of all the parameter predicted 
- `MODEL`:
  - `INPUT_LSTM`: Size of the vector before the LSTM block
  - `OUTPUT_LSTM`: Size of the vector after the LSTM block
  - `LSTM_LAYER`: Number of LSTM layers
- `TRAINING`:
  - `LEARNING_RATE`: Not use
  - `EPOCH`: Number of epoch of training
  - `BATCH_SIZE`: Batch size for training
  - `DEVICE`: cpu or cuda
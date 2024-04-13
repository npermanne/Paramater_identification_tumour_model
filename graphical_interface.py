from tkinter import *
from tkinter.ttk import Combobox, Progressbar, Style
from PIL import ImageTk, Image, ImageDraw
from networks.model import Network
import pandas as pd
import threading

import os
import yaml

# All images types:
ALL_IMAGES_TYPES = ["cells_types", "cells_densities", "oxygen", "glucose"]
ALL_PARAMETERS = ["sources", "average_healthy_glucose_absorption", "average_cancer_glucose_absorption",
                  "average_healthy_oxygen_consumption", "average_cancer_oxygen_consumption",
                  "quiescent_multiplier", "critical_multiplier", "radiosensitivity_G1",
                  "radiosensitivity_S", "radiosensitivity_G2", "radiosensitivity_M", "radiosensitivity_G0",
                  "source_glucose_supply", "source_oxygen_supply", "glucose_diffuse_rate",
                  "oxygen_diffuse_rate", "h_cells", "cell_cycle"]
FONT_TITLE = ("Verdana", 15, "bold")
FONT_TABLE = ("Verdana", 12)
VALUES_LEARNING_RATE = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
BLEU = ["#9AB6D3","#011E41","#5EB3E4"]
CONFIGURATION_COLOR = BLEU[1]
DATASET_TITLE_COLOR = BLEU[0]
DATASET_BODY_COLOR = BLEU[0]
MODEL_TITLE_COLOR = BLEU[0]
MODEL_BODY_COLOR = BLEU[0]
LEARNING_TITLE_COLOR = BLEU[0]
LEARNING_BODY_COLOR = BLEU[0]
RESULT_COLOR = BLEU[0]
CONTOUR = BLEU[1]
BACKGROUND_TEXT = BLEU[0]
BACKGROUND_COMBO = "black"
RUN = BLEU[2]


# Custom Slider
class CustomScale(Scale):
    def __init__(self, master=None, variable=None, allowedValue=None, command=None, **kw):
        if allowedValue is None:
            super().__init__(master=master, variable=variable, command=command, **kw)
        else:
            self.variable = variable
            self.allowedValue = allowedValue

            def update_value(new_value):
                new_value = int(new_value)
                self.variable.set(min(self.allowedValue, key=lambda x: abs(x - new_value)))
                command(new_value)

            super().__init__(master=master, variable=self.variable, command=update_value, **kw)


# Window
window = Tk()
window.title("Parameter identification of a tumour model")
window.geometry('1920x1080')
window.config()

# General structure
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=5)
window.grid_rowconfigure(2, weight=6)
window.grid_columnconfigure(0, weight=1)

configuration_frame = Frame(window, highlightbackground=CONTOUR, highlightthickness=5, bg=CONFIGURATION_COLOR)
configuration_frame.grid(row=0, column=0, sticky="nesw")

parameters_frame = Frame(window)
parameters_frame.grid(row=1, column=0, sticky="nesw")

parameters_frame.grid_columnconfigure(0, weight=1)
parameters_frame.grid_columnconfigure(1, weight=1)
parameters_frame.grid_columnconfigure(2, weight=1)
parameters_frame.grid_rowconfigure(0, weight=1)

dataset_parameter_frame = Frame(parameters_frame, highlightbackground=CONTOUR, highlightthickness=5)
dataset_parameter_frame.grid(row=0, column=0, sticky="nesw")
dataset_parameter_frame.grid_rowconfigure(0, weight=1)
dataset_parameter_frame.grid_rowconfigure(1, weight=11)
dataset_parameter_frame.grid_columnconfigure(0, weight=1)

dataset_parameter_frame_title = Frame(dataset_parameter_frame, bg=DATASET_TITLE_COLOR)
dataset_parameter_frame_title.grid(row=0, column=0, sticky="nesw")
dataset_parameter_frame_title.grid_columnconfigure(0, weight=1)

dataset_parameter_frame_body = Frame(dataset_parameter_frame, bg=DATASET_BODY_COLOR)
dataset_parameter_frame_body.grid(row=1, column=0, sticky="nesw")

model_parameter_frame = Frame(parameters_frame, highlightbackground=CONTOUR, highlightthickness=5)
model_parameter_frame.grid(row=0, column=1, sticky="nesw")
model_parameter_frame.grid_rowconfigure(0, weight=1)
model_parameter_frame.grid_rowconfigure(1, weight=11)
model_parameter_frame.grid_columnconfigure(0, weight=1)

model_parameter_frame_title = Frame(model_parameter_frame, bg=MODEL_TITLE_COLOR)
model_parameter_frame_title.grid(row=0, column=0, sticky="nesw")
model_parameter_frame_title.grid_columnconfigure(0, weight=1)

model_parameter_frame_body = Frame(model_parameter_frame, bg=MODEL_BODY_COLOR)
model_parameter_frame_body.grid(row=1, column=0, sticky="nesw")

learning_parameter_frame = Frame(parameters_frame, highlightbackground=CONTOUR, highlightthickness=5)
learning_parameter_frame.grid(row=0, column=2, sticky="nesw")
learning_parameter_frame.grid_rowconfigure(0, weight=1)
learning_parameter_frame.grid_rowconfigure(1, weight=11)
learning_parameter_frame.grid_columnconfigure(0, weight=1)

learning_parameter_frame_title = Frame(learning_parameter_frame, bg=LEARNING_TITLE_COLOR)
learning_parameter_frame_title.grid(row=0, column=0, sticky="nesw")
learning_parameter_frame_title.grid_columnconfigure(0, weight=1)

learning_parameter_frame_body = Frame(learning_parameter_frame, bg=LEARNING_BODY_COLOR)
learning_parameter_frame_body.grid(row=1, column=0, sticky="nesw")

results_frame = Frame(window, highlightbackground=CONTOUR, highlightthickness=5, bg=RESULT_COLOR)
results_frame.grid(row=2, column=0, sticky="nesw")
results_frame.grid_columnconfigure(0, weight=1)
results_frame.grid_columnconfigure(1, weight=1)
results_frame.grid_columnconfigure(2, weight=1)
results_frame.grid_rowconfigure(0, weight=1)
results_frame.grid_rowconfigure(1, weight=1)

# All variables
configuration = StringVar()
dataset_name = StringVar()
number_of_draws = IntVar()
images_types = {image_type: BooleanVar() for image_type in ALL_IMAGES_TYPES}
parameters_of_interest = {parameter: BooleanVar() for parameter in ALL_PARAMETERS}
input_lstm = IntVar()
output_lstm = IntVar()
lstm_layers = IntVar()
conv_layers = IntVar()
learning_rate = DoubleVar()
n_epochs = IntVar()
batch_size = IntVar()
iter_epoch = DoubleVar()


# Brain and logic
def config_change(event):
    selection = config_box.get()
    file = open(os.path.join("configurations", f"{selection}.yaml"))
    config = yaml.safe_load(file)
    dataset_name.set(config["DATASET"]["FOLDER_NAME"])
    number_of_draws.set(config["DATASET"]["N_DRAWS"])
    for image_type in ALL_IMAGES_TYPES:
        images_types[image_type].set(image_type in config["DATASET"]["IMG_TYPES"])
    for param in ALL_PARAMETERS:
        parameters_of_interest[param].set(param in config["DATASET"]["PARAMETERS_OF_INTEREST"])
    input_lstm.set(config["MODEL"]["INPUT_LSTM"])
    output_lstm.set(config["MODEL"]["OUTPUT_LSTM"])
    lstm_layers.set(config["MODEL"]["LSTM_LAYERS"])
    conv_layers.set(config["MODEL"]["CONV_LAYERS"])
    learning_rate.set(config["TRAINING"]["LEARNING_RATE"])
    n_epochs.set(config["TRAINING"]["EPOCH"])
    batch_size.set(config["TRAINING"]["BATCH_SIZE"])


def onValueChange(*event):
    selection = config_box.get()
    if selection in [file[:-5] for file in os.listdir("configurations") if file.endswith(".yaml")]:
        file = open(os.path.join("configurations", f"{selection}.yaml"))
        config = yaml.safe_load(file)
        cond = True
        cond &= config["DATASET"]["FOLDER_NAME"] == dataset_name.get()
        cond &= config["DATASET"]["N_DRAWS"] == number_of_draws.get()
        for image_type in ALL_IMAGES_TYPES:
            cond &= ((image_type in config["DATASET"]["IMG_TYPES"]) == images_types[image_type].get())
        for param in ALL_PARAMETERS:
            cond &= ((param in config["DATASET"]["PARAMETERS_OF_INTEREST"]) == parameters_of_interest[param].get())
        cond &= config["MODEL"]["INPUT_LSTM"] == input_lstm.get()
        cond &= config["MODEL"]["OUTPUT_LSTM"] == output_lstm.get()
        cond &= config["MODEL"]["LSTM_LAYERS"] == lstm_layers.get()
        cond &= config["MODEL"]["CONV_LAYERS"] == conv_layers.get()
        cond &= config["TRAINING"]["LEARNING_RATE"] == learning_rate.get()
        cond &= config["TRAINING"]["EPOCH"] == n_epochs.get()
        cond &= config["TRAINING"]["BATCH_SIZE"] == batch_size.get()
        if not cond:
            config_box.set("new config")


def run():
    global curve_image
    global architecture_text
    global table_text
    selection = config_box.get()
    if selection in [file[:-5] for file in os.listdir("configurations") if file.endswith(".yaml")]:
        # Retrieve data
        file = open(os.path.join("configurations", f"{selection}.yaml"))
        config = yaml.safe_load(file)
        config["NAME"] = selection
        # Show architecture
        network = Network(config)
        architecture_text = network.__str__()
        architecture_text = '\n'.join(architecture_text.split('\n')[:-6])
        architecture_canvas.itemconfig(architecture_description, text=architecture_text)

        # Show image
        curve_image = Image.open(os.path.join("results", selection, "performance_curve.png")).resize((600, 400))
        curve_image = ImageTk.PhotoImage(curve_image)
        curve_canvas.itemconfig(curve, image=curve_image)

        # Show table
        df = pd.read_csv(os.path.join("results", selection, "test_data.csv"), index_col=False)
        table_text = df.__str__()
        table_canvas.itemconfig(table, text=table_text)
    else:
        # Generate new config
        config = {"DATASET": {}, "MODEL": {}, "TRAINING": {}}
        config["DATASET"]["FOLDER_NAME"] = dataset_name.get()
        config["DATASET"]["N_DRAWS"] = number_of_draws.get()
        config["DATASET"]["IMG_TYPES"] = [image_type for image_type in ALL_IMAGES_TYPES if
                                          images_types[image_type].get()]
        config["DATASET"]["PARAMETERS_OF_INTEREST"] = [param for param in ALL_PARAMETERS if
                                                       parameters_of_interest[param].get()]

        config["MODEL"]["INPUT_LSTM"] = input_lstm.get()
        config["MODEL"]["OUTPUT_LSTM"] = output_lstm.get()
        config["MODEL"]["LSTM_LAYERS"] = lstm_layers.get()
        config["MODEL"]["CONV_LAYERS"] = conv_layers.get()

        config["TRAINING"]["LEARNING_RATE"] = learning_rate.get()
        config["TRAINING"]["EPOCH"] = n_epochs.get()
        config["TRAINING"]["BATCH_SIZE"] = batch_size.get()
        config["TRAINING"]["DEVICE"] = 'cpu'

        # Write it
        with open(os.path.join("configurations", f"{selection}.yaml"), 'w') as file:
            yaml.dump(config, file)

        config_box["values"] = [file[:-5] for file in os.listdir("configurations") if file.endswith(".yaml")]

        # Add the name argument
        config["NAME"] = selection

        # Show architecture
        network = Network(config)
        architecture_text = network.__str__()
        architecture_text = '\n'.join(architecture_text.split('\n')[:-6])
        architecture_canvas.itemconfig(architecture_description, text=architecture_text)

        # Remove performance curve
        curve_image = None
        curve_canvas.itemconfig(curve, image=curve_image)

        # Remove table
        table_text = ""
        table_canvas.itemconfig(table, text=table_text)

        # Train architecture
        def function():
            global curve_image
            network.train(iter_epoch)
            network.evaluate()
            # Show image
            curve_image = Image.open(os.path.join("results", selection, "performance_curve.png")).resize((600, 400))
            curve_image = ImageTk.PhotoImage(curve_image)
            curve_canvas.itemconfig(curve, image=curve_image)

            # Show table
            df = pd.read_csv(os.path.join("results", selection, "test_data.csv"), index_col=False)
            table_text = df.__str__()
            table_canvas.itemconfig(table, text=table_text)

            iter_epoch.set(0)
            return

        thread = threading.Thread(target=function, daemon=True)
        thread.start()


# Configurations Components
configuration_frame.grid_columnconfigure(0, weight=1)
configuration_frame.grid_columnconfigure(1, weight=1)
configuration_frame.grid_columnconfigure(2, weight=1)
configuration_frame.grid_columnconfigure(3, weight=1)
configuration_frame.grid_rowconfigure(0, weight=1)

Label(configuration_frame, text="Configuration:", bg=RUN).grid(row=0, column=0, sticky='e')
config_box = Combobox(configuration_frame, width=50,
                      values=[file[:-5] for file in os.listdir("configurations") if file.endswith(".yaml")],foreground=BACKGROUND_COMBO)
config_box.grid(row=0, column=1, sticky='w')
config_box.bind("<<ComboboxSelected>>", config_change)
Button(configuration_frame, text="Run", command=run, bg=RUN).grid(row=0, column=2, sticky='e')
Progressbar(configuration_frame, orient="horizontal", length=500, variable=iter_epoch).grid(row=0, column=3, sticky='w')

# Component Dataset
# Title
Label(dataset_parameter_frame_title, text="Dataset Parameters", font=FONT_TITLE, bg=BACKGROUND_TEXT).grid()

dataset_parameter_frame_body.grid_columnconfigure(0, weight=1)
dataset_parameter_frame_body.grid_columnconfigure(1, weight=1)
for i in range(len(ALL_PARAMETERS) + 3): dataset_parameter_frame_body.grid_rowconfigure(i, weight=1)
# Folder name
Label(dataset_parameter_frame_body, text="Folder name:", bg=BACKGROUND_TEXT).grid(row=0, column=0)
folderbox = Combobox(dataset_parameter_frame_body, textvariable=dataset_name, state='readonly', width=60, foreground=BACKGROUND_COMBO,
                     values=[folder for folder in os.listdir("datasets") if not os.path.isfile(folder)])
folderbox.grid(row=0, column=1)
folderbox.bind('<<ComboboxSelected>>', onValueChange)

# Number of draws
Label(dataset_parameter_frame_body, text="Number of draws:", bg=BACKGROUND_TEXT).grid(row=1, column=0)
Scale(dataset_parameter_frame_body, from_=1, to=8, orient='horizontal', variable=number_of_draws,
      command=onValueChange, background=BACKGROUND_TEXT).grid(row=1, column=1)

# Images types
Label(dataset_parameter_frame_body, text="Images types:", bg=BACKGROUND_TEXT).grid(row=2, column=0)
for i, image_type in enumerate(ALL_IMAGES_TYPES):
    Checkbutton(dataset_parameter_frame_body, text=image_type, variable=images_types[image_type],
                command=onValueChange, bg=BACKGROUND_TEXT,highlightbackground=DATASET_BODY_COLOR).grid(row=3 + i,
                                                                column=0,
                                                                sticky='w')

# Parameters of interest
Label(dataset_parameter_frame_body, text="Parameters of interests:", bg=BACKGROUND_TEXT).grid(row=2, column=1)
for i, parameter in enumerate(ALL_PARAMETERS):
    Checkbutton(dataset_parameter_frame_body, text=parameter, variable=parameters_of_interest[parameter],
                command=onValueChange, bg=BACKGROUND_TEXT,highlightbackground=DATASET_BODY_COLOR).grid(
        row=3 + i, column=1, sticky='w')

# Component Model
# Title
Label(model_parameter_frame_title, text="Model Parameters", font=FONT_TITLE, bg=BACKGROUND_TEXT).grid()  # Model title

model_parameter_frame_body.grid_columnconfigure(0, weight=1)
model_parameter_frame_body.grid_columnconfigure(1, weight=1)
for i in range(4): model_parameter_frame_body.grid_rowconfigure(i, weight=1)

# Inputs LSTM
Label(model_parameter_frame_body, text="Inputs LSTM:", bg=BACKGROUND_TEXT).grid(row=0, column=0)
CustomScale(model_parameter_frame_body, allowedValue=[1, 10, 100, 200, 500, 1000], from_=1, to=1000,
            orient='horizontal', variable=input_lstm, command=onValueChange, bg=BACKGROUND_TEXT).grid(row=0, column=1)

# Outputs LSTM
Label(model_parameter_frame_body, text="Outputs LSTM:", bg=BACKGROUND_TEXT).grid(row=1, column=0)
CustomScale(model_parameter_frame_body, allowedValue=[1, 10, 100, 200, 500, 1000], from_=1, to=1000,
            orient='horizontal', variable=output_lstm, command=onValueChange, bg=BACKGROUND_TEXT).grid(row=1, column=1)

# Number of LSTM layers
Label(model_parameter_frame_body, text="Numbers of LSTM Layers:", bg=BACKGROUND_TEXT).grid(row=2, column=0)
Scale(model_parameter_frame_body, from_=1, to=5, orient='horizontal', variable=lstm_layers, command=onValueChange, bg=BACKGROUND_TEXT).grid(
    row=2, column=1)

# Number of convolution layers
Label(model_parameter_frame_body, text="Numbers of Convolution Layers:", bg=BACKGROUND_TEXT).grid(row=3, column=0)
Scale(model_parameter_frame_body, from_=1, to=4, orient='horizontal', variable=conv_layers, command=onValueChange, bg=BACKGROUND_TEXT).grid(
    row=3, column=1)

# Component Learning
# Title
Label(learning_parameter_frame_title, text="Learning Parameters", font=FONT_TITLE, bg=BACKGROUND_TEXT).grid()

learning_parameter_frame_body.grid_columnconfigure(0, weight=1)
learning_parameter_frame_body.grid_columnconfigure(1, weight=1)
for i in range(3): learning_parameter_frame_body.grid_rowconfigure(i, weight=1)

# Learning rate
Label(learning_parameter_frame_body, text="Learning rate:", bg=BACKGROUND_TEXT).grid(row=0, column=0)
learning_rate_box = Combobox(learning_parameter_frame_body, values=VALUES_LEARNING_RATE, state='readonly',
                             textvariable=learning_rate,background=BACKGROUND_COMBO)
learning_rate_box.grid(row=0, column=1)
learning_rate_box.bind('<<ComboboxSelected>>', onValueChange)

# Number of epochs
Label(learning_parameter_frame_body, text="Numbers of epochs:", bg=BACKGROUND_TEXT).grid(row=1, column=0)
CustomScale(learning_parameter_frame_body, allowedValue=[1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200], from_=1, to=200,
            orient='horizontal', variable=n_epochs, command=onValueChange, bg=BACKGROUND_TEXT).grid(row=1, column=1)

# Batch Size
Label(learning_parameter_frame_body, text="Batch Size:", bg=BACKGROUND_TEXT).grid(row=2, column=0)
CustomScale(learning_parameter_frame_body, allowedValue=[1, 2, 4, 8, 16, 32, 64], from_=1, to=64,
            orient='horizontal', variable=batch_size, command=onValueChange, bg=BACKGROUND_TEXT).grid(row=2, column=1)

# Results
# Architecture
Label(results_frame, text="Architecture:", font=FONT_TITLE, bg=BACKGROUND_TEXT).grid(row=0, column=0)
architecture_canvas = Canvas(results_frame, width=600, height=400, bg=BACKGROUND_TEXT, highlightbackground=CONTOUR)
architecture_canvas.grid(row=1, column=0)
architecture_description = architecture_canvas.create_text((520, 200), text=None)

# Performance curve
Label(results_frame, text="Performance Curve:", font=FONT_TITLE, bg=BACKGROUND_TEXT).grid(row=0, column=1)
curve_canvas = Canvas(results_frame, width=600, height=400, bg=BACKGROUND_TEXT, highlightbackground=CONTOUR)
curve_canvas.grid(row=1, column=1)
curve = curve_canvas.create_image((300, 200), image=None)

# Test data
Label(results_frame, text="Test results:", font=FONT_TITLE, bg=BACKGROUND_TEXT).grid(row=0, column=2)
table_canvas = Canvas(results_frame, width=600, height=400, bg=BACKGROUND_TEXT, highlightbackground=CONTOUR)
table_canvas.grid(row=1, column=2)
table = table_canvas.create_text((300, 100), text=None, font=FONT_TABLE)

window.mainloop()

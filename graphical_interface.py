import sys
import os
from email.policy import default

sys.path.append("simulation")
from tkinter import StringVar
from tkinter import IntVar
import pandas as pd
import customtkinter as ctk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from simulation.simulation import Simulation
import pickle
import random
import matplotlib
import matplotlib.ticker as mticker

ctk.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

# COLOR
DARK_BLUE = "#00539D"
LIGHT_BLUE = "#4E91CD"


# PARAM
SIDEBAR_SIZE = 400
SIDEBAR_ELEMENT_SIZE = 275
SPEED = 10
COLORS = ["#FA1818", "black", "#00FA66"]
CMAP = matplotlib.colors.ListedColormap(COLORS)
NORM = matplotlib.colors.Normalize(vmin=-1, vmax=1)

# CONST
HYPERPARAMETERS = {
    "INPUT_LSTM": "Size of the encoded vector that goes into each LSTM blocks",
    "OUTPUT_LSTM": "Size of the output of each LSTM blocks",
    "LSTM_LAYERS": "Number of LSTM layers",
    "CONV_LAYERS": "Number of alternating layers of convolution and max-pooling",
    "LEARNING_RATE": "Learning Rate",
    "BATCH_SIZE": "Batch Size",
    "L2_REGULARIZATION": "L2 Regularization",
    "FEED_FORWARD": "Number of neurons in the final hidden layer"
}

parameters = {
    "cell_cycle": "Duration of the cell cycle $t_{total}$",
    "average_healthy_glucose_absorption": "Average healthy glucose absorption $\mu_{g,healthy}$",
    "average_cancer_glucose_absorption": "Average cancer glucose absorption $\mu_{g,cancer}$",
    "average_healthy_oxygen_consumption": "Average healthy oxygen consumption $\mu_{o,healthy}$",
    "average_cancer_oxygen_consumption": "Average cancer oxygen consumption $\mu_{o,cancer}$"
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.dataset = StringVar()
        self.dataset.trace('w', self.update_model)
        self.draws = IntVar()
        self.draws.trace('w', self.update_model)
        self.model_number = StringVar()
        self.model_number.trace('w', self.update_model)

        # configure window
        self.title("Interface Nico")

        screen_width = self.winfo_screenwidth()  # Get the width of the screen
        screen_height = self.winfo_screenheight()  # Get the height of the screen
        taskbar_height = screen_height - self.winfo_rooty()  # Get the height of the sidebar
        self.geometry("%dx%d+0+0" % (screen_width, screen_height))

        # SIDEBAR

        self.sidebar = Sidebar(self, self.dataset, self.draws, self.model_number)
        self.sidebar.place(relx=1, rely=0.5, anchor="e")

        # TITLE
        self.title = ctk.CTkLabel(self, text="Parameter identification of a tumour growth model: Performance visualisation", font=ctk.CTkFont(size=25, weight="bold", underline=True), text_color=DARK_BLUE)
        self.title.place(relx=0.4, rely=0.01, anchor="n")

        # EPL LOGO
        epl = ctk.CTkImage(light_image=Image.open(os.path.join("pictures", "EPL-logo-with-text.jpg")), dark_image=Image.open(os.path.join("pictures", "EPL-logo-with-text.jpg")), size=(SIDEBAR_SIZE, int(SIDEBAR_SIZE / 4.2)))
        button_epl = ctk.CTkButton(self, text='', image=epl, fg_color='transparent', hover=False)
        button_epl.place(relx=1, rely=1, anchor='se')

        # AUTHORS
        self.author_label = ctk.CTkLabel(self, text="Author: Nicolas Permanne")
        self.author_label.place(relx=0.01, rely=.975, anchor='sw')

        self.supervisor_label = ctk.CTkLabel(self, text='Supervisors: Mélanie Ghislain, Manon Dausort, Florian Martin, Benoît Macq')
        self.supervisor_label.place(relx=0.01, rely=1.0, anchor='sw')

        # TABVIEW VISUALISER
        self.tabview = ctk.CTkTabview(self, width=1470, height=900, segmented_button_selected_color=LIGHT_BLUE)
        self.visualiser1 = Visualiser1(master=self.tabview.add("Visualiser 1"))
        self.visualiser2 = Visualiser2(master=self.tabview.add("Visualiser 2"))
        self.visualiser3 = Visualiser3(master=self.tabview.add("Visualiser 3"))

        self.tabview.place(relx=0.4, rely=0.5, anchor="center")

    def update_model(self, *args):
        if self.draws.get() != 0 and self.dataset.get() != '' and self.model_number.get() != '':
            dataset_name = {
                "Without treatment": "no_dose",
                "With baseline treatment": "baseline_treatment",
                "With RL treatment": "best_model_treatment",
            }[self.dataset.get()]
            model_number = int(self.model_number.get().split('~')[0][5:])
            path = os.path.join("results", f"hyp_search_{dataset_name}_for_{self.draws.get()}_draws", str(model_number))
            self.visualiser1.update_visualiser(path)
            self.visualiser2.update_visualiser(path, self.dataset.get())


class Visualiser1(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

    def update_visualiser(self, path):
        fig, ax = plt.subplots(1, 1, figsize=(24, 15))
        ax.clear()

        f = lambda x: textwrap.fill(x, 20)

        df = pd.read_csv(os.path.join(path, "evaluation_data.csv"))
        data = {key: np.abs(np.array(df[f"predicted_{key}"]) - np.array(df[f"true_{key}"])) for key, value in parameters.items()}

        ax.violinplot(data.values(), vert=False)
        ax.set_yticks(ticks=range(1, len(data) + 1), labels=[f(parameters[a]) for a in data.keys()])

        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().config(highlightthickness=0, borderwidth=0)
        canvas.get_tk_widget().place(relx=0.5, rely=0.5, relwidth=0.95, relheight=0.95, anchor="center")
        plt.close()


class Visualiser2(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.title = ctk.CTkLabel(master=self.master, text="2 simulations separated by the error of prediction of the model", width=250, height=40, font=ctk.CTkFont(size=20, underline=True))
        self.title.place(relx=0.5, rely=0, anchor="n")

        self.pause_button = ctk.CTkButton(master=self.master, width=50, height=50, text="Resume", hover=True, fg_color="red", command=self.pause_action)
        self.pause_button.place(relx=0.5, rely=1, anchor="s")

        self.predicted_simu = None
        self.true_simu = None
        self.pause = False
        self.hours_passed = 0

        self.predicted_fig, self.predicted_ax, self.true_fig, self.true_ax, self.predicted_cbar, self.true_cbar = None, None, None, None, None, None

    def simulate(self):
        self.predicted_fig, self.predicted_ax = plt.subplots(2, 2, figsize=(20, 20))
        self.true_fig, self.true_ax = plt.subplots(2, 2, figsize=(20, 20))
        self.predicted_cbar = np.array([[None for i in range(2)] for j in range(2)])
        self.true_cbar = np.array([[None for i in range(2)] for j in range(2)])
        titles = {(0, 0): "Cell cycle", (0, 1): "Cell density", (1, 0): "Glucose", (1, 1): "Oxygen"}
        for i in range(2):
            for j in range(2):
                self.true_ax[i, j].axis('off')
                self.predicted_ax[i, j].axis('off')
                self.true_ax[i, j].set_title(titles[(i, j)])
                self.predicted_ax[i, j].set_title(titles[(i, j)])
                cmap = CMAP if (i, j) == (0, 0) else "plasma" if (i, j) == (1, 0) else "cividis" if (i, j) == (1, 1) else "viridis"
                norm = NORM if (i, j) == (0, 0) else "linear"
                predicted_img = self.predicted_ax[i, j].imshow(np.random.random((64, 64)), cmap=cmap, norm=norm)
                true_img = self.true_ax[i, j].imshow(np.random.random((64, 64)), cmap=cmap, norm=norm)
                if (i, j) != (0, 0):
                    self.predicted_cbar[i, j] = self.predicted_fig.colorbar(predicted_img, orientation='horizontal', fraction=0.046, pad=0.04)
                    self.true_cbar[i, j] = self.true_fig.colorbar(true_img, orientation='horizontal', fraction=0.046, pad=0.04)
                else:
                    self.predicted_cbar[i, j] = self.predicted_fig.colorbar(predicted_img, orientation='horizontal', ticks=[-0.66, 0, 0.66], format=mticker.FixedFormatter(['Tumour cell', 'Empty', 'Healthy cell']), fraction=0.046, pad=0.04)
                    self.true_cbar[i, j] = self.true_fig.colorbar(true_img, orientation='horizontal', ticks=[-0.66, 0, 0.66], format=mticker.FixedFormatter(['Tumour cell', 'Empty', 'Healthy cell']), fraction=0.046, pad=0.04)

        canvas_predicted = FigureCanvasTkAgg(self.predicted_fig, master=self.master)
        canvas_predicted.draw()
        canvas_predicted.get_tk_widget().place(relx=0.25, rely=0.5, relwidth=0.5, relheight=0.85, anchor="center")
        canvas_true = FigureCanvasTkAgg(self.true_fig, master=self.master)
        canvas_true.draw()
        canvas_true.get_tk_widget().place(relx=0.75, rely=0.5, relwidth=0.5, relheight=0.85, anchor="center")
        plt.close()

        self.update()

    def update(self):
        if not self.pause:
            types = np.array([simu.get_cells_type(color=False) for simu in [self.predicted_simu, self.true_simu]])
            densities = np.array([simu.get_cells_density() for simu in [self.predicted_simu, self.true_simu]])
            dmin, dmax = np.min(densities), np.max(densities)
            glucoses = np.array([simu.get_glucose() for simu in [self.predicted_simu, self.true_simu]])
            gmin, gmax = np.min(glucoses), np.max(glucoses)
            oxygens = np.array([simu.get_oxygen() for simu in [self.predicted_simu, self.true_simu]])
            omin, omax = np.min(oxygens), np.max(oxygens)

            for index, axes, cbar in zip(range(2), [self.predicted_ax, self.true_ax], [self.predicted_cbar, self.true_cbar]):
                for i, j in [(i, j) for i in range(2) for j in range(2)]:
                    axes[i, j].clear()
                    axes[i, j].axis('off')

                im_type = axes[0, 0].imshow(types[index], cmap=CMAP, norm=NORM)
                axes[0, 0].set_title("Cell cycle")

                im_density = axes[0, 1].imshow(densities[index], vmin=dmin, vmax=dmax)
                axes[0, 1].set_title("Cell density")
                cbar[0, 1].mappable.set_clim(vmin=dmin, vmax=dmax)

                im_glucose = axes[1, 0].imshow(glucoses[index], cmap="plasma", vmin=gmin, vmax=gmax)
                axes[1, 0].set_title("Glucose")
                cbar[1, 0].mappable.set_clim(vmin=gmin, vmax=gmax)

                im_oxygen = axes[1, 1].imshow(oxygens[index], cmap="cividis", vmin=omin, vmax=omax)
                axes[1, 1].set_title("Oxygen")
                cbar[1, 1].mappable.set_clim(vmin=omin, vmax=omax)

            self.predicted_fig.suptitle(f"Simulation A at {self.hours_passed} hours")
            self.true_fig.suptitle(f"Simulation B at {self.hours_passed} hours")
            self.predicted_fig.tight_layout()
            self.true_fig.tight_layout()
            self.predicted_fig.canvas.draw()
            self.true_fig.canvas.draw()

            self.predicted_simu.cycle(5)
            self.true_simu.cycle(5)
            self.hours_passed += 5
        self.master.after(SPEED, self.update)

    def pause_action(self):
        if self.pause:
            self.pause = False
            self.pause_button.configure(fg_color="green")
            self.pause_button.configure(text="Pause")
        else:
            self.pause = True
            self.pause_button.configure(fg_color="red")
            self.pause_button.configure(text="Resume")

    def update_visualiser(self, path, dataset):
        stats = pd.read_csv(os.path.join(path, "evaluation_stats.csv"))
        ranges = pd.read_csv(os.path.join("simulation", "parameter_data.csv"), index_col=0)

        predicted_param = dict()
        true_param = dict()
        for index, row in ranges.iterrows():
            default = row['Default Value']
            if index in parameters.keys():
                min = row['Minimum']
                max = row['Maximum']
                normalised_error = stats.loc[stats["Parameters"] == index, "Means"].values[0]

                error = normalised_error * (max - min)
                if index == 'cell_cycle':
                    predicted_param[index] = round(default - error / 2)
                    true_param[index] = round(default + error / 2)
                else:
                    predicted_param[index] = default - (error / 2)
                    true_param[index] = default + (error / 2)
            else:
                predicted_param[index] = default
                true_param[index] = default

        treatment = None
        if dataset == "With RL treatment":
            result_florian = pickle.load(open(os.path.join("simulation", "best_result_rl_Florian.pickle"), "rb"))
            best_treatments = list()
            for dose_per_hour in result_florian["doses_per_hour"].values():
                treatment = np.zeros(1100)
                for key, value in dose_per_hour.items():
                    treatment[350 + key] = value
                best_treatments.append(treatment)
            treatment = random.choice(np.array(best_treatments))
        elif dataset == "With baseline treatment":
            dose_hours = list(range(350, 1071, 24))
            baseline_treatment = np.zeros(1100)
            baseline_treatment[dose_hours] = 2
            treatment = np.array(baseline_treatment)

        self.predicted_simu = Simulation(64, 64, predicted_param, treatment_planning=treatment)
        self.true_simu = Simulation(64, 64, true_param, treatment_planning=treatment)
        self.hours_passed = 0
        self.pause = True
        self.pause_button.configure(fg_color="red")
        self.pause_button.configure(text="Resume")

        if self.predicted_fig is None:
            self.simulate()


class Visualiser3(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)


class Sidebar(ctk.CTkFrame):
    def __init__(self, master, dataset, draws, model_number):
        super().__init__(master, width=SIDEBAR_SIZE, height=1080, fg_color=LIGHT_BLUE, corner_radius=0)

        # STRINGVAR
        self.dataset = dataset
        self.draws = draws
        self.model_number = model_number

        # TITLE
        self.title = ctk.CTkLabel(self, text="Model selection", font=ctk.CTkFont(size=20, weight="bold", underline=True), text_color="white")
        self.title.place(relx=0.5, rely=0.02, anchor="n")

        # DATASET
        self.dataset_label = ctk.CTkLabel(self, text="Dataset:", font=ctk.CTkFont(size=15), width=SIDEBAR_ELEMENT_SIZE)
        self.dataset_label.place(relx=0.5, rely=0.07, anchor="n")

        self.dataset_selection = ctk.CTkComboBox(self, values=["Without treatment", "With baseline treatment", "With RL treatment"], width=SIDEBAR_ELEMENT_SIZE, justify="center", state="readonly", variable=self.dataset, command=self.update_model)
        self.dataset_selection.place(relx=0.5, rely=0.1, anchor="n")

        # DRAWS
        self.draws_label_text = StringVar(value="Number of draws:")
        self.draws_label = ctk.CTkLabel(self, text="Number of draws:", font=ctk.CTkFont(size=15), width=SIDEBAR_ELEMENT_SIZE, textvariable=self.draws_label_text)
        self.draws_label.place(relx=0.5, rely=0.17, anchor="n")

        self.draws_slider = ctk.CTkSlider(self, from_=1, to=8, number_of_steps=8, width=SIDEBAR_ELEMENT_SIZE, progress_color=DARK_BLUE, variable=self.draws, command=self.update_model)
        self.draws_slider.place(relx=0.5, rely=0.2, anchor="n")

        # MODEL
        self.model_label = ctk.CTkLabel(self, text="Trained model:", font=ctk.CTkFont(size=15), width=SIDEBAR_ELEMENT_SIZE)
        self.model_label.place(relx=0.5, rely=0.25, anchor="n")

        self.model_selection = ctk.CTkComboBox(self, values=[], width=SIDEBAR_ELEMENT_SIZE, justify="center", state="readonly", variable=self.model_number, command=self.update_sidebar)
        self.model_selection.place(relx=0.5, rely=0.28, anchor="n")

        # HYPER-PARAMETERS
        self.hyperparameter_title = StringVar()
        self.hyperparameter_title_label = ctk.CTkLabel(self, font=ctk.CTkFont(size=18, weight="bold"), text_color="white", textvariable=self.hyperparameter_title)
        self.hyperparameter_title_label.place(relx=0.5, rely=0.4, anchor="center")

        self.hyperparameters_string = [StringVar() for _ in range(len(HYPERPARAMETERS))]
        self.hyperparameters_labels = [ctk.CTkLabel(self, font=ctk.CTkFont(size=15), width=SIDEBAR_ELEMENT_SIZE, textvariable=self.hyperparameters_string[i], wraplength=SIDEBAR_ELEMENT_SIZE) for i in range(len(HYPERPARAMETERS))]

        for i in range(len(HYPERPARAMETERS)):
            self.hyperparameters_labels[i].place(relx=0.5, rely=0.45 + i * 0.05, anchor="center")

    def update_sidebar(self, *args):
        if self.draws.get() != 0 and self.dataset.get() != '' and self.model_selection.get() != '':
            dataset_name = {
                "Without treatment": "no_dose",
                "With baseline treatment": "baseline_treatment",
                "With RL treatment": "best_model_treatment",
            }[self.dataset.get()]
            performances = pd.read_csv(os.path.join("results", f"hyp_search_{dataset_name}_for_{self.draws.get()}_draws", "performances.csv"), index_col=0)

            self.hyperparameter_title.set("Hyper-parameters:")
            model_number = int(self.model_number.get().split('~')[0][5:])
            for i, (param, param_name) in enumerate(HYPERPARAMETERS.items()):
                param_value = performances.iloc[model_number][param]
                self.hyperparameters_string[i].set(f"{param_name}: {param_value}")

    def update_model(self, *args):
        if self.draws.get() != 0:
            self.draws_label_text.set(f"Number of draws: {self.draws.get()}")

        if self.draws.get() != 0 and self.dataset.get() != '':
            dataset_name = {
                "Without treatment": "no_dose",
                "With baseline treatment": "baseline_treatment",
                "With RL treatment": "best_model_treatment",
            }[self.dataset.get()]
            performances = pd.read_csv(os.path.join("results", f"hyp_search_{dataset_name}_for_{self.draws.get()}_draws", "performances.csv"), index_col=0)
            perf_dico = sorted(dict(performances["Performance"]).items(), key=lambda a: a[1])
            self.model_selection.configure(values=[f"Model {key} ~ {int(value * 100)} %" for key, value in perf_dico])
            self.model_number.set(f"Model {perf_dico[0][0]} ~ {int(perf_dico[0][1] * 100)} %")
            self.update_sidebar()


if __name__ == '__main__':
    app = App()


    def on_closing():
        app.quit()
        app.destroy()


    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()

from tkinter import StringVar
from tkinter import IntVar

import customtkinter as ctk
from PIL import Image, ImageTk
import os

ctk.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

# COLOR
DARK_BLUE = "#00539D"
LIGHT_BLUE = "#4E91CD"

# SIZE
SIDEBAR_SIZE = 400
SIDEBAR_ELEMENT_SIZE = 275


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
        self.title = ctk.CTkLabel(self, text="Parameter identification of a tumour growth model: Performance visualisation", font=ctk.CTkFont(size=25, weight="bold"), text_color=DARK_BLUE)
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
        self.tabview.add("Visualiser 1")
        self.tabview.add("Visualiser 2")
        self.tabview.add("Visualiser 3")
        self.tabview.place(relx=0.4, rely=0.5, anchor="center")

    def update_model(self, *args):
        print(self.draws.get())
        print(self.dataset.get())
        print(self.model_number.get())


class Sidebar(ctk.CTkFrame):
    def __init__(self, master, dataset, draws, model_number):
        super().__init__(master, width=SIDEBAR_SIZE, height=1080, fg_color=LIGHT_BLUE, corner_radius=0)

        # TITLE
        self.title = ctk.CTkLabel(self, text="Model selection", font=ctk.CTkFont(size=20, weight="bold", underline=True), text_color="white")
        self.title.place(relx=0.5, rely=0.02, anchor="n")

        # DATASET
        self.dataset_label = ctk.CTkLabel(self, text="Dataset:", font=ctk.CTkFont(size=15), width=SIDEBAR_ELEMENT_SIZE)
        self.dataset_label.place(relx=0.5, rely=0.07, anchor="n")

        self.dataset_selection = ctk.CTkComboBox(self, values=["Without treatment", "With baseline treatment", "With RL treatment"], width=SIDEBAR_ELEMENT_SIZE, justify="center", state="readonly", variable=dataset)
        self.dataset_selection.place(relx=0.5, rely=0.1, anchor="n")

        # DRAWS
        self.draws_label_text = StringVar(value="Number of draws:")
        self.draws_label = ctk.CTkLabel(self, text="Number of draws:", font=ctk.CTkFont(size=15), width=SIDEBAR_ELEMENT_SIZE, textvariable=self.draws_label_text)
        self.draws_label.place(relx=0.5, rely=0.17, anchor="n")

        self.draws_slider = ctk.CTkSlider(self, from_=1, to=8, number_of_steps=8, width=SIDEBAR_ELEMENT_SIZE, progress_color=DARK_BLUE, variable=draws, command=lambda *args: self.draws_label_text.set(f"Number of draws: {draws.get()}"))
        self.draws_slider.place(relx=0.5, rely=0.2, anchor="n")

        # MODEL
        self.model_label = ctk.CTkLabel(self, text="Trained model:", font=ctk.CTkFont(size=15), width=SIDEBAR_ELEMENT_SIZE)
        self.model_label.place(relx=0.5, rely=0.25, anchor="n")

        self.model_selection = ctk.CTkComboBox(self, values=[str(i) for i in range(40)], width=SIDEBAR_ELEMENT_SIZE, justify="center", state="readonly", variable=model_number)
        self.model_selection.place(relx=0.5, rely=0.28, anchor="n")


app = App()
app.mainloop()

import customtkinter
import sys
# from tkinter import *
from PIL import ImageTk, Image
from tkinter_trials_trainer.text_redirector import TextRedirector


class TrainerApp:
    def __init__(self):
        self.root = customtkinter.CTk()
        self.root = customtkinter.CTkToplevel()
        # self.root.attributes('-fullscreen', True)
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")
        self.root.title("Trainer App - Im Classifier")
        self.root.maxsize(1800, 1200)  # width x height
        self.root.geometry("644x434")
        # self.root.config(bg="skyblue")
        self._create_initial_layout()

    def train_model(self):
        print('initialize training')

    def _create_initial_layout(self):
        # header = customtkinter.CTkLabel(self.root, text="Hello Tkinter!")
        # header.grid(row=0, column=2, rowspan=3)

        # log_frame = customtkinter.CTkLabel(self.root, text="log me!")
        # log_frame.grid(row=1, column=0, rowspan=6)

        # sample_frame = customtkinter.CTkLabel(self.root, text="trer!")
        # sample_frame.grid(row=2, column=2, rowspan=6)
        n_rows = 4
        n_cols = 2

        header = customtkinter.CTkLabel(self.root, text="Trainer Application")
        header.grid(row=0, column=0, columnspan=2, sticky='NESW')

        label2 = customtkinter.CTkLabel(self.root, text="dir")
        label2.grid(row=1, column=0, pady=5, sticky='W')

        label3 = customtkinter.CTkLabel(self.root, text="exp dir")
        label3.grid(row=1, column=1, pady=5, sticky='W')

        self.text = customtkinter.CTkTextbox(self.root, wrap="word")
        self.text.grid(row=2, column=0, columnspan=2, sticky='NESW')
        # self.text.pack(side="top", fill="both", expand=True)
        # self.text.tag_configure("stderr", foreground="#b22222")
        # train button
        self.train_button = customtkinter.CTkButton(self.root, text='Train', command=self.train_model)
        self.train_button.grid(row=3, column=0)

        sys.stdout = TextRedirector(self.text, "stdout")
        sys.stderr = TextRedirector(self.text, "stderr")

        for i in range(n_cols):
            self.root.grid_columnconfigure(i, weight=1)
        for i in range(n_rows):
            self.root.grid_rowconfigure(i, weight=1)
        print('testme')

    def main_loop(self):
        self.root.mainloop()

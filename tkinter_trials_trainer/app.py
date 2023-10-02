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
        # self.root.config(bg="skyblue")
        self._create_initial_layout()

    def _create_initial_layout(self):
        # header = customtkinter.CTkLabel(self.root, text="Hello Tkinter!")
        # header.grid(row=0, column=2, rowspan=3)

        # log_frame = customtkinter.CTkLabel(self.root, text="log me!")
        # log_frame.grid(row=1, column=0, rowspan=6)

        # sample_frame = customtkinter.CTkLabel(self.root, text="trer!")
        # sample_frame.grid(row=2, column=2, rowspan=6)

        header = customtkinter.CTkLabel(self.root, text="Trainer Application")
        header.grid(row=0, column=0, columnspan=8)

        label2 = customtkinter.CTkLabel(self.root, text="dir")
        label2.grid(row=1, column=0, columnspan=4, padx=50, pady=5)

        label3 = customtkinter.CTkLabel(self.root, text="exp dir")
        label3.grid(row=1, column=4, padx=50, pady=5)

        label4 = customtkinter.CTkLabel(self.root, text="R2/C0")
        label4.grid(row=2, column=0, sticky='e')

        self.text = customtkinter.CTkTextbox(self.root, wrap="word")
        self.text.grid(row=8, column=0, columnspan=8)
        # self.text.pack(side="top", fill="both", expand=True)
        # self.text.tag_configure("stderr", foreground="#b22222")

        sys.stdout = TextRedirector(self.text, "stdout")
        sys.stderr = TextRedirector(self.text, "stderr")

        print('testme')
        # Create left and right frames
        # left_frame = customtkinter.CTkFrame(self.root, width=200, height=400)
        # left_frame.grid(row=0, column=0, padx=10, pady=5)
        #
        # right_frame = customtkinter.CTkFrame(self.root, width=650, height=400)
        # right_frame.grid(row=0, column=1, padx=10, pady=5)
        #
        # # Create frames and labels in left_frame
        # customtkinter.CTkLabel(left_frame, text="Original Image").grid(row=0, column=0, padx=5, pady=5)
        # im_path = r'C:\data\bee_spotter\data\batch_00\000000.png'
        #
        # def clicked():
        #     print("Clicked.")
        #
        # tool_bar = customtkinter.CTkFrame(left_frame, width=90, height=185)
        # tool_bar.grid(row=1, column=0, padx=5, pady=5)
        #
        # filter_bar = customtkinter.CTkFrame(left_frame, width=90, height=185)
        # filter_bar.grid(row=1, column=1, padx=5, pady=5)
        #
        # customtkinter.CTkButton(tool_bar, text="Button 1", command=clicked).pack(anchor='n', padx=5, pady=3, ipadx=10)
        # customtkinter.CTkButton(filter_bar, text="Button 2", command=clicked).pack(anchor='n', padx=5, pady=3, ipadx=10)
        # # img = PhotoImage(file=im_path)
        # # customtkinter.CTkButton(master=customtkinter.CTkToplevel(), image=img).pack(side=LEFT)
        # # customtkinter.CTkButton(master=right_frame, image=img).pack(side=customtkinter.LEFT)
        # im_size = 800, 800
        # img = Image.open(im_path)
        # img = img.resize(im_size, Image.ANTIALIAS)
        #
        # self.large_test_image = customtkinter.CTkImage(img, size=im_size)
        # self.home_frame_large_image_label = customtkinter.CTkLabel(right_frame, text="", image=self.large_test_image)
        # self.home_frame_large_image_label.pack(side=customtkinter.LEFT)

    def main_loop(self):
        self.root.mainloop()

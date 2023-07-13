import customtkinter
# from tkinter import *
from PIL import ImageTk, Image


class ImageSorterApp:
    def __init__(self):
        self.root = customtkinter.CTk()
        self.root = customtkinter.CTkToplevel()
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")
        self.root.title("Basic GUI Layout with Grid")
        # self.root.maxsize(1800, 1200)  # width x height
        # self.root.config(bg="skyblue")
        self._create_initial_layout()

    def _create_initial_layout(self):
        # Create left and right frames
        left_frame = customtkinter.CTkFrame(self.root, width=200, height=400)
        left_frame.grid(row=0, column=0, padx=10, pady=5)

        right_frame = customtkinter.CTkFrame(self.root, width=650, height=400)
        right_frame.grid(row=0, column=1, padx=10, pady=5)

        # Create frames and labels in left_frame
        customtkinter.CTkLabel(left_frame, text="Original Image").grid(row=0, column=0, padx=5, pady=5)
        im_path = r'C:\data\bee_spotter\data\batch_00\000000.png'

        def clicked():
            print("Clicked.")

        tool_bar = customtkinter.CTkFrame(left_frame, width=90, height=185)
        tool_bar.grid(row=1, column=0, padx=5, pady=5)

        filter_bar = customtkinter.CTkFrame(left_frame, width=90, height=185)
        filter_bar.grid(row=1, column=1, padx=5, pady=5)

        customtkinter.CTkButton(tool_bar, text="Button 1", command=clicked).pack(anchor='n', padx=5, pady=3, ipadx=10)
        customtkinter.CTkButton(filter_bar, text="Button 2", command=clicked).pack(anchor='n', padx=5, pady=3, ipadx=10)
        # img = PhotoImage(file=im_path)
        # customtkinter.CTkButton(master=customtkinter.CTkToplevel(), image=img).pack(side=LEFT)
        # customtkinter.CTkButton(master=right_frame, image=img).pack(side=customtkinter.LEFT)
        im_size = 800, 800
        img = Image.open(im_path)
        img = img.resize(im_size, Image.ANTIALIAS)

        self.large_test_image = customtkinter.CTkImage(img, size=im_size)
        self.home_frame_large_image_label = customtkinter.CTkLabel(right_frame, text="", image=self.large_test_image)
        self.home_frame_large_image_label.pack(side=customtkinter.LEFT)

    def main_loop(self):
        self.root.mainloop()

import customtkinter
from tkinter import *
from PIL import ImageTk, Image

from image_sorting.app import ImageSorterApp

app = customtkinter.CTk()  # create CTk window like you do with the Tk window
app.geometry("400x240")


def open_img():
    im_path = r'C:\data\bee_spotter\data\batch_00\000000.png'
    img = Image.open(im_path)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(app, image=img)
    panel.image = img
    panel.pack()

# def main():
# customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
# customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
#
# def button_function():
#     print("button pressed")
#
# # Use CTkButton instead of tkinter Button
# button = customtkinter.CTkButton(master=app, text="CTkButton", command=open_img)
# button.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)
#
# app.mainloop()


if __name__ == '__main__':
    app = ImageSorterApp()
    app.main_loop()


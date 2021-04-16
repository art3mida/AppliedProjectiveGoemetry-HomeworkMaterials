import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from numpy import linalg as la
import algorithms as lib


class Main(object):

    def __init__(self):
        self.canvas = None
        self.photo = None
        self.image = None
        self.points = []
        self.points_proj = [[200,200,1], [400,200,1], [200,400,1], [400,400,1]]

    def main(self):
        master = Tk()

        right_frame = Frame(master, width=500, height=500, cursor="dot")
        right_frame.pack(side=LEFT)

        Button(right_frame, text = 'Ucitaj sliku', command = self.load_image).pack()

        self.canvas = Canvas(right_frame, width=800, height=700)
        self.canvas.create_image(0, 0, image=None, anchor="nw")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        mainloop()
    
    def load_image(self):
        path = fd.askopenfilename()
        if path != None:
            self.image = Image.open(path)
        else: 
            print("Morate selektovati validnu putanju!")

        self.image = self.image.resize((800, 600), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        mb.showinfo("Uputstvo","Selektujte 4 temena na istoj stranici objekta. Nakon toga ce biti uklonjena projektivna distorzija po toj strani.")

    def on_button_press(self, event):
        self.x = event.x
        self.y = event.y

    def on_button_release(self, event):
        red = "#FF0000"
        x0,y0 = (self.x, self.y)
        x1,y1 = (event.x, event.y)
        self.canvas.create_oval(x0, y0, x1, y1, fill=red, outline=red, width=10)
        self.points.append([x0,y0,1])

        if len(self.points) == 4:
            self.remove_distortion()
    
    def remove_distortion(self):
        P = lib.naive_algorithm(self.points, self.points_proj)

        P_inv = la.inv(P)

        width = 800
        height = 600
        new_image = Image.new('RGB', (width, height), color=1)
        for i in range(width):
            for j in range(height):
                undist = np.matmul(P_inv, np.array([i, j, 1], dtype=np.int32)).T

                x1 = undist[0]
                x2 = undist[1]
                x3 = undist[2]
                
                if x3 != 0:
                    x = x1/x3
                    y = x2/x3

                if x3 != 0 and int(x) < width and int(y) < height and int(x) >= 0 and int(y) >= 0:
                    r, g, b = self.image.getpixel((x, y))

                    new_image.putpixel((i, j), (r, g, b))

        new_image.show()


if __name__ == "__main__":
    Main().main()

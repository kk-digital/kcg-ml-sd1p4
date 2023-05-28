import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageDraw


class ImageMaskCreator:
    def __init__(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()
        self.image = None
        self.mask = None
        self.points = []
        self.current_point = None

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)

        self.save_button = tk.Button(self.root, text="Save Mask", state=tk.DISABLED, command=self.save_mask)
        self.save_button.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("MasterTuto *.jpg")])
        if file_path:
            self.image = Image.open(file_path)
            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
            self.save_button.config(state=tk.NORMAL)

    def start_drawing(self, event):
        self.points.append((event.x, event.y))
        self.current_point = (event.x, event.y)

    def draw(self, event):
        self.canvas.create_line(self.current_point, (event.x, event.y), width=2)
        self.points.append((event.x, event.y))
        self.current_point = (event.x, event.y)

    def end_drawing(self, event):
        self.mask = self.create_mask()

    def create_mask(self):
        mask = np.zeros((self.image.height, self.image.width), dtype=np.uint8)
        if len(self.points) > 2:
            ImageDraw.Draw(Image.fromarray(mask)).polygon(self.points, outline=1, fill=1)
        return mask

    def save_mask(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if file_path:
            Image.fromarray((255 * self.mask).astype(np.uint8)).save(file_path)

    def run(self):
        self.root.title("Image Mask Creator")
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu_bar)
        self.root.mainloop()


if __name__ == "__main__":
    mask_creator = ImageMaskCreator()
    mask_creator.run()

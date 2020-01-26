import os
import csv
import numpy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from config import DATAPATH, KERASPATH, HERE

class Displayer:
    project_dir = DATAPATH

    def __init__(self, bounding_boxes=[], fileNumber=""):
        self.bounding_boxes_list = bounding_boxes
        self.file = fileNumber

    def get_bounding_boxes(self, file):
        with open(f"{self.project_dir}bounding_boxes.csv") as bounding_boxes_csv:
            reader = csv.reader(bounding_boxes_csv, delimiter=',')
            next(bounding_boxes_csv)  # Skip the header
            for row in reader:
                if int(row[0]) == int(file):
                    self.bounding_boxes_list.append(row[1:5])
                if int(row[0]) > int(file):
                    break

    def display_image_and_boxes(self, image, title=""):
        fig, ax = plt.subplots(1)
        ax.axis('off')
        ax.set_title(title)
        ax.imshow(numpy.array(Image.open(image)))
        for bounding_box in self.bounding_boxes_list:
            bounding_box = list(map(float, bounding_box))
            rect = patches.Rectangle((bounding_box[0] - 0.5, bounding_box[1] - 0.5), bounding_box[2], bounding_box[3],
                                     linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def display_image(self, image, title=""):
        fig, ax = plt.subplots(1)
        ax.axis('off')
        ax.set_title(title)
        ax.imshow(image)
        plt.show()

    def clean(self):
        plt.close('all')


    def run_without_image(self):
        self.file = input('Number of the image? -- or EXIT : \n')
        while self.file != "EXIT":
            self.get_bounding_boxes(self.file)
            zeros = '0' * (4 - len(self.file))
            image = f"{self.project_dir}images/ground/ground{zeros}{self.file}.png"
            self.display_image_and_boxes(image, f"image number {self.file}")
            self.file = input('Number of the image? -- or EXIT')






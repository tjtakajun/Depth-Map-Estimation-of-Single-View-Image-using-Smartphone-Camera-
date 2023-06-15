import cv2
import numpy as np

class Median():
    def __init__(self, image):
        self.image = image

    def median(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, ksize=3)
        dst = cv2.cvtColor(median, cv2.COLOR_GRAY2RGB)
        return dst
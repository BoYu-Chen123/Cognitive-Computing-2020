import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

im = Image.open('input.png')
im.show()

input_img = cv2.imread("input.png", cv2.IMREAD_COLOR)
input_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

kernel_smooth = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
kernel_smooth = kernel_smooth / 273
# kernel_sharp = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, -476, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
# kernel_sharp = kernel_sharp / (-256)
kernel_sharp = np.array([[-1,-1,-1,-1,-1], [-1,2,2,2,-1], [-1,2,8,2,-1], [-1,2,2,2,-1], [-1,-1,-1,-1,-1]])
kernel_sharp = kernel_sharp / 8
#########################################
def text(image):
    h = np.shape(image)[0]
    w = np.shape(image)[1]
    temp = Image.fromarray(image)
    draw = ImageDraw.Draw(temp)
    font = ImageFont.truetype("JetBrainsMono-ExtraBold-6.ttf", 12)
    draw.text((50, h-50), "r09922055", font=font, fill=(0, 0, 0))
    return temp

gaussian = cv2.filter2D(input_rgb, -1, kernel_smooth)
laplacian = cv2.filter2D(input_rgb, -1, kernel_sharp)

text_gaussian = text(gaussian)
text_laplacian = text(laplacian)
text_gaussian.save("smoothing.png", "PNG")
text_laplacian.save("sharpening.png", "PNG")
import cv2
import numpy as np


def shapen_Laplacian(in_img):
    I = in_img.copy()
    # kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    # L = cv2.filter2D(I, -1, kernel)
    kernel_2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    L = cv2.filter2D(I, -1, kernel_2)
    a = 0.5
    O = cv2.addWeighted(I, 1, L, a, 0)
    O[O > 255] = 255
    O[O < 0] = 0
    return O


if __name__ == "__main__":
    origin = cv2.imread("./crop_img/demo.jpg")
    shapened = shapen_Laplacian(origin)
    cv2.imshow("name", shapened)
    cv2.waitKey()

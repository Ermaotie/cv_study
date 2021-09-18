import cv2
from util import shapen_Laplacian
from matplotlib import pyplot as plt

path = './crop_img/demo.jpg'


def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.jpg", gray)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    cv2.imwrite("binary.jpg", binary)
    contours, hie = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.putText(
        img,
        "{:.3f}".format(len(contours)),
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        1,
    )
    # cv2.imshow("img", img)
    cv2.imwrite("contours.jpg", img)
    cv2.waitKey(0)


img = cv2.imread(path)
shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
shapen = shapen_Laplacian(shifted)
find_contours(shapen)

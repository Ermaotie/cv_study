# import the necessary packages
import cv2
import os
from matplotlib import pyplot as plt


def saliency_detect(input, out=None):
    # load the input image
    image = cv2.imread(input)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    threshMap = cv2.threshold(saliencyMap, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # subplots
    figure, ax = plt.subplots(3, 1)
    axes = ax.flatten()
    for axis in axes:
        axis.axis('off')
    axes[0].imshow(image[:, :, ::-1])
    axes[0].set_title("Origin")
    axes[1].imshow(threshMap)
    axes[1].set_title("Saliency")
    axes[2].imshow(saliencyMap)
    axes[2].set_title("Thresh")

    if out is None:
        plt.show()
    else:

        plt.savefig(out)


# show the images
def cv_show(name, src):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # 创建窗口
    cv2.resizeWindow(name, 1200, 800)
    cv2.imshow(name, src)


def main():
    for i, j, k in os.walk(path):
        if not os.path.exists(outpath):
            print("路径不存在，正在创建...")
            os.makedirs(outpath)
            print("创建成功！")
        for kk in k:
            out = outpath + sub_name + kk
            saliency_detect(i + kk, out)
            print("第{}个图片创建完成".format(k.index(kk) + 1))


def test():
    saliency_detect('./image/demo.jpg')


if __name__ == '__main__':
    path = './crop_img/'
    outpath = './saliency_detection/'
    sub_name = 'out_'
    main()
    # test()

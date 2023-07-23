import matplotlib.pyplot as plt
import numpy as np
import cv2


# 读一张灰度图
def read_img(path):
    img = cv2.imread(path, 0)
    return img


def max_filter(img, N):
    row, col = img.shape
    img_A = np.zeros_like(img)
    one_H = np.ones([N, N])
    # padding
    l = N // 2
    one_H[1, 1] = 0  # 抠掉中心像素
    img_padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # max-filtered
    for r in range(1, row + 1):
        for c in range(1, col + 1):
            print(img_padded[r - l:r + l + 1, c - l:c + l + 1].shape, one_H.shape)
            neighborhood = img_padded[r - l:r + l + 1, c - l:c + l + 1] * one_H
            # neighborhood = numpy.dot(img_padded[r - l:r + l + 1, c - l:c + l + 1], one_H)
            img_A[r - l, c - l] = np.max(neighborhood)
    return img_A


def min_filter(img, N):
    row, col = img.shape
    img_B = np.zeros_like(img)
    one_H = np.ones([N, N])
    # padding
    l = N // 2
    one_H[1, 1] = 0  # 抠掉中心像素
    img_padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    # min-filtered
    for r in range(1, row + 1):
        for c in range(1, col + 1):
            neighborhood = img_padded[r - l:r + l + 1, c - l:c + l + 1] * one_H
            neighborhood[1, 1] = 255
            img_B[r - l, c - l] = np.min(neighborhood)
    return img_B


def pic_sub(img_1, img_2):
    img_O = np.zeros_like(img_1)
    row, col = img_1.shape
    for r in range(row):
        for c in range(col):
            img_O[r, c] = img_1[r, c].astype("int32") - img_2[r, c].astype("int32")
            if img_O[r, c] <= 0:
                img_O[r, c] += 255
    return img_O


# M = input("Input the value: ")
# M = int(M)
#
# if M == 0:
#     img = read_img("Particles.png")
#     img_A = max_filter(img, N=7)
#     img_B = min_filter(img_A, N=7)
#     img_O = pic_sub(img, img_B)
#     plt.imshow(img_O, 'gray')
#
# if M == 1:
#     img = read_img("Cells.png")
#     img_A = min_filter(img, N=21)
#     img_B = max_filter(img_A, N=21)
#     img_O = cv2.subtract(img, img_B)
#     plt.imshow(img_O, 'gray')

def task1():
    img = read_img("Particles.png")
    plt.imshow(img, 'gray')
    img_A = max_filter(img, N=7)
    img_B = min_filter(img_A, N=3)
    plt.imshow(img_B, 'gray')
    plt.show()


if __name__ == "main":
    img = cv2.imread("Particles.png", cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, 'gray')
    # img_A = max_filter(img, N=7)
    # plt.imshow(img_A, 'gray')
    plt.show()

import numpy as np
import math
import copy
import cv2


def spilt(a):
    if a % 2 == 0:
        x1 = x2 = a / 2
    else:
        x1 = math.floor(a / 2)
        x2 = a - x1
    return -x1, x2


def original(i, j, k, a, b, img):
    x1, x2 = spilt(a)
    y1, y2 = spilt(b)
    temp = np.zeros(a * b)
    count = 0
    for m in range(x1, x2):
        for n in range(y1, y2):
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
                temp[count] = img[i, j, k]
            else:
                temp[count] = img[i + m, j + n, k]
            count += 1
    return temp


# 最大值滤波
def max_functin(n, img):
    img0 = copy.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, n, n, img0)
                img[i, j, k] = np.max(temp)
    return img


# 最小值滤波
def min_functin(n, img):
    img0 = copy.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, n, n, img0)
                img[i, j, k] = np.min(temp)
    return img


def background_subtraction(image_I, image_B):
    img_O = np.zeros_like(image_I)
    row = image_I.shape[0]
    col = image_I.shape[1]
    for r in range(row):
        for c in range(col):
            img_O[r, c] = image_I[r, c].astype("int32") - image_B[r, c].astype("int32")
            # if img_O[r, c] <= 0:
            #     img_O[r, c] += 255
    return img_O


def task1():
    img = cv2.imread("Particles.png")
    max_N = 15
    min_N = 11
    img_A = max_functin(max_N, copy.copy(img))
    # img_B = min_functin(min_N, copy.copy(img_A))

    # cv2.imshow("img_I", img)
    cv2.imshow("img_A", img_A)
    # cv2.imshow("img_B", img_B)
    cv2.imwrite("A_15.png", img_A)
    # cv2.imshow("min_img", min_img)

    cv2.waitKey(0)


def task2():
    img = cv2.imread("Particles.png", 0)
    img_B = cv2.imread("B.png", 0)
    img_O = background_subtraction(img, img_B)
    cv2.imshow("img_O", img_O)
    cv2.imwrite("img_O.png", img_O)
    cv2.waitKey(0)


def task3():
    while (True):
        M = input("Input the value(0 or 1): ")
        M = int(M)
        if M == 0 or M == 1:
            break
    if M == 0:
        img = cv2.imread("Cells.png")
        max_N = 11
        min_N = 11
        img_A = max_functin(max_N, copy.copy(img))
        img_B = min_functin(min_N, copy.copy(img_A))
        cv2.imshow("img_B", img_B)
        cv2.imwrite("task_3_img_B_M0.png", img_B)

        img_I = cv2.imread("Particles.png", 0)
        img_B = cv2.imread("task_3_img_B_M1.png.png", 0)
        img_O = background_subtraction(img_I, img_B)
        cv2.imshow("img_O", img_O)
        cv2.imwrite("task_3_img_O_M0", img_O)

    if M == 1:
        # img = cv2.imread("Cells.png")
        # max_N = 21
        # min_N = 21
        # img_A = min_functin(max_N, copy.copy(img))
        # img_A = cv2.imread("task_3_A_M1_21.png")
        # img_B = max_functin(min_N, copy.copy(img_A))
        # cv2.imshow("img_A_21", img_A)
        # cv2.imwrite("A_M1_21.png", img_A)
        # cv2.imshow("img_B_21", img_B)
        # cv2.imwrite("B_M1_21.png", img_B)

        # img_I = cv2.imread("Particles.png", 0)
        # img_B = cv2.imread("task_3_img_B_M1.png.png", 0)
        # img_O = background_subtraction(img_I, img_B)
        # cv2.imshow("img_O", img_O)
        # cv2.imwrite("task_3_img_O_M1", img_O)

        img = cv2.imread("Cells.png", 0)
        img_B = cv2.imread("B_M1_21.png", 0)
        img_O = background_subtraction(img, img_B)
        cv2.imshow("img_O", img_O)
        cv2.imwrite("Task3_img_O.png", img_O)
        cv2.waitKey(0)
    cv2.waitKey(0)


if __name__ == "__main__":
    task3()

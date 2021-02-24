# canny 이외에 여러가지 사용해서 이미지 깔끔하게 불러오기

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../dacon12/data/train/00000.png", cv2.IMREAD_GRAYSCALE)

canny = cv2.Canny(img, 30, 70)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
laplacian = cv2.Laplacian(img, cv2.CV_8U)

images = [canny, sobelx, sobely, laplacian]
titles = ['canny', 'sobelx', 'sobely', 'laplacian']

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i]), plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

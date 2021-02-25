# 캐리로 저장한 이미지 OCR로 찾기

import cv2
import numpy as np

large = cv2.imread('../dacon12/data/newtrain/00007.png')

small = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
mask = np.zeros(bw.shape, dtype=np.uint8)
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    if r > 0.45 and w > 8 and h > 8:
        cv2.rectangle(large, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

    # show image with contours rect
cv2.imshow('rects', large)
cv2.waitKey()


    # 검출한 것만 쓰기
    # img_trim = large[y:y+h, x:x+w]

    # cv2.imshow('rects', img_trim)
    # cv2.waitKey(0)


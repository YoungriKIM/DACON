# 테스트 이미지를 위한 파일

import cv2
import numpy as np

large = cv2.imread('../dacon12/data/test/50002.png')


# 이미지 전처리 -------------------------------------
#254보다 작고 0이아니면 0으로 만들어주기
x_df2 = np.where((large <= 254) & (large != 0), 0, large)

# 이미지 팽창
x_df3 = cv2.dilate(x_df2, kernel=np.ones((2, 2), np.uint8), iterations=1)

# 블러 적용, 노이즈 제거
x_df4 = cv2.medianBlur(src=x_df3, ksize= 5)
# --------------------------------------------------


small = cv2.cvtColor(x_df4, cv2.COLOR_BGR2GRAY)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
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


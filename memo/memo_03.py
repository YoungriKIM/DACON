import cv2


tmp_x = []

for a in [0,1,10,11,14,15,16,17]:
    pred_x = cv2.imread('../dacon12/data/newtrain/' + str(a).zfill(5) + '.png', cv2.IMREAD_GRAYSCALE)
    print

print(tmp_x.shape)


img_trim = image_data[y:y+h, x:x+w]
cv2.imwrite(f'../dacon12/data/splittrain/org_trim{i}.png', img_trim)
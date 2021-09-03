import cv2 as cv

img = cv.imread("images/1.png")
out_img = img.copy()
div = 128
out_img = img // div * div + div // 2
cv.imshow("out",out_img)
cv.waitKey(0)
cv.imwrite("images/out-1.png",out_img)
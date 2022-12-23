import cv2 as cv

img = cv.imread("./faces_4/an2i/an2i_left_angry_open_4.pgm")
cv.imshow("input image", img)
cv.waitKey()
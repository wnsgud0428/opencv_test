import cv2

image = cv2.imread("media/good.png_removebg.png")

# contour 그리기
image1_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
contours, hierarcy = cv2.findContours(
    image1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
output_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 사진 자르기 (450x600)
shoulder_x = 150
shoulder_y = 250
hip_x = 250
hip_y = 400
cut = image[shoulder_y:hip_y, shoulder_x:hip_x]
cv2.imshow("cut", cut)

# 보여주는 부분
cv2.imshow("output_image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

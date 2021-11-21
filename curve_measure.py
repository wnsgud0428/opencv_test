import cv2
import numpy
from scipy.interpolate import splprep, splev

image = cv2.imread("images/resize_wrong2.jpg_removebg.png")
# good-crop.jpg_removebg.png
# wrong-crop.jpg_removebg.png
# contour 그리기
image1_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
contours, hierarcy = cv2.findContours(
    image1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
output_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow("output_image", output_image)
cv2.waitKey(0)

left_shoulder_x = 347
left_shoulder_y = 100
left_hip_x = 512
left_hip_y = 263
left_shoulder = (left_shoulder_x, left_shoulder_y)
left_hip = (left_hip_x, left_hip_y)


smoothened = []
for contour in contours:
    x, y = contour.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    tck, u = splprep([x, y], u=None, s=1.0, per=1)
    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
    u_new = numpy.linspace(u.min(), u.max(), 25)
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    x_new, y_new = splev(u_new, tck, der=0)
    # Convert it back to numpy format for opencv to be able to display it
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
    smoothened.append(numpy.asarray(res_array, dtype=numpy.int32))
cv2.drawContours(image, smoothened, -1, (255, 255, 255), 2)
cv2.line(
    image,
    left_shoulder,
    left_hip,
    (255, 255, 255),
    thickness=None,
    lineType=None,
    shift=None,
)

# print(smoothened)
print(smoothened)
c1 = max(smoothened, key=cv2.contourArea)
print(c1)

# 보여주는 부분
cv2.imshow("output_image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

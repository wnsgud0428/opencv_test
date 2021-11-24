import cv2
import numpy
from scipy.interpolate import splprep, splev
import matplotlib.pylab as pl  # 지금은 안씀

"""
<to do>
지금 문제점: 등 위의 점 vs 어깨-골반 직선의 기울기 비교는 별로 좋지 않은듯. 
해결 방법: 자체적인 등 위의 점들의 기울기 변화량을 가지고 측정해야 될듯.

"""


def returnLineEquCoef(p1, p2):
    """[기울기m, y절편] 리턴"""
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)  # 기울기 m 계산(a값)
        n = y1 - (m * x1)  # y 절편 계산(b값)
    return [m, n]


def isPointUnderTheLine(line_equ_coef, point):
    """점이 직선의 밑에있는지 따져 T/F 리턴"""
    m = line_equ_coef[0]
    n = line_equ_coef[1]
    x1 = point[0]
    y1 = point[1]
    result = m * x1 + n - y1
    if result > 0:
        return True
    else:
        return False


image_input_type = "good"
image = cv2.imread("images/resize_" + image_input_type + "2.jpg_removebg.png")
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

origin_left_hip = []
origin_left_shoulder = []
if image_input_type == "wrong":
    origin_left_shoulder = [347, 100]
    origin_left_hip = [512, 263]
elif image_input_type == "good":
    origin_left_shoulder = [362, 106]
    origin_left_hip = [464, 280]

# ROI 설정하여 이미지 slice
roi = {
    "x_begin": origin_left_shoulder[0],
    "x_end": 640,
    "y_begin": 0,
    "y_end": origin_left_hip[1],
}
image = image[roi["y_begin"] : roi["y_end"], roi["x_begin"] : roi["x_end"]]
# slice한 이미지의 관절 포인트 값 수정
left_shoulder = [origin_left_shoulder[0], origin_left_shoulder[1]]
left_hip = [origin_left_hip[0], origin_left_hip[1]]

left_shoulder[0] -= origin_left_shoulder[0]
left_shoulder[1] -= 0
left_hip[0] -= origin_left_shoulder[0]
left_hip[1] -= 0

# contour 그리기
image1_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
contours, hierarcy = cv2.findContours(
    image1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
output_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# cv2.imshow("output_image", output_image)
# cv2.waitKey(0)

smoothened = []
for contour in contours:
    x, y = contour.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    tck, u = splprep([x, y], u=None, s=1.0, per=1)
    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
    u_new = numpy.linspace(u.min(), u.max(), 20)
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    x_new, y_new = splev(u_new, tck, der=0)
    # Convert it back to numpy format for opencv to be able to display it
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
    smoothened.append(numpy.asarray(res_array, dtype=numpy.int32))
cv2.drawContours(image, smoothened, -1, (255, 255, 255), 2)
print(smoothened)
print("skeleton:")
print(left_shoulder)
print(left_hip)
cv2.line(
    image,
    left_shoulder,
    left_hip,
    (255, 255, 255),
    thickness=None,
    lineType=None,
    shift=None,
)


line_equ_coef = returnLineEquCoef(left_shoulder, left_hip)
print(line_equ_coef)


want_point_list = []
for i in range(len(smoothened)):
    for j in range(len(smoothened[i])):
        point = [smoothened[i][j][0][0], smoothened[i][j][0][1]]

        if point[0] > left_shoulder[0] and point[1] < left_hip[1]:
            if isPointUnderTheLine(line_equ_coef, point):
                want_point_list.append(point)
print("want_point_list:")
print(want_point_list)
# print((smoothened[0][0][0]))
# print(smoothened[1])

### 우리가 원하던 부분의 좌표값들 얻어내기
# 선으로 그리기
# for i in range(len(want_point_list) - 1):
#     cv2.line(
#         image,
#         want_point_list[i],
#         want_point_list[i + 1],
#         (255, 0, 0),
#         thickness=3,
#         lineType=None,
#         shift=None,
#     )

slope_diff_sum = 0
shoulder_to_hip_slope = returnLineEquCoef(left_shoulder, left_hip)[0]
print("shoulder_to_hip_slope:", shoulder_to_hip_slope)
want_point_list_len = len(want_point_list)
want_point_list.reverse()  # 어깨에 있는 점을 먼저 list에  저장하기 위함

# for i in range(want_point_list_len):
#     # 파란색으로 등에있는 점 찍기
#     cv2.line(
#         image,
#         want_point_list[i],
#         want_point_list[i],
#         (255, 0, 0),
#         thickness=3,
#         lineType=None,
#         shift=None,
#     )
# print(want_point_list_len)

for i in range(5):
    # 파란색으로 등에있는 점 찍기
    cv2.line(
        image,
        want_point_list[i],
        want_point_list[i],
        (255, 0, 0),
        thickness=3,
        lineType=None,
        shift=None,
    )

    if i < 4:
        coef = returnLineEquCoef(want_point_list[i], want_point_list[i + 1])
        back_point_slope = coef[0]
        slope_diff = abs(shoulder_to_hip_slope - back_point_slope)
        print(slope_diff)
        slope_diff_sum += slope_diff
print("slope_diff_sum:", slope_diff_sum)

# 보여주는 부분
cv2.imshow("output_image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

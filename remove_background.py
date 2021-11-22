import cv2
from matplotlib import pyplot as plt
import numpy as np
import requests
from removebg import RemoveBg
import logging

# opencv 처리
"""
image = cv2.imread("input/wrong.png")
image_gray = cv2.imread("input/wrong.png", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

edged = cv2.Canny(blur, 10, 250)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(
    closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
total = 0


contours_image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# 보여주는 부분
cv2.imshow("contours_image", contours_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

### 한달에 50번 까지만 무료임
# 기본제공하는 api사용
""" 
response = requests.post(
    "https://api.remove.bg/v1.0/removebg",
    files={"image_file": open("input/resize1.jpg", "rb")},
    data={"size": "auto"},
    headers={"X-Api-Key": "eihrhcx6ZmqHYx69CZna8mHa"},
)
if response.status_code == requests.codes.ok:
    with open("output/no-bg.png", "wb") as out:
        out.write(response.content)

        cv2.imshow("contours_image", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Error:", response.status_code, response.text)
"""

# 라이브러리 설치해서 사용 -> 속도향상
API_ENDPOINT = "https://api.remove.bg/v1.0/removebg"


class NewRemoveBg(RemoveBg):  # 메소드 오버라이드
    def __init__(self, api_key, error_log_file):
        self.__api_key = api_key
        logging.basicConfig(filename=error_log_file)

    def remove_background_from_img_file(
        self, img_file_path, size="regular", bg_color=None
    ):
        # Open image file to send information post request and send the post request
        img_file = open(img_file_path, "rb")
        response = requests.post(
            API_ENDPOINT,
            files={"image_file": img_file},
            data={"size": size, "bg_color": bg_color},
            headers={"X-Api-Key": self.__api_key},
        )
        response.raise_for_status()
        self.__output_file__(response, img_file.name + "_removebg.png")  # 출력 파일 이름 변경

        img_file.close()


rmbg = NewRemoveBg("iH4AoLSs2an9EiS8LPSiqAqp", "error.log")
rmbg.remove_background_from_img_file("images/resize_good2.jpg")
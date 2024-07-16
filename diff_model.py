import os
import cv2
import easyocr
import numpy as np
import re


def crop_background(image, bboxes):
    """Blackens all pixels in a image except the bounding boxes.
        Args:
            image: The image to be processed
            bboxes ([cord1, cord2, cord3, cord4]): The areas to be left unmasked.
        Returns:
            (2x2array, float): The result image and the remaining bright area.
    """
    # create full black image
    img = np.zeros(image.shape)

    # keep track of bright area
    bright_area = 0
    for bbox in bboxes:
        # detect box area
        left = bbox[0][0]
        top = bbox[0][1]
        right = bbox[2][0]
        bottom = bbox[2][1]
        bright_area += (right - left) * (bottom - top)
        # substitute area with original image
        for i in range(top, bottom + 1):
            for j in range(left, right + 1):
                img[i][j] = image[i][j]
    return img, bright_area


def image_diff(img1, img2, bboxes):
    """Calculate the difference ratio in the bounding boxes of two images.
        Args:
            img1, img2: Two images with the same size
            bboxes ([cord1, cord2, cord3, cord4]): The focused(compared) part of the images.

        Returns:
            float: The difference ratio.
    """
    clean_img1, area1 = crop_background(img1, bboxes)
    clean_img2, area2 = crop_background(img2, bboxes)

    result_img = cv2.subtract(clean_img1, clean_img2)
    h, w, _ = img1.shape
    diff_count = 0
    for i in range(h):
        for j in range(w):
            if not np.all(result_img[i][j] == 0):
                diff_count += 1
    return diff_count * 2 / (area1 + area2)


img = cv2.imread("temp/test.png")
reader = easyocr.Reader(['ch_tra'])

vid = cv2.VideoCapture("temp/13b.mp4")
if not vid.isOpened():
    raise FileNotFoundError("Please check your input video path.")

frame_cnt = 0
last_frame = None
bboxes = []
curr_text = ""
while True:
    print(f"current frame: {frame_cnt}")
    ret, frame = vid.read()
    if not ret:
        break

    h, w, _ = frame.shape
    crop_frame = frame[int(0.75 * h):h, :]

    # check the current text, if two images similar, simply assign the text to it.
    if len(curr_text) > 0 and image_diff(last_frame, crop_frame, bboxes) < 0.8:
        pass
    # not same as last frame, do OCR.
    else:
        bboxes = []
        curr_text = ""
        result = reader.readtext(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))

        if len(result) != 0:
            max_conf = 0
            # record the bounding boxes
            for (bbox, text, conf) in result:
                if conf > 0.25:
                    max_conf = max(max_conf, conf)
                    curr_text += text
                    bboxes.append(bbox)
            curr_text = re.sub(r'[<>:"/\\|?*.]', '', curr_text)
            print(f"new subtitle detected: {curr_text}, confidence: {max_conf}")
            os.makedirs(f"result/{curr_text}", exist_ok=True)

    if len(curr_text) != 0:
        cv2.imencode('.png', frame)[1].tofile(f"result/{curr_text}/{frame_cnt}.png")
    last_frame = crop_frame
    frame_cnt += 1

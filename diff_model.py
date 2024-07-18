import os
import cv2
import easyocr
import re
from frame import frame_diff


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
    if len(curr_text) > 0 and frame_diff(last_frame, crop_frame, bboxes) < 0.8:
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

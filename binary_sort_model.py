import cv2
import os
from lev_dist import str_similarity
from frameOps import frame_diff, binary_search
import sys
import easyocr
import re

# path variables set up
vid_path = r"temp/lost2.mp4"
output_folder = r"result1"
if not os.path.exists(vid_path):
    print("請更改vid_path為你影片的儲存位置")
    sys.exit(0)
os.makedirs(output_folder, exist_ok=True)
vid = cv2.VideoCapture(vid_path)

# store and display video metadata
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = vid.get(cv2.CAP_PROP_FPS)
print(f"size: {height}x{width}, fps: {int(fps)}, total frames: {int(frame_count)}")
check = input("Continue Process? (Y/n)").strip(" ")
if check == 'Y' or check == 'y' or check == 'yes':
    reader = easyocr.Reader(['ch_tra'])
else:
    sys.exit(0)

# setup buffer
prev_crop_frame = None
prev_text = ""
prev_bboxes = []
prev_conf = 0
history_buffer = []
process_buffer = []
prev_bboxes.clear()
history_buffer.clear()
process_buffer.clear()
# setup string edit costs
insert_cost = {" ": 0.1, ".": 0.2, "-": 0.2, "0": 0.2}
delete_cost = {" ": 0.1, ".": 0.2, "-": 0.2, "0": 0.2}
replace_cost = {("-", "一"): 0.2}
# loop through the entire video
for frame_index in range(frame_count):
    # read the video frame by frame and use each 3 frame
    ret, frame = vid.read()
    if not ret:
        break
    # step size of 5
    if frame_index % 12 != 0:
        history_buffer.append(frame)
        continue
    print(f"current frame: {frame_index}")

    # setup frame for comparison
    crop_frame = frame[int(0.75 * height):height, :]
    # try diff it to find if prev and current have same text
    if len(prev_text) != 0:
        difference = frame_diff(prev_crop_frame, crop_frame, prev_bboxes)
        print(f"difference: {difference}")
        if difference < 0.4:
            print(f"same sub by diff")
            # add gap frames to process buffer
            process_buffer += history_buffer
            process_buffer.append(frame)
            history_buffer.clear()
            print(f"processor buffer size: {len(process_buffer)}")
            continue
    # diff doesn't work out, do OCR.
    results = reader.readtext(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
    curr_text = ""
    curr_bboxes = []
    curr_conf = 1
    for (bbox, text, conf) in results:
        # check if box aligned in middle
        left = bbox[0][0]
        right = bbox[2][0]
        mid = (left+right)//2
        if mid > 0.7*width or mid < 0.3*width:
            continue
        curr_text += text
        curr_bboxes.append(bbox)
        curr_conf = min(curr_conf, conf)
    curr_text = re.sub(r'[<>:"/\\|?*.]', '', curr_text)
    print(f"OCR outcome: {curr_text}, {curr_conf}")
    # if frame has subtitles:
    if len(curr_text) != 0 and curr_conf > 0.3:
        # frame has text but previous doesn't, search for starting
        if len(prev_text) == 0:
            print(f"Got new subtitle: {curr_text}, confidence {curr_conf}")
            print("searching for starting")
            process_buffer += binary_search(history_buffer, crop_frame, curr_bboxes, "start")
            process_buffer.append(frame)
            history_buffer.clear()
            prev_conf = curr_conf
            prev_text = curr_text
            prev_bboxes = curr_bboxes
            prev_crop_frame = crop_frame
        else:
            sim = str_similarity(prev_text, curr_text, insert_costs=insert_cost, delete_costs=delete_cost,
                                 replace_costs=replace_cost)
            print(f"similarity: {prev_text}/{curr_text}: {sim}")
            # frame has same text with previous text
            if sim > 0.9:
                print("Same text")
                # add gap frames to process buffer
                process_buffer += history_buffer
                process_buffer.append(frame)
                if curr_conf > prev_conf:
                    prev_conf = curr_conf
                    prev_text = curr_text
                    prev_bboxes = curr_bboxes
                    prev_crop_frame = crop_frame
                history_buffer.clear()
            # frame has text not equal to previous text
            else:
                print(f"Got new subtitle: {curr_text}, confidence {curr_conf}")
                # add end to process buffer
                process_buffer += binary_search(history_buffer, prev_crop_frame, prev_bboxes, "end")
                # process the process buffer
                os.makedirs(f"{output_folder}/{prev_text}", exist_ok=True)
                index = 0
                for image in process_buffer:
                    cv2.imencode('.png', image)[1].tofile(f"{output_folder}/{prev_text}/{index}.png")
                    index += 1
                process_buffer.clear()
                # add start to process buffer
                process_buffer += binary_search(history_buffer, crop_frame, curr_bboxes, "start")
                process_buffer.append(frame)
                # set current buffer metadata
                prev_conf = curr_conf
                prev_text = curr_text
                prev_bboxes = curr_bboxes
                prev_crop_frame = crop_frame
            history_buffer.clear()
    # text detected, but unclear
    elif len(curr_text) != 0:
        print("text unclear")
        history_buffer.append(frame)
    # if frame doesn't contain text && last frame contains text, search the end of subtitle
    elif len(prev_text) != 0:
        print("finding end of subtitle")
        # add end to process buffer
        process_buffer += binary_search(history_buffer, prev_crop_frame, prev_bboxes, "end")
        # process the process buffer
        os.makedirs(f"{output_folder}/{prev_text}", exist_ok=True)
        index = 0
        for image in process_buffer:
            cv2.imencode('.png', image)[1].tofile(f"{output_folder}/{prev_text}/{index}.png")
            index += 1
        process_buffer.clear()
        history_buffer.clear()
        prev_text = ""
    print(f"processor buffer size: {len(process_buffer)}")

vid.release()

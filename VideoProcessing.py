import cv2
import os
import re
import sys
import lev_dist

# path variables set up
vid_path = r"temp/13b.mp4"
output_folder = r"result"
if not os.path.exists(vid_path):
    print("請更改vid_path為你影片的儲存位置")
    sys.exit(0)
os.makedirs(output_folder, exist_ok=True)
vid = cv2.VideoCapture(vid_path)

# store and display video metadata
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = vid.get(cv2.CAP_PROP_FPS)
print(f"size: {height}x{width}, fps: {int(fps)}, total frames: {int(frame_count)}")
check = input("Continue Process? (Y/n)").strip(" ")
if check == 'Y' or check == 'y' or check == 'yes':
    import ezocr
else:
    sys.exit(0)

# setup buffer
recent_text = ""
buffer_frame_index = 0
frame_index = 0
recent_conf = 0
buffer = []
in_between_buffer = []
buffer.clear()
in_between_buffer.clear()
for frame in range(frame_count):
    # read the video frame by frame and use each 3 frame
    ret, img = vid.read()
    if not ret:
        break
    if frame % 3 != 0:
        continue

    # crop the image where the subtitles might be
    img_height, img_width, _ = img.shape
    corp_img = img[int(img_height * 0.75):img_height, int(0.2 * img_width):int(0.8*img_width)]
    # convert image to RGB for easyocr to read
    ocr_img = cv2.cvtColor(corp_img, cv2.COLOR_BGR2RGB)
    text, conf = ezocr.image2text(ocr_img)  # OCR
    text = re.sub(r'[<>:"/\\|?*.]', '', text)  # clear bad filename characters
    print(f"current frame: {frame}")
    # check if got something in OCR
    if text != "" and conf > 0.5:
        # calculate string similarity
        insert_cost = {" ": 0.1, ".": 0.2, "-": 0.2, "0": 0.2}
        delete_cost = {" ": 0.1, ".": 0.2, "-": 0.2, "0": 0.2}
        replace_cost = {("-", "一"): 0.2}
        sim = lev_dist.str_similarity(recent_text, text, insert_costs=insert_cost, delete_costs=delete_cost,
                                      replace_costs=replace_cost)

        # if small similarity, new string found. process the buffer and update the buffer.
        if sim < 0.8:
            # store images in buffer
            if recent_text != "":
                os.makedirs(f"{output_folder}/{recent_text}", exist_ok=True)
                for image in buffer:
                    cv2.imencode('.png', image)[1].tofile(f"{output_folder}/{recent_text}/{buffer_frame_index}.png")
                    buffer_frame_index += 1
            # reset buffer and put current image in buffer
            buffer.clear()
            in_between_buffer.clear()
            recent_text = text
            buffer_frame_index = frame_index
            recent_conf = conf
            print("Got one: " + text + " confidence: " + str(conf))
        # if similar, add the in between buffer to buffer
        # then, update text and conf if needed.
        else:
            for img in in_between_buffer:
                buffer.append(img)
            in_between_buffer.clear()
            if conf > recent_conf:
                print("Got one: " + text + " confidence: " + str(conf))
                recent_text = text
                recent_conf = conf
        buffer.append(img)
    # no text detected, adds to in between buffer.
    else:
        in_between_buffer.append(img)
    frame_index += 1

# store images in buffer
if recent_text != "":
    os.makedirs(f"{output_folder}/{recent_text}", exist_ok=True)
    for image in buffer:
        cv2.imencode('.png', image)[1].tofile(f"{output_folder}/{recent_text}/{buffer_frame_index}.png")
        buffer_frame_index += 1

vid.release()

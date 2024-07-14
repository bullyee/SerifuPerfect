import cv2
import os
import re
import sys

vid_path = r"temp/13b.mp4"
output_folder = r"result"
if not os.path.exists(vid_path):
    print("請更改vid_path為你影片的儲存位置")
    sys.exit(0)
os.makedirs(output_folder, exist_ok=True)
vid = cv2.VideoCapture(vid_path)

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

recent_text = ""
recent_count = 0
for frame in range(frame_count):
    ret, img = vid.read()
    if not ret:
        break
    if frame % 3 != 0:
        continue

    img_height, img_width, _ = img.shape
    corp_img = img[int(img_height * 0.75):img_height, 0:img_width]
    ocr_img = cv2.cvtColor(corp_img, cv2.COLOR_BGR2RGB)
    text, conf = ezocr.image2text(cv2.cvtColor(ocr_img, cv2.COLOR_BGR2RGB))
    text = re.sub(r'[<>:"/\\|?*.]', '', text)
    print(f"current frame: {frame}")
    if text != "" and conf > 0.5:
        if recent_text != text:
            recent_text = text
            recent_count = 0
            print("Got one: " + text)
            os.makedirs(f"{output_folder}/{recent_text}", exist_ok=True)
        cv2.imencode('.png', img)[1].tofile(f"{output_folder}/{recent_text}/{text}{recent_count}.png")
        recent_count += 1
    else:
        recent_text = ""
        recent_count = 0
    # if text != "":
    #     print(text + str(conf))


vid.release()

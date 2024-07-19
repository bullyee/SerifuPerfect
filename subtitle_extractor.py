import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import easyocr
from frame import *
from lev_dist import *


class SubtitleExtractor:
    # video attributes
    video = None
    video_metadata = None
    # OCR attributes
    reader = None
    # parameters
    # I/O
    video_path = "temp/13b.mp4"
    output_folder_path = "default"
    # processing threshold
    ocr_accept_threshold = 0.3
    str_diff_threshold = 0.95
    frame_diff_threshold = 0.95
    step_size = 8
    mid_attention = 0.2
    # levenshtein distance parameters
    insert_cost = {" ": 0.2, ".": 0.5, "-": 0.5, "0": 0.5, ":": 0.8, "!": 0.8, "%": 0.5, ")": 0.8, "(": 0.8, "=": 0.5,
                   ";": 0.8, "_": 2}
    delete_cost = {" ": 0.2, ".": 0.5, "-": 0.5, "0": 0.5, ":": 0.8, "!": 0.8, "%": 0.5, ")": 0.8, "(": 0.8, "=": 0.5,
                   ";": 0.8, "_": 2}
    replace_cost = {("-", "一"): 0.5, ("哦", "我"): 0.5, ("]", "因"): 0.5, ("哦", "我"): 0.5}

    # video Processing attributes
    def __init__(self, video_path, output_path):
        self.video = cv2.VideoCapture(video_path)
        self.output_folder_path = output_path
        if not self.video.isOpened():
            raise ValueError("Video path invalid")
        self.video_metadata = {
            "frame_height": int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_width": int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "fps": int(self.video.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        self.reader = easyocr.Reader(['ch_tra'])

    def ocr_config(self, reader_langs):
        self.reader = easyocr.Reader(reader_langs)

    def run(self):
        history_buffer = []
        buffer_frame = None
        process_buffer = []
        for frame_index in range(self.video_metadata["frame_count"]):
            ret, curr_frame_img = self.video.read()
            if not ret:
                return

            if frame_index % self.step_size != 0:
                history_buffer.append(Frame(curr_frame_img, frame_index))
                continue

            curr_frame = Frame(curr_frame_img, frame_index)
            self.__frame_ocr(curr_frame)

            # handle first frame
            if buffer_frame is None:
                buffer_frame = curr_frame
                continue
            print(f"{curr_frame.index}: {curr_frame.text}, {curr_frame.confidence}, buffer text: {buffer_frame.text}")

            # No text detected
            if curr_frame.text is None:
                if buffer_frame.text is not None:
                    end_index = self.__binary_search(process_buffer[-1], history_buffer, "end")
                    process_buffer += history_buffer[:end_index + 1]
                    # store images
                    self.__save_images(buffer_frame, process_buffer)
                    process_buffer.clear()

                buffer_frame = curr_frame
                history_buffer.clear()
            # Text detected but not confident
            elif curr_frame.text == "":
                history_buffer.append(curr_frame)
            # text detected
            else:
                # buffer doesn't contain text
                if buffer_frame.text is None:
                    # search for the staring frame of this subtitle
                    start_index = self.__binary_search(curr_frame, history_buffer, "start")
                    process_buffer += history_buffer[start_index:]
                    process_buffer.append(curr_frame)
                    # clear history buffer and set buffer representation frame
                    history_buffer.clear()
                    buffer_frame = curr_frame
                print(
                    f"{curr_frame.text}, {buffer_frame.text} similarity: {str_similarity(buffer_frame.text, curr_frame.text, insert_costs=self.insert_cost, delete_costs=self.delete_cost, replace_costs=self.replace_cost)}")
                # buffer has text != current text
                if str_similarity(buffer_frame.text, curr_frame.text, insert_costs=self.insert_cost,
                                  delete_costs=self.delete_cost,
                                  replace_costs=self.replace_cost) < self.str_diff_threshold:
                    # search for end of previous subtitle end frame
                    end_index = self.__binary_search(process_buffer[-1], history_buffer, "end")
                    process_buffer += history_buffer[:end_index + 1]
                    # store images
                    print("got different text")
                    self.__save_images(buffer_frame, process_buffer)
                    process_buffer.clear()
                    # search for current subtitle start frame
                    history_buffer = history_buffer[end_index + 1:]
                    start_index = self.__binary_search(buffer_frame, history_buffer, "start")
                    process_buffer += history_buffer[start_index:]
                    process_buffer.append(curr_frame)
                    buffer_frame = curr_frame
                    history_buffer.clear()
                # buffer has text = current text
                else:
                    process_buffer += history_buffer
                    process_buffer.append(curr_frame)
                    history_buffer.clear()
                    if curr_frame.confidence > buffer_frame.confidence:
                        buffer_frame = curr_frame

    def __save_images(self, example_frame, frames):
        root_dir = example_frame.text
        os.makedirs(f"{self.output_folder_path}/{root_dir}", exist_ok=True)

        def save_image(path, image):
            cv2.imencode('.png', image)[1].tofile(f"{path}.png")

        with ThreadPoolExecutor(max_workers=2) as pool_executor:
            futures = []
            for frame in frames:
                frame_path = f"{self.output_folder_path}/{root_dir}/{frame.index}"
                future = pool_executor.submit(save_image, frame_path, frame.image)
                futures.append(future)
            for future in futures:
                future.result()

    def __frame_ocr(self, frame):
        outcome_text = ""
        bboxes = []
        min_confidence = 1
        for (bbox, text, conf) in self.reader.readtext(frame.ocr_image()):
            if self.mid_attention != 1:
                # check if box aligned in middle
                left = bbox[0][0]
                right = bbox[2][0]
                mid = (left + right) // 2
                if mid < (self.mid_attention / 2 - 0.5) * frame.width or mid > (
                        self.mid_attention / 2 + 0.5) * frame.width:
                    continue
            outcome_text = outcome_text + text + "_"
            bboxes.append(bbox)
            min_confidence = min(min_confidence, conf)
        outcome_text = outcome_text.rstrip("_")
        # No text detected, frame.text = None
        if outcome_text == "":
            pass
        # Text detected but not confident, frame text = ""
        elif min_confidence < self.ocr_accept_threshold:
            frame.set_ocr(bounding_boxes=bboxes, text="", confidence=min_confidence)
        # Has text and confident
        else:
            frame.set_ocr(bounding_boxes=bboxes, text=outcome_text, confidence=min_confidence)

    def __binary_search(self, pivot_frame, frames, search_for):
        if search_for == "end":
            return self.__end_bs(pivot_frame, frames, 0, len(frames) - 1)
        elif search_for == "start":
            return self.__start_bs(pivot_frame, frames, 0, len(frames) - 1)
        else:
            raise ValueError("search for muse be 'end' or 'start'")

    def __end_bs(self, pivot_frame, frames, start, end):
        if start >= end:
            return end

        if (end + start) % 2 != 0:
            mid1, mid2 = (end + start) // 2, (end + start) // 2 + 1
            if frame_diff(pivot_frame, frames[mid2], pivot_frame.bboxes, "ssim") > self.frame_diff_threshold:
                return self.__end_bs(pivot_frame, frames, mid2 + 1, end)
            elif frame_diff(pivot_frame, frames[mid1], pivot_frame.bboxes, "ssim") > self.frame_diff_threshold:
                return mid1
            else:
                return self.__end_bs(pivot_frame, frames, start, mid1 - 1)
        else:
            mid = (end + start) // 2
            if frame_diff(pivot_frame, frames[mid], pivot_frame.bboxes, "ssim") > self.frame_diff_threshold:
                return self.__end_bs(pivot_frame, frames, mid + 1, end)
            else:
                return self.__end_bs(pivot_frame, frames, start, mid - 1)

    def __start_bs(self, pivot_frame, frames, start, end):
        if start >= end:
            return end

        if (end + start) % 2 != 0:
            mid1, mid2 = (end + start) // 2, (end + start) // 2 + 1
            if frame_diff(pivot_frame, frames[mid1], pivot_frame.bboxes, "ssim") > self.frame_diff_threshold:
                return self.__start_bs(pivot_frame, frames, start, mid1 - 1)
            elif frame_diff(pivot_frame, frames[mid2], pivot_frame.bboxes, "ssim") > self.frame_diff_threshold:
                return mid2
            else:
                return self.__start_bs(pivot_frame, frames, mid2 + 1, end)
        else:
            mid = (end + start) // 2
            if frame_diff(pivot_frame, frames[mid], pivot_frame.bboxes, "ssim") > self.frame_diff_threshold:
                return self.__start_bs(pivot_frame, frames, start, mid - 1)
            else:
                return self.__start_bs(pivot_frame, frames, mid + 1, end)

    def config(self, ocr_accept_threshold=None,
               str_diff_threshold=None,
               frame_diff_threshold=None,
               step_size=None,
               mid_attention=None,
               insert_costs=None,
               delete_costs=None,
               replace_costs=None):
        if ocr_accept_threshold is not None:
            self.ocr_accept_threshold = ocr_accept_threshold
        if str_diff_threshold is not None:
            self.str_diff_threshold = str_diff_threshold
        if frame_diff_threshold is not None:
            self.frame_diff_threshold = frame_diff_threshold
        if step_size is not None:
            self.step_size = step_size
        if mid_attention is not None:
            self.mid_attention = mid_attention
        if insert_costs is not None:
            self.insert_cost = insert_costs
        if delete_costs is not None:
            self.delete_cost = delete_costs
        if replace_costs is not None:
            self.replace_cost = replace_costs

import numpy as np
import cv2


def crop_background(image, bboxes):
    """Blackens all pixels in an image except the bounding boxes.
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


def frame_diff(img1, img2, bboxes):
    """Calculate the difference ratio in the bounding boxes of two images.
    Args:
        img1 (image): image1
        img2 (image): image2
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


def binary_search(frames, img, bboxes, search_for):
    #print(f"Fetching {search_for} in {len(frames)} frames")
    temp_lst = []
    h, w, _ = frames[0].shape
    for f in frames:
        temp_lst.append(f[int(0.75 * h):h, :])
    if search_for == "end":
        index = end_bs(temp_lst, 0, len(frames) - 1, img, bboxes)
        #print(len(frames[:int(index) + 1]))
        return frames[:int(index) + 1]
    else:
        index = start_bs(temp_lst, 0, len(frames) - 1, img, bboxes)
        #print(len(frames[int(index):]))
        return frames[int(index):]


def end_bs(frames, start, end, img, bboxes):
    if start >= end:
        return start - 1
    mid = start + (end - start + 1) // 2
    difference = frame_diff(img, frames[mid], bboxes)
    if difference < 0.25:
        return end_bs(frames, mid + 1, end, img, bboxes)
    else:
        return end_bs(frames, start, mid - 1, img, bboxes)


def start_bs(frames, start, end, img, bboxes):
    if start >= end:
        return end + 1
    mid = start + (end - start + 1) // 2
    difference = frame_diff(img, frames[mid], bboxes)
    if difference < 0.25:
        return start_bs(frames, start, mid - 1, img, bboxes)
    else:
        return start_bs(frames, mid + 1, end, img, bboxes)

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse


class Frame:
    # frame images and it's metadata
    image = None
    height = 0
    width = 0
    index = 0
    # processed image
    crp_img = None
    ocr_img = None
    # OCR
    bboxes = None
    text = None
    confidence = None

    def __init__(self, img, index=0):
        # frame metadata
        self.image = img
        self.index = index
        self.height, self.width, _ = img.shape

    def cropped_image(self):
        """
        Crops the image into the lower 3/4 of the original image.
        :return: The cropped image
        :rtype: numpy.ndarray
        """
        if self.crp_img is None:
            self.crp_img = self.image[int(0.75 * self.height): int(self.height+1), :]
        return self.crp_img

    def ocr_image(self):
        """
        Returns the image ready bo be OCRed.
        :return: The processed image
        :rtype: numpy.ndarray
        """
        if self.ocr_img is None:
            self.ocr_img = cv2.cvtColor(self.cropped_image(), cv2.COLOR_BGR2RGB)
        return self.ocr_img

    def set_ocr(self, bounding_boxes, text, confidence):
        self.bboxes = bounding_boxes
        self.text = text
        self.confidence = confidence

    def focused_image(self, bounding_boxes=None):
        if bounding_boxes is None:
            if self.bboxes is None:
                return self.cropped_image()
            else:
                bounding_boxes = self.bboxes

        images = []
        for bbox in bounding_boxes:
            # detect box area
            left = bbox[0][0]
            top = bbox[0][1]
            right = bbox[2][0]
            bottom = bbox[2][1]
            # set pixels in bbox
            images.append(self.cropped_image()[top:bottom+1, left:right+1])

        return images


def frame_diff(frame1, frame2, bounding_boxes=None, method="default"):
    images1 = frame1.focused_image(bounding_boxes=bounding_boxes)
    images2 = frame2.focused_image(bounding_boxes=bounding_boxes)

    if method == "default":
        area = 0
        diff = 0
        for i in range(len(images1)):
            diff_img = cv2.subtract(images1[i], images2[i])
            h, w, _ = images1[i].shape
            for x in range(w):
                for y in range(h):
                    if not np.all(diff_img[i][j] == 0):
                        diff += 1
            area += h * w
        return diff/area
    elif method == "ssim":
        min_sim = 1
        for i in range(len(images1)):
            min_sim = min(min_sim, ssim(images1[i], images2[i], channel_axis=2, data_range=255))
        return min_sim
    elif method == "mse":
        area = 0
        error = 0
        for i in range(len(images1)):
            error += mse(images1[i], images2[i])
            h, w, _ = images1[i].shape
            area += h * w
        return error / area
    else:
        raise ValueError("difference method invalid")

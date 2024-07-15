import easyocr

reader = easyocr.Reader(['ch_tra'])


def image2text(image):
    result = reader.readtext(image)

    ret_text = ""
    min_conf = 1.0
    for (bbox, text, prob) in result:
        ret_text += text
        min_conf = min(prob, min_conf)

    return ret_text, min_conf

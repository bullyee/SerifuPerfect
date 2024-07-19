# Serifu Perfect

## Dependencies:
> opencv-python
> > pip install opencv-python
>
> easyOCR
> > pip install easyocr

## SubtitleExtractor class
### from subtitle_extractor import SubtitleExtractor
### Class Functions:
* `SubtitleExtractor(video_path, output_path)` Constructor. `video_path` is the video file, and the `output_path` is the path to the output folder.
* `run()` Runs the Extractor
* `ocr_config(list)` Sets the model used for OCR. `list` should be the language list for ocr detection. see [easyOCR github page](<https://github.com/JaidedAI/EasyOCR>) for more information
* `config(*args)` Sets the parameters of run.
### Demo
```python
from subtitle_extractor import *
import os

os.makedirs("default")
s = SubtitleExtractor("temp/13b.mp4","default")
s.config(mid_attention=0.1)
s.run()

```



## <span style="color:red;">Run (decrypted):</span>
### <span style="color:red;">Videoprocessing.py </span>
<span style="color:red;">frame by frame, accurate processing.</span>
### <span style="color:red;">diff_model.py</span>
<span style="color:red;">compares images by difference value, speedy method</span>
### <span style="color:red;">binary_sort_model.py</span>
<span style="color:red;">hybrid method of the two.</span>

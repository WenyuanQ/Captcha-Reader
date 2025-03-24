import cv2
import numpy as np
import pytesseract
from pytesseract import Output

class CaptchaSimple(object):
    def __init__(self):
        pass

    def __call__(self, im_path, save_path=False):
        """
        Algo for inference
        args:
        im_path: .jpg image path to load and to infer
        save_path: output file path to save the one-line outcome
        """
        # pre-processing
        img = cv2.imread(im_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Tesseract
        custom_config = r'--oem 3 --psm 6' # 3: auto 6: simple scenario ocr
        result = pytesseract.image_to_string(binary, config=custom_config, lang='eng')

        # post processing
        predicted_text = result.strip()
        # print(f"Result: {predicted_text}")
        if save_path:
            with open(save_path, 'w') as f:
                f.write(predicted_text)
        return predicted_text


if __name__ == "__main__":
    captcha = CaptchaSimple()
    captcha('sampleCaptchas/input/input00.jpg')
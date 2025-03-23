# Captcha-Reader

## Abstract

In this work, the author proposes two approaches to detect a specific type of CAPTCHA pattern, which can be summarized as:

- The number of characters remains the same each time.
- The font and spacing are consistent.
- The background and foreground colors and textures are largely unchanged.
- There is no skew in the structure of the characters.
- The CAPTCHA generator creates strictly 5-character captchas, with each character being either an uppercase letter (A-Z) or a numeral (0-9).

### Approach 1 (Neural)
In the first approach, the author trains a convolutional classification network based on PyTorch to classify the 36 characters (A-Z, 0-9). Due to the highly consistent pattern of the CAPTCHA, the use of RNNs for sequence recognition is avoided to reduce the computational complexity.

### Approach 2 (Simple)
In the second approach, the author employs an open-source Tesseract-based string recognition model. This method not only can efficiently recognize the CAPTCHAs in the current task but also provides sufficient ground truth data to finetune the model from Approach 1 in future tasks, enhancing the scalability of this work.


# Methods

## Neural Mode

### Problem Analysis
The author found that all input images share this characteristic, which can be represented by the following code:

```python
split_points = [4, 13, 22, 31, 40, 49]
char_images = [
    image.crop((split_points[i], 0, split_points[i + 1], image.size[1])) 
    for i in range(5)
]
```
By using this segmentation method, we can obtain individual character images, as shown below:
<p align="center"><img src="tools/input_image_slide_concat.jpg"\></p>

With this approach, we can use a single letter classification network to solve the problem, thereby saving computation resources.

### Dataset Processing
- The dataset is indivisible because only all 25 images in the dataset include all 36 characters.
- The test data uses a single image from the input ```input100.jpg```.
- Also, the ground truth is missing output21.txt, which was manually completed.

### Model Architecture
The model architecture is shown below:
<p align="center"><img src="tools/captcha_model.onnx.png"\></p>

## Simple Mode
Tesseract is an open-source Optical Character Recognition (OCR) engine maintained by Google, originally developed by Hewlett-Packard. It is capable of recognizing text in a wide variety of languages and can extract text from images. Tesseract can handle different types of documents, including scanned files, handwritten text, and text in photographs. Author uses pytesseract API to call its Captcha recognition model. 


# Getting Started

## Neural Mode installation
pytorch

## Simple Mode installation
pytesseract安装 包括tesseract


# Things To Do
测试集缺失 一张图 100


# References


两种方法
    pytesseract安装 包括tesseract
    pytorch

切分方式的发现，避免使用rnn

数据集缺失output补全

测试集缺失 一张图 100


pip install -r requirements_simple.txt
需要额外安装 tesseract
# Linux
sudo apt-get update
sudo apt-get install tesseract-ocr
# macOS
brew install tesseract
# Windows
https://github.com/tesseract-ocr/tesseract
add the installation path into system env path


pip install -r requirements_neural.txt

导出onnx: python model.py

训练：python train.py

inference：
python demo.py sampleCaptchas/input/input00.jpg --model neural
python demo.py sampleCaptchas/test/input100.jpg --model neural --save 'sampleCaptchas/test/output.txt'
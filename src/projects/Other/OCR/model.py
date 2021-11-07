"""
## Import Modules
"""
import matplotlib.pyplot as plt
import PIL
import pytesseract
import re
%matplotlib inline
# prerequisites
# !pip install pytesseract
# install desktop version of pytesseract
"""
## Load the image
"""
img = PIL.Image.open('test.JPG')
plt.imshow(img)
"""
## Convert Image to Text
"""
# config
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'
TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'
text_data = pytesseract.image_to_string(img.convert('RGB'), lang='eng')
print(text_data)
"""
## Extract Specific Fields
"""
m = re.search("Name: (\w+)", text_data)
name = m[1]
name
m = re.search("Start Date: (\S+)", text_data)
start_date = m[1]
start_date
m = re.search("Geo-Coordinates: (\S+)", text_data)
coordinates = m[1]
coordinates

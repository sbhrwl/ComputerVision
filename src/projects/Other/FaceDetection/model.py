# %%
## pip install opencv-python

# %%
"""
## Import Modules
"""

# %%
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# %%
"""
## HAAR Cascade File Path
"""

# %%
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# %%
"""
## Load the Image
"""

# %%
image = cv2.imread('test image.jpg')
# convert to rgb
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

# %%
# resize the image
image = cv2.resize(image, (400, 600))

# %%
# convert to gray scale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')

# %%
"""
## Detect Faces
"""

# %%
faces = face_cascade.detectMultiScale(gray)

# %%
len(faces)

# %%
# diplay the faces in the image
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Faces ",image)
cv2.waitKey(0)

# %%

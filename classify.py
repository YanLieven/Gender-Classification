
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import face_recognition
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

print("[INFO] loading network...")
classifier = load_model(args['model'])

imag = face_recognition.load_image_file(args['image'])
top,right,bottom,left = face_recognition.face_locations(imag)[0]
face_image = imag[top:bottom,left:right]
pil_image = Image.fromarray(face_image)
face_name = 'face_'+args['image'][12:]
pil_image.save(face_name,'JPEG')

test_image = image.load_img(face_name, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

print("[INFO] classifying image...")
idx = result[0]
cs = ["Female","Male"]
label = cs[int(idx)]

im = cv2.imread(args['image'])
im = imutils.resize(im, width=400)
cv2.putText(im, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
cv2.imshow('im',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

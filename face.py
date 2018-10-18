import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import glob
import face_recognition
from PIL import Image

male_training = 'dataset/training_set/Male/'
male_test = 'dataset/test_set/Male/'

female_training = 'dataset/training_set/Female/'
female_test = 'dataset/test_set/Female/'

print('\n\n')

#male_training
data_path1 = os.path.join(male_training,'*g')
files = glob.glob(data_path1)
print('Checking for Male training faces:\n\n')
for f1 in files:
	image = face_recognition.load_image_file(f1)
	try:
		top,right,bottom,left = face_recognition.face_locations(image)[0]
		face_image = image[top:bottom,left:right]
		pil_image = Image.fromarray(face_image)
		pil_image.save('face_dataset/face_training/Male/'+f1[26:],'JPEG')
	except IndexError:
		print('No face detected on image: ' + f1)
print('\n\n')

#male_test
data_path2 = os.path.join(male_test,'*g')
files = glob.glob(data_path2)
print('Checking for Male test faces:\n\n')
for f1 in files:
	image = face_recognition.load_image_file(f1)
	try:
		top,right,bottom,left = face_recognition.face_locations(image)[0]
		face_image = image[top:bottom,left:right]
		pil_image = Image.fromarray(face_image)
		pil_image.save('face_dataset/face_test/Male/'+f1[22:],'JPEG')
	except IndexError:
		print('No face detected on image: ' + f1)
print('\n\n')

#female_training
data_path3 = os.path.join(female_training,'*g')
files = glob.glob(data_path3)
print('Checking for Female training faces:\n\n')
for f1 in files:
	image = face_recognition.load_image_file(f1)
	try:
		top,right,bottom,left = face_recognition.face_locations(image)[0]
		face_image = image[top:bottom,left:right]
		pil_image = Image.fromarray(face_image)
		pil_image.save('face_dataset/face_training/Female/'+f1[28:],'JPEG')
	except IndexError:
		print('No face detected on image: ' + f1)
print('\n\n')

#female_test
data_path4 = os.path.join(female_test,'*g')
files = glob.glob(data_path4)
print('Checking for Female test faces:\n\n')
for f1 in files:
	image = face_recognition.load_image_file(f1)
	try:
		top,right,bottom,left = face_recognition.face_locations(image)[0]
		face_image = image[top:bottom,left:right]
		pil_image = Image.fromarray(face_image)
		pil_image.save('face_dataset/face_test/Female/'+f1[24:],'JPEG')
	except IndexError:
		print('No face detected on image: ' + f1)
print('\n\n')
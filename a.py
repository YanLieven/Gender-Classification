import cv2

result1 = 1;
result2 = 1;
result3 = 0;
result4 = 0; 

font = cv2.FONT_HERSHEY_SIMPLEX

# img1
img1 = cv2.imread('dataset/predictions/actor.jpg')
img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5) 

if result1 == 1:
	stri = 'Male'
else:
	stri = 'Female'

textsize = cv2.getTextSize(stri, font, 1, 2)[0]
textX = int((img1.shape[1] - textsize[0]) / 2)
textY = int((img1.shape[0] + textsize[1]) / 2)

cv2.putText(img1, stri, (textX, textY), font, 1, (0, 255, 0), 2)
cv2.imshow('img1',img1)

#img2
img2 = cv2.imread('dataset/predictions/actor2.jpg')
img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5) 


if result2 == 1:
	stri = 'Male'
else:
	stri = 'Female'

textsize = cv2.getTextSize(stri, font, 1, 2)[0]
textX = int((img2.shape[1] - textsize[0]) / 2)
textY = int((img2.shape[0] + textsize[1]) / 2)

cv2.putText(img2, stri, (textX, textY), font, 1, (0, 255, 0), 2)
cv2.imshow('img2',img2)

#img3
img3 = cv2.imread('dataset/predictions/actress.jpg')
img3 = cv2.resize(img3, (0,0), fx=0.5, fy=0.5) 


if result3 == 1:
	stri = 'Male'
else:
	stri = 'Female'

textsize = cv2.getTextSize(stri, font, 1, 2)[0]
textX = int((img3.shape[1] - textsize[0]) / 2)
textY = int((img3.shape[0] + textsize[1]) / 2)

cv2.putText(img3, stri, (textX, textY), font, 1, (0, 255, 0), 2)
cv2.imshow('img3',img3)

#img4
img4 = cv2.imread('dataset/predictions/actress2.jpg')
img4 = cv2.resize(img4, (0,0), fx=0.5, fy=0.5)

if result4 == 1:
	stri = 'Male'
else:
	stri = 'Female'

textsize = cv2.getTextSize(stri, font, 1, 2)[0]
textX = int((img4.shape[1] - textsize[0]) / 2)
textY = int((img4.shape[0] + textsize[1]) / 2)

cv2.putText(img4, stri, (textX, textY), font, 1, (0, 255, 0), 2)
cv2.imshow('img4',img4)

cv2.waitKey(0)
cv2.destroyAllWindows()

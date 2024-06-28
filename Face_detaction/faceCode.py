import cv2

#STEP:1 Read The Image
img=cv2.imread("face.jpeg",cv2.IMREAD_COLOR)

#STEP:2 Load The Classifier
face_caascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#STEP:3 Convert Image Into Greyscale
grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#STEP:4 Use Classifier To Detect Object in Grayscale Image
        #detectMultiScale(Img_array,
        #                            scalingFactor(Default:1.1),
        #                            minimumNeighbours(default:3))
faces = face_caascade.detectMultiScale(grey_img,1.1,10)  #1.1,4
print(faces)

#STEP:5 plot Detected Object Using Classifier On the Actual Image

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,x+h),(255,0,0),1)

#STEP:6 Show the Iamge
cv2.imshow("Face Detector",img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os 
import pandas as pd 
import random
from keras.preprocessing.image import ImageDataGenerator 

###############PARAMERTERS######################
path = 'D:/Project/Traffic-Sign-Identification-using-CNN/Train' # folder with all the class folders
labelFile = 'D:/Project/Traffic-Sign-Identification-using-CNN/labels.csv' # file with all name of classes
batch_size_val=50  # how many to process together
steps_per_epoch_val=2000
epochs_val=10
imageDimensions = (32,32,3)
testRatio = 0.2 #if 1000 images then 200 will be testing images
validationRatio = 0.2 #if 1000 images then 20% of remaining 800 will be 160 for validation 

################### Importting of the images###########################
count = 0
images = [] # images
classNo = [] # labels
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

####################### Splitting the data ###########################
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
# X_train = Array of images to train
# y_train = Corresponding class Id

############################# to number of labels for each data set ##########################
print("Data Shapes")
print("Train",end="");
print(X_train.shape,y_train.shape)
print("Validation",end="");
print(X_validation.shape,y_validation.shape)
print("Test",end="");
print(X_test.shape,y_test.shape)
assert(X_train.shape[0]==y_train.shape[0]), "The number of images is not equal to the number of labels in training set"
assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images is not equal to the number of labels in validation set"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images is not equal to the number of labels in test set"
assert(X_train.shape[1:]==imageDimensions), "The dimensions of the images are wrong in training set"
assert(X_validation.shape[1:]==imageDimensions), "The dimensions of the images are wrong in validation set"
assert(X_test.shape[1:]==imageDimensions), "The dimensions of the images are wrong in test set" 

########################## Read the csv file ##########################
data = pd.read_csv(labelFile)
print("data shape ",data.shape, type(data))

########################### Display some samples of images of all the classes ##########################
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,300))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + data["Name"])
            num_of_samples.append(len(x_selected))

##################### display a bar chart showing no of samples for each category ##########################
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

########################### Preprocessing the images ##########################
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
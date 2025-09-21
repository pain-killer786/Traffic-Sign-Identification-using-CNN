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
def preprocessing(img):
    img = grayscale(img) # convert to grayscale
    img = equalize(img) # standardize the lighting in the image
    img = img/255 # normalize the values between 0 and 1 instead of 0 to 255
    return img
X_train = np.array(list(map(preprocessing, X_train)))  # to iterate and preprocess all images
X_validation = np.array(list(map(preprocessing, X_validation))) 
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("GrayScale Images", X_train[random.randint(0, len(X_train)-1)])  # To check if the training is done properly

############################ Add a depth of 1 to images ##########################
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

########################### Augmentation of images to make it more generic ##########################
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20) # Requesting Data Generator to Generate images Batch Size = No. Of images Created each time its called
X_batch, y_batch = next(batches) # to show augmented image samples
fig, axs = plt.subplots(1, 15, figsize=(20,5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]))
    axs[i].axis('off')
plt.show()
y_train = to_categorical(y_train, noOfClasses) # convert to one hot encoding
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

########################### Convolution Neural Network Model ##########################
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5) # This is the kernel that move around the image to get the features. This would remove 2 pixels from each border when using 32 32 image
    size_of_Filter2 = (3, 3)
    size_of_Pool = (2, 2) # scale down all feature map to generalize more, to reduce overfitting.
    no_Of_Nodes = 500 # no of nodes in fully connected layer

    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_Pool)) # Doesn't effect the depth/no of filters

    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_Pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    # Compile the model
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

############################ Training the model ##########################
model = myModel()
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val,epochs=epochs_val,
                    validation_data=(X_validation, y_validation), shuffle=1)

########################### Plotting the model accuracy and loss ##########################
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

########################### Testing the model ##########################
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])
# Save the model
pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0) # waits until a key is pressed
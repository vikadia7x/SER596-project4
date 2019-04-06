# -*- coding: utf-8 -*-
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from tkinter import *


def make_predictions():
    #print("Hello World")
    pred_dir = 'test_parent'
    classifier = Sequential()
    classifier = Sequential()
    # First Convolution Layer and Pooling Layer
    classifier.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(BatchNormalization())
    
    # Second Convolution Layer and Pooling Layer
    classifier.add(Conv2D(64,(3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(BatchNormalization())
    
    classifier.add(Conv2D(64,(3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(BatchNormalization())
    
    classifier.add(Conv2D(96,(3,3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(BatchNormalization())
    
    # Flattening Layer
    classifier.add(Flatten())
    
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 3, activation = 'softmax'))
    
    # Time to compile the network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    classifier.load_weights("./model/weights-Test-CNN.h5")
    
    test_data_scale = ImageDataGenerator(rescale = 1./255)
    
    test_generator = test_data_scale.flow_from_directory(
        directory=pred_dir,
        target_size=(64, 64),
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    ) 
    
    test_generator.reset()
    
    pred=classifier.predict_generator(test_generator,verbose=1,steps=306/32)
    print(pred)
    
    predicted_class_indices=np.argmax(pred,axis = 1)
    print(predicted_class_indices)
    
    train_data_scale = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    
    train_set = train_data_scale.flow_from_directory('./hci_plots/train_set',target_size = (64,64), batch_size = 32, class_mode = 'categorical')
    
    labels = (train_set.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    
    print(predictions,labels)
    #create_gui(predictions,labels)

def create_gui(predictions,labels):
    window = Tk()
    window.geometry('600x600')
 
    window.title("Welcome to Prediction app")
 
    #predictions=['something','one thing','another']
    for i in range(len(predictions)):
        exec('Label%d=Label(predictions,text="%s")\nLabel%d.pack()' % (i,predictions[i],i))
    window.mainloop()
    

if __name__ == "__main__":
    make_predictions()
    
 

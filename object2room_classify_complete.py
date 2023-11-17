#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:59:45 2020

@author: gpu-server
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import pandas as pd
# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list):
     # load the image
     data = pyplot.imread(filename)
     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)
     # show the plot
     pyplot.show()

# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
# load photograph
img = load_img('elephant.jpg')
img = img_to_array(img)
# make prediction
results = rcnn.detect([img], verbose=0)
# visualize the results
draw_image_with_boxes('elephant.jpg', results[0]['rois'])

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np 
# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
 
# define the test configuration
class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80
 
# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

# load room example
#img = load_img('/home/gpu-server/Mask_RCNN/home_data/bedroom1.jpg')
#img = img_to_array(img)
## make prediction
#results = rcnn.detect([img], verbose=0)
## get dictionary for first prediction
#r = results[0]
## show photo with bounding boxes, masks, class labels and scores
#display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


#predict object from 500 rooms and create csv file of object list
table1 = np.zeros((499,81), dtype = int)
for i in range (499):
    i = i+1
    img = load_img('/home/gpu-server/Mask_RCNN/room_data100/room' + str(i) + '.jpg')
    path = '/home/gpu-server/Mask_RCNN/room_data100/room' + str(i) + '.jpg'
    img = img_to_array(img)
    results = rcnn.detect([img], verbose=0)
    r = results[0]
    #display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    obj = np.unique(r['class_ids'])
    n = len(obj)
    for j in range (n):
        x = obj[j]
        if x > 0:
            table1[i-1,x] = 1
#    for j in range (n):
#    arr = np.reshape(obj, (1,n))
#        table1[i-1,j] = arr[0,j-1]

pd.DataFrame(table1).to_csv("onehot_object500_2.csv")

#Oblect detect for test data
table2 = np.zeros((24,81), dtype = int)
for i in range (24):
    i = i+1
    img = load_img('/home/gpu-server/Mask_RCNN/home_data/room' + str(i) + '.jpg')
    path = '/home/gpu-server/Mask_RCNN/home_data/room' + str(i) + '.jpg'
    img = img_to_array(img)
    results = rcnn.detect([img], verbose=0)
    r = results[0]
    #display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    obj = np.unique(r['class_ids'])
    n = len(obj)
    for j in range (n):
        x = obj[j]
        if x > 0:
            table2[i-1,x] = 1    
#    arr = np.reshape(obj, (1,n))
#    for j in range (n):
#        table2[i-1,j] = arr[0,j-1]

pd.DataFrame(table2).to_csv("onehot_object.csv")

# cnn model for classify scence from object list
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
import pandas as pd

trainpath = "/home/gpu-server/Mask_RCNN/dat_w_matlab/onehot_object500.csv"
train_label = "/home/gpu-server/Mask_RCNN/dat_w_matlab/room_label_500.csv"

testpath = "/home/gpu-server/Mask_RCNN/dat_w_matlab/testing_onehot_object.csv"
test_label = "/home/gpu-server/Mask_RCNN/dat_w_matlab/room.csv"

# load the dataset, returns train and test X and y elements
def load_dataset(trainpath, train_label, testpath, test_label):
	# load all train
    trainX = pd.read_csv(trainpath) 
    trainy = pd.read_csv(train_label)
    print(trainX.shape, trainy.shape)
	# load all test
    testX = pd.read_csv(testpath) 
    testy = pd.read_csv(test_label) 
    print(testX.shape, testy.shape)
	# zero-offset class values
#	trainy = trainy - 1
#	testy = testy - 1
#	# one hot encode y
#	trainy = to_categorical(trainy)
#	testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

#fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 100, 10
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    model.save_weights('onehot_room_500.h5')
    
    # evaluate model
    model.load_weights('onehot_room_500.h5')
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)
#    model.summary()
    return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=1):
    trainX, trainy, testX, testy = load_dataset(trainpath, train_label, testpath, test_label)
    trainX = np.expand_dims(trainX, axis=2)
    testX = np.expand_dims(testX, axis=2)
	# repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, trainX, trainy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
	# summarize results
    summarize_results(scores)

# run the experiment
run_experiment()

  
# test with random images
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
# load photograph
#img = load_img('kitchen.jpeg')
#img = img_to_array(img)
## make prediction
#results = rcnn.detect([img], verbose=0)
## visualize the results
#draw_image_with_boxes('kitchen.jpeg', results[0]['rois'])
#
#test_obj = np.unique(r['class_ids'])
#print(test_obj)

def prediction_model(test_obj):
    batch_size = 1
    n_timesteps, n_features = test_obj.shape[1], test_obj.shape[2]
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('onehot_room_500.h5')
    y = model.predict(test_obj, batch_size=batch_size,)
#    model.summary()
    return y


def run_test(repeats=1):
    table3 = np.zeros((1,81), dtype = int)
    trainX, trainy, testX, testy = load_dataset(trainpath, train_label, testpath, test_label)
    img = load_img('test9.jpg')
    img = img_to_array(img)
    results = rcnn.detect([img], verbose=0)
    draw_image_with_boxes('test9.jpg', results[0]['rois']) 
    r = results[0]
    test_obj = np.unique(r['class_ids'])
    print(test_obj)
    n = len(test_obj) 
    for j in range (n):
        x = test_obj[j]
        if x > 0:
            table3[0,x]=1
#    arr = np.reshape(test_obj, (1,n))
#    for j in range (n):
#        table3[0,j] = arr[0,j-1]
#    pd.DataFrame(table3).to_csv("onehot_Test_obj.csv")

    trainX = np.expand_dims(trainX, axis=2)
    test_obj = table3
    test_obj = np.expand_dims(test_obj, axis=2)
	# repeat experiment
    for n in range(repeats):
        y = prediction_model(test_obj)
        print(y)
        room_index = np.argmax(y, axis=1)
        print(room_index)
        if room_index == 0:
            print('bathroom')
        elif room_index == 1:
            print('bedroom')
        elif room_index == 2:
            print('diningroom')
        elif room_index == 3:
            print('kitchen')
        elif room_index == 4:
            print('livingroom')
        else:
            print("Unknown room")
            
run_test()    

def run_validation(repeats=1):

    trainX, trainy, testX, testy = load_dataset(trainpath, train_label, testpath, test_label)

#    arr = np.reshape(test_obj, (1,n))
#    for j in range (n):
#        table3[0,j] = arr[0,j-1]


    trainX = np.expand_dims(trainX, axis=2)
    test_obj = np.expand_dims(testX, axis =2)

	# repeat experiment
    for n in range(repeats):
        y = prediction_model(test_obj)
        print(y)
        room_index = np.argmax(y, axis=1)
        print(room_index)
        if room_index == 0:
            print('bathroom')
        elif room_index == 1:
            print('bedroom')
        elif room_index == 2:
            print('diningroom')
        elif room_index == 3:
            print('kitchen')
        elif room_index == 4:
            print('livingroom')
        else:
            print("Unknown room")
# run the experiment
run_validation()



   

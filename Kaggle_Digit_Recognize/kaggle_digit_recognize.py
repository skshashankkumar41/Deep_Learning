import cv2
import numpy as np
import os
from random import shuffle
import pickle
import tflearn
import pandas as pd
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import fully_connected,input_data,dropout
from tflearn.layers.estimator import regression

IMG_SIZE=28
LR=1e-3

MODEL_NAME = 'digit_recognize-{}-{}.model'.format(LR, '6conv-basic-kaggle-3')

training_data=pd.read_csv('Dataset/train.csv')
testing_data=pd.read_csv('Dataset/test.csv')

X_train=np.array(training_data.iloc[:,1:]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_train=training_data.iloc[:,0]
y_train = tflearn.data_utils.to_categorical(y_train, nb_classes=10)



X_test=np.array(testing_data.iloc[:,:]).reshape(-1,IMG_SIZE,IMG_SIZE,1)


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')


model=tflearn.DNN(convnet,tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

else:
    model.fit({'input':X_train},{'targets':y_train},n_epoch=10,snapshot_step=500,show_metric=True,run_id='MODEL_NAME')

    model.save(MODEL_NAME)


prediction=[]
image_id=list(range(1,28001))
for data in X_test:
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    prediction.append(np.argmax(model_out))



df=pd.DataFrame({'ImageId':image_id,'Label':prediction},index=None)

df.to_csv('Submission.csv')

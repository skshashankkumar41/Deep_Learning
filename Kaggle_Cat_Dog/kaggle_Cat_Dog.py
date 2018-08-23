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

TRAIN_DIR='C:/Users/SKS/Desktop/Projects/Python/Tensorflow/Kaggle_Cat_Dog/train'
TEST_DIR='C:/Users/SKS/Desktop/Projects/Python/Tensorflow/Kaggle_Cat_Dog/test'
IMG_SIZE=50
LR=1e-3

MODEL_NAME = 'dogsandcats-{}-{}.model'.format(LR, '6conv-basic-kaggle-2')

def image_label(img):
    word_label=img.split('.')[-3]
    if word_label=='dog':
        return [0,1]
    else:
        return [1,0]

def create_training_data():
    training_data=[]
    for img in os.listdir(TRAIN_DIR):
        label=image_label(img)
        path=os.path.join(TRAIN_DIR,img)
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('training_data_dog_cat.npy',training_data)
    return training_data

def create_testing_data():
    testing_data = []
    for img in sorted(os.listdir(TEST_DIR),key=lambda x: int(os.path.splitext(x.split('.')[0])[0])):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

train_data=create_training_data()
test_data=create_testing_data()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')


model=tflearn.DNN(convnet,tensorboard_dir='log')

X_train=np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X_test=np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)

y_train=[i[1] for i in train_data]



if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

else:
    model.fit({'input':X_train},{'targets':y_train},n_epoch=10,snapshot_step=500,show_metric=True,run_id='MODEL_NAME')

    model.save(MODEL_NAME)


prediction=[]
id=[]
for num,data in enumerate(test_data):
    img_num = data[1]
    img_data = data[0]

    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    prediction.append(np.argmax(model_out))
    id.append(img_num)


df=pd.DataFrame({'id':id,'label':prediction},index=None)

df.to_csv('Submission-2.csv')

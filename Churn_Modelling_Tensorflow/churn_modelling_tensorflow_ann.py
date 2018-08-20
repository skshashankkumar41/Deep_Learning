import pandas as pd
import numpy as np
import tensorflow as tf

df=pd.read_csv('Dataset/Churn_Modelling.csv')
X=df.iloc[:,3:13].values
y=df.iloc[:,13:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_1=LabelEncoder()
label_encoder_2=LabelEncoder()
onehot=OneHotEncoder(categorical_features=[1])
X[:,1]=label_encoder_1.fit_transform(X[:,1])
X[:,2]=label_encoder_2.fit_transform(X[:,2])
X=onehot.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)


hidden_layer_1_nodes=6
hidden_layer_2_nodes=6


n_classes=1
batch_size=10

x_tf=tf.placeholder('float',[None,11])
y_tf=tf.placeholder('float')

def next_batch(num, data, labels):

    idc = np.arange(0 , len(data))
    np.random.shuffle(idc)
    idc = idc[:num]
    data_shuffle = [data[ i] for i in idc]
    labels_shuffle = [labels[ i] for i in idc]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def model(data):
    hidden_layer_1={'weights':tf.Variable(tf.random_normal([11,hidden_layer_1_nodes])),
                    'biases':tf.Variable(tf.random_normal([hidden_layer_1_nodes]))}

    hidden_layer_2={'weights':tf.Variable(tf.random_normal([hidden_layer_1_nodes,hidden_layer_2_nodes])),
                    'biases':tf.Variable(tf.random_normal([hidden_layer_2_nodes]))}


    output_layer={'weights':tf.Variable(tf.random_normal([hidden_layer_2_nodes,n_classes])),
                  'biases':tf.Variable(tf.random_normal([n_classes]))}

    layer_1=tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    layer_1=tf.nn.relu(layer_1)

    layer_2=tf.add(tf.matmul(layer_1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    layer_2=tf.nn.relu(layer_2)

    output=tf.matmul(layer_2,output_layer['weights'])+output_layer['biases']

    return output

def train_network(x_tf):
    prediction=model(x_tf)
    cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=y_tf))
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    epochs=100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss=0
            for i in range(int(len(X_train)/batch_size)):
                epoch_x,epoch_y=next_batch(batch_size,X_train,y_train)
                k,c=sess.run([optimizer,cost],feed_dict={x_tf:epoch_x,y_tf:epoch_y})
                epoch_loss+=c

            print('Epoch', epoch+1, 'completed out of',epochs,'loss:',epoch_loss)


        new_pred=sess.run(prediction,feed_dict={x_tf:X_test})
        for i in range(len(new_pred)):
            if new_pred[i]>=0.5:
                new_pred[i]=1
            else:
                new_pred[i]=0

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, new_pred)

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, new_pred)

        print('Confusion Metrics:',cm)
        print('Accuracy:',accuracy)

train_network(x_tf)

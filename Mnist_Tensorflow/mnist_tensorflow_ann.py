import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('/tmp/data/',one_hot=True)

hidden_layer_1_nodes=500
hidden_layer_2_nodes=500
hidden_layer_3_nodes=500

n_classes=10
batch_size=100

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def model(data):
    hidden_layer_1={'weights':tf.Variable(tf.random_normal([784,hidden_layer_1_nodes])),
                    'biases':tf.Variable(tf.random_normal([hidden_layer_1_nodes]))}

    hidden_layer_2={'weights':tf.Variable(tf.random_normal([hidden_layer_1_nodes,hidden_layer_2_nodes])),
                    'biases':tf.Variable(tf.random_normal([hidden_layer_2_nodes]))}

    hidden_layer_3={'weights':tf.Variable(tf.random_normal([hidden_layer_2_nodes,hidden_layer_3_nodes])),
                    'biases':tf.Variable(tf.random_normal([hidden_layer_3_nodes]))}

    output_layer={'weights':tf.Variable(tf.random_normal([hidden_layer_3_nodes,n_classes])),
                  'biases':tf.Variable(tf.random_normal([n_classes]))}

    layer_1=tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    layer_1=tf.nn.relu(layer_1)

    layer_2=tf.add(tf.matmul(layer_1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    layer_2=tf.nn.relu(layer_2)

    layer_3=tf.add(tf.matmul(layer_2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    layer_3=tf.nn.relu(layer_3)

    output=tf.matmul(layer_3,output_layer['weights'])+output_layer['biases']

    return output

def train_network(x):
    prediction=model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    epochs=10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss=0
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                k,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss+=c

            print('Epoch', epoch+1, 'completed out of',epochs,'loss:',epoch_loss)

        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy= tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_network(x)

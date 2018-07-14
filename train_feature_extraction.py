import pickle
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
    train = pickle.load(f)


# TODO: Split data into training and validation sets.
X = train['features']
y = train['labels']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
resize = tf.image.resize_images(x, (227, 227))


# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resize, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)


# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], 43)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(43))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])


# TODO: Train and evaluate the feature extraction model.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

batch_size = 128
epochs = 10
num_examples = len(X_train)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    print('Training...')
    print()
    
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for start in range(0, num_examples, batch_size):
            end = start + batch_size
            batch_x, batch_y = X_train[start:end], y_train[start:end]
            sess.run(training_operation, feed_dict={x:batch_x, y:batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
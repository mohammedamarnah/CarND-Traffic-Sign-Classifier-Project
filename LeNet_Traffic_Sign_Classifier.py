# Import all the needed libraries

print("Importing Libraries...")

import pickle
import random
import cv2
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

print("All libraries imported.")

# # # # # # # # # # # # # # # # # #

print("Loading the Data set...")

training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Data set loaded successfully.")

# # # # # # # # # # # # # # # # # #

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# # # # # # # # # # # # # # # # # #

classmap = []
classes_data = open("./signnames.csv", "r")
lines = classes_data.readlines()
for i in range(len(lines)):
    if lines[i][2:][0] == ',':
        classmap.append(lines[i][3:-1])
    else:
        classmap.append(lines[i][2:-1])

plt.figure(figsize=(16, 16))
for i in range(20):
    plt.subplot(8, 3, i+1)
    idx = random.randint(0, n_train-2)
    image = X_train[idx]
    plt.imshow(image)
    plt.axis('off')
    plt.title(classmap[y_train[idx]+1])
plt.show()

classes = np.unique(y_train)
plt.figure(figsize=(12, 4))
for i in range(len(classes)):
    plt.bar(classes[i], (y_train == classes[i]).sum())
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.title("Distribution of the dataset")
plt.show()

# # # # # # # # # # # # # # # # # #

index = random.randint(0, len(X_train))
fig, axs = plt.subplots(1,4, figsize=(10, 3))
axs = axs.ravel()

print("Converting to grayscale...")

# CONVERT TO GRAYSCALE
X_train_color = X_train
X_test_color = X_test
X_valid_color = X_valid

X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)
X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)

print('RGB shape:', X_train_color.shape)
print('Grayscale shape:', X_train_gray.shape)

axs[0].axis('off')
axs[0].set_title('RGB')
axs[0].imshow(X_train_color[index].squeeze(), cmap='gray')

axs[1].axis('off')
axs[1].set_title('Gray')
axs[1].imshow(X_train_gray[index].squeeze(), cmap='gray')

X_train = X_train_gray
X_test = X_test_gray
X_valid = X_valid_gray

print("Normalizing the images...")

# NORMALIZE
X_train_normalized = (X_train - 128)/128 
X_test_normalized = (X_test - 128)/128
X_valid_normalized = (X_valid - 128)/128


print("Original shape:", X_train.shape)
print("Normalized shape:", X_train_normalized.shape)

axs[2].axis('off')
axs[2].set_title('Normalized')
axs[2].imshow(X_train_normalized[index].squeeze(), cmap='gray')

axs[3].axis('off')
axs[3].set_title('Original(grayscale)')
axs[3].imshow(X_train[index].squeeze(), cmap='gray')

X_train = X_train_normalized
X_test = X_test_normalized
X_valid = X_valid_normalized

# # # # # # # # # # # # # # # # # #

def LeNet(x):    
    mu = 0
    sigma = 0.1
    
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)

    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    conv2 = tf.nn.relu(conv2)

    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    fc0   = flatten(conv2)
    
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    fc1    = tf.nn.relu(fc1)

    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    fc2    = tf.nn.relu(fc2)
    fc2    = tf.nn.dropout(fc2, keep_prob)
    
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    
    return logits

EPOCHS = 53 # because 53 is a prime number :P 
BATCH_SIZE = 128
rate = 0.001
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

X_train, y_train = shuffle(X_train, y_train)

xs = []
ys = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        xs.append(i)
        ys.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

plt.xlabel("EPOCH")
plt.ylabel("Accuracy")
plt.plot(xs, ys)

# # # # # # # # # # # # # # # # # #

print("Running the model on the test set...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet.meta')
    saver2.restore(sess, "./lenet")
    test_accuracy = evaluate(X_test_normalized, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))

# # # # # # # # # # # # # # # # # #

fig, axs = plt.subplots(1,6, figsize=(10, 3))
axs = axs.ravel()
i = 0

new_images = []

for file in os.listdir("./german"):
    if (file[0] != '.'):
        image = mpimg.imread("german/"+file)
        #image = cv2.imread("german/"+file)
        #print(image.shape)
        axs[i].set_title(i)
        axs[i].axis('off')
        axs[i].imshow(image)
        new_images.append(image)
        i += 1
print(new_images[0].shape)

print("Processing new images...")

new_images = np.asarray(new_images)
print(new_images.shape)

nImagesProcessed = np.sum(new_images/3, axis = 3, keepdims = True)
nImagesNormalized = (nImagesProcessed - 128) / 128

print(nImagesNormalized.shape)

labels = [0, 18, 22, 33, 14, 14]

print("Running the model on new images...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    acc = evaluate(nImagesNormalized, labels)
    print("Test Set Accuracy = {:.3f}".format(acc))

softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: nImagesNormalized, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: nImagesNormalized, keep_prob: 1.0})

    
    fig, axs = plt.subplots(len(new_images),2, figsize=(12, 14))
    fig.subplots_adjust(hspace = .4, wspace=.2)
    axs = axs.ravel()

    for i, image in enumerate(new_images):
        axs[2*i].axis('off')
        axs[2*i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[2*i].set_title('input')
        
        guess1 = my_top_k[1][i][0]
        index1 = np.argwhere(y_valid == guess1)[0]
        axs[2*i+1].axis('off')
        axs[2*i+1].imshow(X_valid[index1].squeeze(), cmap='gray')
        axs[2*i+1].set_title('guess: {} ({:.0f}%)'.format(guess1, 100*my_top_k[0][i][0]))

fig, axs = plt.subplots(6,2, figsize=(9, 19))
axs = axs.ravel()

for i in range(len(my_softmax_logits)*2):
    if i%2 == 0:
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(new_images[i//2], cv2.COLOR_BGR2RGB))
    else:
        axs[i].bar(np.arange(n_classes), my_softmax_logits[(i-1)//2]) 
        axs[i].set_ylabel('Softmax probability')
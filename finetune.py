"""
With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. 
Specify the configuration settings at the beginning according to your 
problem.
This script was written for TensorFlow 1.0 and come with a blog post 
you can find here:
  
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert 
contact: f.kratzert(at)gmail.com
"""
import os
from glob import glob

import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator

"""
Configuration settings
"""
img_path = '/home/spartonia/datasets/CervicalCancerScreening/299x299/train/Type_{}'


def create_train_val_sets():
    """Creates train validation text files."""
    train = []
    validation = []
    for label in [1, 2, 3]:
        target_label = img_path.format(label) + '/*.jpg'
        lst = np.array(glob(target_label)).reshape(-1, 1)
        indices = np.random.permutation(lst.shape[0])
        train_idx, test_idx = int(lst.shape[0]*.7), int(lst.shape[0]*.85)
        train_set, val_set = lst[indices[:train_idx]], lst[indices[train_idx:]]
        train_set = np.insert(train_set, 1, label-1, axis=1)
        val_set = np.insert(val_set, 1, label-1, axis=1)
        train.append(train_set)
        validation.append(val_set)

    train = np.vstack(train)
    validation = np.vstack(validation)

    np.savetxt('train.txt', train, delimiter=',', fmt="%s")
    np.savetxt('validation.txt', validation, delimiter=',', fmt="%s")

# create_train_val_sets()

# Path to the textfiles for the trainings and validation set
train_file = 'train.txt'
val_file = 'validation.txt'

# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 3
train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'logs',
)
checkpoint_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'checkpoints',
)

# Create parent path if it doesn't exist
if not os.path.exists(checkpoint_path): os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if
            v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)

# Add gradients to summary  
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy_loss', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
train_writer = tf.summary.FileWriter(logdir=filewriter_path + '/train')
test_writer = tf.summary.FileWriter(logdir=filewriter_path + '/test')


# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initialize the data generator separately for the training and validation set
train_generator = ImageDataGenerator(train_file,
                                     horizontal_flip=True, shuffle=True)
val_generator = ImageDataGenerator(val_file, shuffle=True)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(
    train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(
    np.int16)

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    # writer.add_graph(sess.graph)
    train_writer.add_graph(sess.graph)
    test_writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1
        val_step = 1

        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            feed_dict_train = {x: batch_xs,
                         y: batch_ys,
                         keep_prob: dropout_rate}
            i_global, _, s = sess.run(
                [global_step, train_op, merged_summary],
                feed_dict=feed_dict_train)

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                # calculate accuracy on training batch
                batch_acc, summary_str = sess.run(
                    [accuracy, merged_summary],
                    feed_dict=feed_dict_train
                )
                train_writer.add_summary(summary_str, global_step=i_global)

                # calculate accuracy for validation batch
                batch_tx, batch_ty = val_generator.next_batch(batch_size)
                val_acc, summary_str_val = sess.run(
                    [accuracy, merged_summary],
                    feed_dict={x: batch_tx, y:batch_ty, keep_prob: 1.}
                )
                test_writer.add_summary(summary_str_val, global_step=i_global)

            step += 1

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()

        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(
            epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))


import argparse
import csv
import datetime
import gc
import glob
import json
import math
import operator
import os
import pickle
import re
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from itertools import zip_longest
from operator import itemgetter

import cv2
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm as tqdm
from imutils import paths
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import (callbacks, constraints, initializers, layers,
                              optimizers, regularizers, utils)
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        LearningRateScheduler, ModelCheckpoint,
                                        TensorBoard)
from tensorflow.keras.layers import (GRU, LSTM, Activation, Add, Bidirectional,
                                     Conv1D, Conv2D, Dense, Dropout, Embedding,
                                     Flatten, GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Input, Layer,
                                     MaxPooling1D, Reshape, SpatialDropout1D,
                                     add, concatenate, multiply)
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image, sequence, text
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence, plot_model
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score

from vqa_util.vqa_model import build_model

print("Load Questions and Image Training Data")
with open('preprocessed_data/vqa_raw_train2014_top1000.json', 'rb') as f:
    questions_train, answer_train, answers_train, images_train = joblib.load(f)

with open('preprocessed_data/vqa_raw_val2014_top1000.json', 'rb') as f:
    questions_val, answer_val, answers_val, images_val = joblib.load(f)

print("Load Text Tokenizers and Encoders")
tok = text.Tokenizer(filters='')
# load from disk
with open('vqa_objects/text_tokenizer.pkl', 'rb') as f:
    tok = joblib.load(f)

# load from disk
with open('vqa_objects/tokenised_data_post.pkl', mode='rb') as f:
    question_data_train, question_data_val = pickle.load(f)

# load from disk
with open('vqa_objects/labelencoder.pkl', 'rb') as f:
    labelencoder = joblib.load(f)

def get_answers_matrix(answers, encoder):
	'''
	One-hot-encodes the answers

	Input:
		answers:	list of answer
		encoder:	a scikit-learn LabelEncoder object
  
	Output:
		A numpy array of shape (# of answers, # of class)
	'''
	y = encoder.transform(answers) #string to numerical class
	nb_classes = encoder.classes_.shape[0]
	Y = utils.to_categorical(y, nb_classes)
	return Y

# Prepare data matrices
print("Prepare data matrices")

sss = StratifiedShuffleSplit(n_splits=1, test_size= 0.25,random_state=42)

for train_index, val_index in sss.split(images_train, answer_train):
    TRAIN_INDEX = train_index
    VAL_INDEX = val_index

# image data
image_list_tr, image_list_vl = np.array(images_train)[TRAIN_INDEX.astype(int)], np.array(images_train)[VAL_INDEX.astype(int)]

# question data
question_tr, question_vl = question_data_train[TRAIN_INDEX], question_data_train[VAL_INDEX]

# answer data
answer_matrix = get_answers_matrix(answer_train, labelencoder)
answer_tr, answer_vl = answer_matrix[TRAIN_INDEX], answer_matrix[VAL_INDEX]

print("Create TF Dataset")
BATCH_SIZE = 300
BUFFER_SIZE = 5000

# loading the numpy files
def map_func(img_name, ques, ans):
    img_tensor = np.load('features/' + img_name.decode('utf-8').split('.')[0][-6:] + '.npy')
    return img_tensor, ques, ans

dataset_tr = tf.data.Dataset.from_tensor_slices((image_list_tr, question_tr, answer_tr))

# Use map to load the numpy files in parallel
dataset_tr = dataset_tr.map(lambda item1, item2, item3: tf.numpy_function(
    map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset_tr = dataset_tr.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_tr = dataset_tr.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

dataset_vl = tf.data.Dataset.from_tensor_slices((image_list_vl, question_vl, answer_vl))

# Use map to load the numpy files in parallel
dataset_vl = dataset_vl.map(lambda item1, item2, item3: tf.numpy_function(
    map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Shuffle and batch
dataset_vl = dataset_vl.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_vl = dataset_vl.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print("Model Creation")

BATCH_SIZE = 300
BUFFER_SIZE = 5000

# params 1
max_answers = 1000
max_seq_len = 22
vocab_size  = len(tok.word_index) + 1
EPOCHS      = 60

dim_d       = 512
dim_k       = 256
l_rate      = 1e-4
d_rate      = 0.5
reg_value   = 0.01

base_path = './temp'

# create model
model = build_model(max_answers, max_seq_len, vocab_size, dim_d, dim_k, l_rate, d_rate, reg_value)
model.summary()

steps_per_epoch = int(np.ceil(len(image_list_tr)/BATCH_SIZE))
boundaries      = [50*steps_per_epoch]
values          = [l_rate, l_rate/10]

# we reduce the l_rate after 50th epoch (from 1e-4 to 1e-5)
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer        = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

loss_object      = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='auto')

print("Checkpoint Manager")
checkpoint_directory = base_path+"/training_checkpoints/"+str(l_rate)+"_"+str(dim_k)
SAVE_CKPT_FREQ = 5
ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_directory, max_to_keep=3)

# Create stateful metrics that can be used to accumulate values during training and logged at any point:

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
train_score = F1Score(num_classes=max_answers, average='micro', name='train_score')
val_score = F1Score(num_classes=max_answers, average='micro', name='val_score')

# %%
train_log_dir = base_path+'/logs/'+str(l_rate)+"_"+str(dim_k)+'/train'
val_log_dir   = base_path+'/logs/'+str(l_rate)+"_"+str(dim_k)+'/validation'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

print("Define training functions")
# @tf.function
def train_step(model, img, ques, ans, optimizer):
    with tf.GradientTape() as tape:
        # forward pass
        predictions = model([img, ques], training=True)
        loss = loss_object(ans, predictions)

    # backward pass
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # record results
    train_loss(loss)
    train_score(ans, predictions)

    # all gradients
    grads_ = list(zip(grads, model.trainable_variables))
    return grads_

def test_step(model, img, ques, ans):
    predictions = model([img, ques])
    loss = loss_object(ans, predictions)

    # record results
    val_loss(loss)
    val_score(ans, predictions)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print("Restored from {}".format(manager.latest_checkpoint))
    START_EPOCH = int(manager.latest_checkpoint.split('-')[-1]) * SAVE_CKPT_FREQ
    print("Resume training from epoch: {}".format(START_EPOCH))
else:
    print("Initializing from scratch")
    START_EPOCH = 0

for epoch in range(START_EPOCH, EPOCHS):

    start = time.time()

    for img, ques, ans in (dataset_tr):
        grads = train_step(model, img, ques, ans, optimizer)

    # tensorboard  
    with train_summary_writer.as_default():
        # Create a summary to monitor cost tensor
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar('f1_score', train_score.result(), step=epoch)
        # Create summaries to visualize weights
        for var in model.trainable_variables:
            tf.summary.histogram(var.name, var, step=epoch)
        # Summarize all gradients
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad, step=epoch)

    for img, ques, ans in (dataset_vl):
        test_step(model, img, ques, ans)

    # tensorboard
    with val_summary_writer.as_default():
        # Create a summary to monitor cost tensor
        tf.summary.scalar('loss', val_loss.result(), step=epoch)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar('f1_score', val_score.result(), step=epoch)

    template = 'Epoch {}, loss: {:.4f}, f1_score: {:.4f}, val loss: {:.4f}, val f1_score: {:.4f}, time: {:.0f} sec'
    print (template.format(epoch + 1,
                         train_loss.result(), 
                         train_score.result(),
                         val_loss.result(), 
                         val_score.result(),
                         (time.time() - start)))

    # Reset metrics every epoch
    train_loss.reset_states()
    train_score.reset_states()
    val_loss.reset_states()
    val_score.reset_states()

    # save checkpoint every SAVE_CKPT_FREQ step
    ckpt.step.assign_add(1)
    if int(ckpt.step) % SAVE_CKPT_FREQ == 0:
        manager.save()
        print('Saved checkpoint.')

model.save('hierarchical_vqa_model')

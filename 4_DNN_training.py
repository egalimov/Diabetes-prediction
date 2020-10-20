# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:48:37 2020

@author: Evgeny Galimov, PhD
"""

import pandas as pd
import numpy as np
import os
import random

from sklearn import model_selection
import pandas
from pandas.plotting import scatter_matrix
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, SpatialDropout2D
import tensorflow as tf
tf.__version__
import keras
import keras.backend as K
from tensorflow.keras import regularizers



save = "4_dnn_L1_l2"

random.seed(2) # Python
np.random.seed(2017)
seed = 7
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTHONHASHSEED'] = '0'
shuffle=False
 

#########################################
# Data load from feature selection/training/test split 
data_train = pd.read_csv("C:\\1_data\\For_github\\2_Diabetes_prediction\\all_samples\\dnn\\1_train_set.csv", sep=',', header=[0])
data_test = pd.read_csv("C:\\1_data\\For_github\\2_Diabetes_prediction\\all_samples\\dnn\\1_test_set.csv", sep=',', header=[0])
res_file_name = 'C:\\1_data\\For_github\\2_Diabetes_prediction\\all_samples\\dnn\\1_results_FS_train_test_split\\'+ save+ '.csv'


X_train = data_train[['AgeAge_65_70','AgeAge_70_75','AgeAge_75plus','Sexmale','EthnicityEthnic_group_2','EthnicityEthnic_group_3',
 'SmokingCurrent','SmokingEx',
 'G2','G3','G4','G5','G6','G7','G8','G9','G10',
 'G11','G12','G13','G14','G15','G16','G17','G18','G19','G20','G21','G22','G23',
 'G24','G25','G26','G27','G28','G29','G30','G31','G32','G33','G34','G35',
 'G36','G37']]
Y_train = data_train[['outcome']]
X_test = data_test[['AgeAge_65_70','AgeAge_70_75','AgeAge_75plus','Sexmale','EthnicityEthnic_group_2','EthnicityEthnic_group_3',
 'SmokingCurrent','SmokingEx',
 'G2','G3','G4','G5','G6','G7','G8','G9','G10',
 'G11','G12','G13','G14','G15','G16','G17','G18','G19','G20','G21','G22','G23',
 'G24','G25','G26','G27','G28','G29','G30','G31','G32','G33','G34','G35',
 'G36','G37']]
Y_test = data_test[['outcome']]
Y_train_t = np_utils.to_categorical(Y_train, 2)
Y_test_t = np_utils.to_categorical(Y_test, 2)


######################################################

if not os.path.exists(res_file_name):
    res_m = {"input": [],
         "d4": [],
         "d1": [],
         "drop1": [],
         "d2": [],
         "drop2": [],
         "d3": [],
         "l1_h": [],
         "l2_h": [],
         "l1_l": [],
         "l2_l": [],
         "addlayer": [],
         "epoch": [],
         "acc": [],
  #       "acc_val": [],
         "acc_test": []}
    r = pd.DataFrame(res_m)
    r.to_csv(res_file_name, index = False)
    
     

import argparse
## initialize parser
parser = argparse.ArgumentParser()
## add arguments
parser.add_argument("param1", help="input size", type=int)
parser.add_argument("param2", help="d4 size", type=int)
parser.add_argument("param3", help="d1 size", type=int)
parser.add_argument("param4", help="drop1 value", type=float)
parser.add_argument("param5", help="d2 size", type=int)
parser.add_argument("param6", help="drop2 value", type=float)
parser.add_argument("param7", help="d3 size", type=int)
parser.add_argument("param8", help="l1 value for internal layer", type=float)
parser.add_argument("param9", help="l2 value for internal layer", type=float)
parser.add_argument("param10", help="l1 value for last layer", type=float)
parser.add_argument("param11", help="l2 value for last layer", type=float)
parser.add_argument("param12", help="the number of internal layers", type=int)
parser.add_argument("param13", help="number of epochs", type=int)

# pass arguments to the list args
args = parser.parse_args()



###  Variables
#fixed
input_s = args.param1
d4 = args.param2

# varied
d1 = args.param3                  
drop1 = args.param4
d2 = args.param5
drop2 = args.param6   
d3 = args.param7
l1_h = args.param8 
l2_h = args.param9 
l1_l = args.param10 
l2_l = args.param11
addlayer = args.param12  
epoch = args.param13



# ###  Variables
# #fixed
# input_s = 42
# d4 = 2

# # varied
# d1 = 268            
# drop1 = 0.2
# d2 = 500
# drop2 = 0   
# d3 = 500 
# l1_h = 0 
# l2_h = 0 
# l1_l = 0 
# l2_l = 0
# addlayer = 1  
# epoch = 1000


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# Storing metrics
metricsFile = "C:\\Users\\evgeny\\Dropbox\\metrics.csv"
from keras.callbacks import CSVLogger
csv_logger = CSVLogger(metricsFile, separator=",", append=True)
callbacksList = []
callbacksList.append(csv_logger)

############################################################################
############################################################################
############################################################################

my_init = keras.initializers.glorot_uniform(seed=1)
def create_classical_model(input_s,d1,drop1,d2,drop2,d3,d4,l1_3,l2_3,l1_5,l2_5, addlayer,my_init):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(d1, input_dim=input_s, activation="relu", kernel_initializer=my_init))                                                      #1
    model.add(tf.keras.layers.Dropout(drop1))
    for z in range(0, addlayer):
        model.add(tf.keras.layers.Dense(d2, activation="relu", kernel_initializer=my_init, kernel_regularizer=regularizers.l1_l2(l1=l1_h, l2=l2_h)))                                                                                          #2
    model.add(tf.keras.layers.Dropout(drop2))                                                                                              #4
    model.add(tf.keras.layers.Dense(d3, activation="relu", kernel_initializer=my_init, kernel_regularizer=regularizers.l1_l2(l1=l1_l, l2=l2_l)))    #5
    model.add(tf.keras.layers.Dense(d4, activation="softmax", kernel_initializer=my_init))                                               #6
    return model



model = create_classical_model(input_s,d1,drop1,d2,drop2,d3,d4,l1_h,l2_h,l1_l,l2_l,addlayer,my_init)
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=['acc'])
history = model.fit(X_train, Y_train_t, batch_size=1238, epochs=epoch, verbose=2, callbacks=callbacksList)
h = history.history
acc = h["acc"][epoch-1]
loss, acc_test = model.evaluate(X_test, Y_test_t, verbose=0)

sum_line = str(input_s) +','+ str(d4) +','+ str(d1) +','+ str(drop1) +','+ str(d2) +','+ str(drop2) +','+ str(d3) +','+ str(l1_h) +','+ str(l2_h) +','+ str(l1_l) +','+ str(l2_l) +','+ str(addlayer) +','+ str(epoch) +','+ str(acc) +','+ str(acc_test)+'\n' 

with open(res_file_name, 'a') as file_object:
    file_object.write(sum_line)











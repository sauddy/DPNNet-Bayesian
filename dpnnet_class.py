## Importing the necessary modules ##

import numpy as np
import pandas as pd
import glob as glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

## For MLP build
import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

## Modules for TensorFlow probabilty
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


print("We have imported the dpnnet_class")


class dpnnet():
  def __init__(self,data_path):
    self.folder_address = data_path

  def data_filtering(self,filtering=True):
    ## concatinating all the .csv files into one csv
    dataset0 = pd.concat(map(pd.read_csv, glob.glob(self.folder_address + '*.csv')), ignore_index=True)

    ################## DATA Filtering #############
    if filtering is True:

        # Filtering 1
        dataset0 = dataset0[dataset0['Dust_gap_1'] > 0.05]  # filtering out very narrow gaps
        dataset = pd.concat([dataset0], ignore_index=True).sort_index()  # important when merging multiple datasets
        df = shuffle(dataset, random_state=3)
        dataset = df.reset_index(drop=True)
        dataset['Planet_Mass'] = dataset['Planet_Mass'] / (3 * 10**-6)  # writing in unit of earth mass

        # Filtering 2 (removing simulation with more than two gaps)
        dataset = dataset[dataset['#_DG'] <= 2]  # keeping one and two dust gap disks
        dataset = dataset.drop(columns=['Sample#']) # dropping the Sample#

        # dataset_filtered.to_csv('data_folder/dataset_filered.csv')   # saving the filtered data as csv file for future reference
        dataset = dataset.drop(columns=['Gas_gap_1', 'Dust_depth_1', 'Dust_depth_2', 'Gas_depth_1', '#_DG', '#_GG'])  # droping the unimportant columns
        # dataset.to_csv('../data_folder/dataset_filered.csv')
        # dataset = dataset[['Sample#','Planet_Mass']] # droping the unimportant columns
    #     dataset = dataset.sort_values(by="Sample#")

        ## cleaning the data##
        dataset.isna().sum()  # summing the number of na
        dataset0 = dataset.dropna()

    return dataset0
  
  @staticmethod
  def normalization(x):
    ## Normalization wrt to the train dataset // Train stat is defined as a global varibale
    ## It is calculated in the data processing routine
    return (x - train_stats['mean']) / train_stats['std']

  def data_processing(self,test_data_size = None):

    '''
    Input: dataset and the test_data_size 
    Output: X_train, X_test, y_train, y_test
    Labels is hardwired to planet Mass (The user needs to modify that if needed)
    '''
    # Reading the dataset for processing
    dataset = self.data_filtering()

    ## seleting labels and the features
    y = dataset.Planet_Mass # labels
    X = dataset.drop(["Planet_Mass"], axis=1) ## features

    if test_data_size == None:
      test_data_size = 0.2 ## the default is set to 20 %
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_size, random_state=42)
    global train_stats ## making it global 
    train_stats = X_train.describe()
    train_stats = train_stats.transpose()
    
    
    ## Normalization of data using the mean and std of the training dataset
    print("The feature variables are normalized for better scaling")
    X_train_scaled= self.normalization(X_train)
    X_test_scaled = self.normalization(X_test)

    print("[INFO] data processing done, returns the splitted data...")
    print("[INFO] X_train and X_test data are normalized ...")

    return X_train_scaled, X_test_scaled, y_train, y_test

  


  def build_model(self,unit1=None, unit2=None,learning_rate=None,Bayesian = False,input_shape = None):
    print("We are now using the DPNNet Architecture")
    '''
    This is the DPPNet sequencial MLP architecture. 
    Input the X_trained_data for input shape
    This model is hardwired with its architeture
    The architecture is decided by using Keras Tuner
    
    '''
    ## For the shape of the input 
    X_train_scaled, X_test_scaled, y_train, y_test = self.data_processing()
    
    if unit1 == None and unit2 == None:
      unit1 = 256 ## these are the default values used in DPNNet paper1
      unit2 = 128 ## these are the default values used in DPNNet paper1
  
    learning_rate = 0.001
    # learning_rate = 0.0001 #optimized using keras optimizer
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=[len(X_train_scaled.keys())])) 
    model.add(layers.Dense(unit1, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.0001),input_shape=[len(X_train_scaled.keys())]))
    if Bayesian == False:
      model.add(layers.Dense(unit2, activation=tf.nn.relu,kernel_regularizer=regularizers.l2(0.0001)))
    elif Bayesian == True:
      print("Implementing Bayesian in the last layer")
      model.add(tfp.layers.DenseVariational(unit2, activation=tf.nn.relu, make_posterior_fn = self.posterior_mean_field, make_prior_fn= self.prior_trainable))
    model.add(layers.Dense(1))
  
    optimizer = tf.keras.optimizers.Adam(learning_rate,decay=learning_rate/200)
    model.compile(loss='mean_squared_error',
                optimizer=optimizer , metrics=['mean_absolute_error', 'mean_squared_error'])
    return model



  # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
  @staticmethod
  def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                      scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])
  # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
  @staticmethod
  def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])


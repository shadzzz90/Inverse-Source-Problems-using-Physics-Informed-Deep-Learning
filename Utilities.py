
#This script contains utilies used while traninig and testing

# Author: Shadab Anwar Shaikh (sshaikh4@charlotte.edu)


import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




# Define initial parameters

def get_parameters(csvfile,dof=None):

    param_df = pd.read_csv(csvfile)

    if dof == 1:

        param_dict = {i: j for i, j in zip(param_df['input_param'], param_df['param_value'])}

        return param_dict



# get simulation data

def get_simdata(csvfile, dof=None):

    simdata_df = pd.read_csv(csvfile)

    if dof == 1:

        sim_t = np.array(simdata_df['time'])

        sim_disp = np.array(simdata_df['x'])

        sim_vel = np.array(simdata_df['x_dot'])
        sim_F = np.array(simdata_df['input'])

        return sim_t, sim_disp, sim_vel, sim_F


    if dof == 2:

        sim_t = np.array(simdata_df['time'])

        sim_disp1 = np.array(simdata_df['x1'])
        sim_disp2 = np.array(simdata_df['x2'])

        sim_vel1 = np.array(simdata_df['x_dot1'])
        sim_vel2 = np.array(simdata_df['x_dot2'])

        sim_F1 = np.array(simdata_df['input1'])
        sim_F2 = np.array(simdata_df['input2'])


        return sim_t, sim_disp1, sim_disp2, sim_vel1, sim_vel2, sim_F1, sim_F2




# Create model

def create_model(nn_architecture , activation = None ,initializer = None ):

    inputs = tf.keras.Input(shape=nn_architecture[0], name="input_layer")

    x = tf.keras.layers.Dense(nn_architecture[1], activation=activation,kernel_initializer=initializer, name="dense_layer_1")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    for i, nunits in enumerate(nn_architecture[2:-1]):
        x = tf.keras.layers.Dense(nunits, activation=activation, name=f"dense_layer_{i + 2}")(x)
        x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(nn_architecture[-1], name="output_layer")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='My_Dense_model')

    model.summary()

    return model


if __name__ == "__main__":

    initializer = tf.keras.initializers.he_normal()

    mymodel = create_model([1,15,30,60,120,240,120,60,30,15,3],activation='elu',initializer=initializer)

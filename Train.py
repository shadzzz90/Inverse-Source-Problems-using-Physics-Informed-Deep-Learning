import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
import Utilities as MyUtils
from datetime import datetime


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

print(physical_devices)

tf.keras.backend.set_floatx('float64')

LAMBDA = 0.1

file_extension = '_DuffingNL_normal2_Dense_e60000_b250_p80013_lr0.001_elu_he_ini_BN_samp500_t50_lamb0.1'
param_csvfile = 'D://InverseSourceProblems//duffingNormal_sim_param.csv'
sim_csvfile = 'D://InverseSourceProblems//duffingNormal_sim_data.csv'


# Loss Calculator


def calculate_loss(model, sim_vel,sim_disp,sim_force_term,sim_t,param_dict):

    index_of_zero = np.array(np.where(sim_t == 0))

    sim_t = tf.convert_to_tensor(sim_t.reshape(-1,1), dtype=tf.float64)
    sim_disp = tf.convert_to_tensor(sim_disp.reshape(-1,1), dtype=tf.float64)
    sim_vel = tf.convert_to_tensor(sim_vel.reshape(-1,1), dtype=tf.float64)
    sim_force_term = tf.convert_to_tensor(sim_force_term.reshape(-1,1), dtype=tf.float64)

    with tf.GradientTape() as tape:

        tape.watch(sim_t)
        output = model(sim_t)
        pred_disp = tf.reshape(output[:, 0],[-1, 1])
        pred_vel = tf.reshape(output[:, 1], [-1, 1])
        pred_forcing_term = tf.reshape(output[:, 2],[-1, 1])

    pred_acc = tape.gradient(pred_vel, sim_t)

    physics_loss = tf.square(pred_acc+param_dict['delta']*pred_vel+param_dict['alpha']*pred_disp++param_dict['beta']*tf.pow(pred_disp,3) - pred_forcing_term)

    disp_loss = tf.square(pred_disp-sim_disp)

    vel_loss = tf.square(pred_vel-sim_vel)

    # force_loss = tf.square(pred_forcing_term-sim_force_term)

    if not np.size(index_of_zero):

        return tf.reduce_mean(disp_loss +vel_loss+ LAMBDA*physics_loss), \
               tf.reduce_mean(physics_loss), tf.reduce_mean(disp_loss),tf.reduce_mean(vel_loss)


    else:

        disp_ICloss = tf.square(pred_disp[index_of_zero[0][0]] - param_dict['x0'])

        vel_ICloss = tf.square(pred_vel[index_of_zero[0][0]] - param_dict['x_dot_0'])

        return tf.reduce_mean(disp_loss +vel_loss+ disp_ICloss +vel_ICloss+ LAMBDA*physics_loss), \
               tf.reduce_mean(physics_loss),tf.reduce_mean(disp_loss),tf.reduce_mean(vel_loss)



def training(model, lr, epochs, batchsize):

   sim_t,sim_disp,sim_vel,sim_force_term = MyUtils.get_simdata(sim_csvfile,dof=1)


   param_dict = MyUtils.get_parameters(param_csvfile,dof=1)

   index_vec = np.arange(0,len(sim_t))

   np.random.shuffle(index_vec)

   rand_index = index_vec[0:batchsize]



   total_loss_lst = []
   physics_loss_lst = []
   disp_loss_lst =[]
   vel_loss_lst = []
   disp_ICloss_lst = []
   vel_ICloss_lst = []
   vel_term_lst = []
   grads_total_record = []


   optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
 
   epochs_list = [i for i in range(epochs)]

   # Getting the current date and time


   dt = datetime.now()

   # print("Date and time is:", dt)

   with open('output.txt', 'w') as f:
       f.writelines('DuffingNL3_Dense_e60000_b250_p80013_lr0.001_elu_he_ini_BN_samp500_t50_lamb0.1 \n')
       f.write(str(dt))


       for epoch in range(epochs):

          print(f"\nStart of epoch {epoch}")

          with tf.GradientTape(persistent=True) as tape:
              total_loss, physics_loss, disp_loss,force_loss = calculate_loss(model=model, sim_vel=sim_vel[rand_index],
                                          sim_disp=sim_disp[rand_index],sim_force_term =sim_force_term[rand_index],
                                          sim_t=sim_t[rand_index], param_dict=param_dict)

          # grads_ff = tape.gradient(physics_loss, model_ff.trainable_weights)
          # optimizer_ff.apply_gradients(zip(grads_ff, model_ff.trainable_weights))


          grads_total = tape.gradient(total_loss, model.trainable_weights)

          optimizer.apply_gradients(zip(grads_total, model.trainable_weights))


          print(f'Epoch No {epoch} completed\n')


          total_loss_lst.append(total_loss.numpy())
          physics_loss_lst.append(physics_loss.numpy())
          disp_loss_lst.append(disp_loss.numpy())
          vel_loss_lst.append(force_loss.numpy())
          grads_total_record.append(grads_total[4].numpy())
          # disp_ICloss_lst.append(disp_ICloss.numpy())
          # vel_ICloss_lst.append(vel_ICloss.numpy())
          # forcing_term_lst.append(forcing_term.numpy())


       with open("total_loss_lst", "w") as fp:
           json.dump(total_loss_lst, fp)

       with open("physics_loss_lst", "w") as fp2:
           json.dump(total_loss_lst, fp2)

       with open("disp_loss_lst", "w") as fp3:
           json.dump(disp_loss_lst, fp3)

       with open("vel_loss_lst", "w") as fp4:
           json.dump(vel_loss_lst, fp4)


       list_for_plot = list([total_loss_lst, physics_loss_lst, disp_loss_lst, vel_loss_lst])

       list_plot_label = ['Total Loss', 'Physics Loss', 'Displacement Data Loss', 'Velocity Data Loss']

       list_plot_title = ['Total Loss vs Epochs', 'Physics Loss vs Epochs', 'Displacement Data Loss vs Epochs', 'Velocity Data Loss vs Epochs']

       list_plot_filename = [f'Total_Loss{file_extension}', f'Physics_Loss{file_extension}', f'Displacement_Data_Loss{file_extension}', f'Velocity_Data_Loss{file_extension}']


       for i in range(0,len(list_for_plot)):

           plt.figure(i, figsize=(15, 15))
           plt.plot(epochs_list, list_for_plot[i], '-r')
           plt.xlabel('Epochs')
           plt.ylabel(list_plot_label[i])
           plt.title(list_plot_title[i])
           plt.savefig(list_plot_filename[i]+'.png')

           # Getting the current date and time
       dt = datetime.now()
       # print("Date and time is:", dt)
       f.write(str(dt))
       # grads_total_record = np.array(grads_total_record)
       # np.save('all_grads.npy',grads_total_record)

if __name__ == "__main__":

    initializer = tf.keras.initializers.he_normal()

    mymodel = MyUtils.create_model([1,15,30,60,120,240,120,60,30,15,3],activation='elu',initializer=initializer)

    # mymodel.load_weights(f'./main_model{file_extension}')

    training(mymodel,lr=0.001,epochs=60000, batchsize=250)

    mymodel.save_weights(f'./main_model{file_extension}')



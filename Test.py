import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import Utilities as MyUtils
import os
import matplotlib


tf.keras.backend.set_floatx('float64')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.rc('font', size=12)
plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams['text.usetex'] = True

cwdir = 'D://InverseSourceProblems//Experiment01//'
file_extension = '_DuffingSMD_comb_Dense_e60000_b250_p80013_lr0.001_elu_he_ini_BN_samp500_t50_lamb0.1'
param_csvfile = cwdir+'duffingSMD_comb_sim_param.csv'
sim_csvfile = cwdir+'duffingSMD_comb_sim_data.csv'



sim_t,sim_disp,sim_vel,sim_F = MyUtils.get_simdata(sim_csvfile,dof=1)

sim_t = sim_t.reshape(-1,1)

sim_disp = sim_disp.reshape(-1,1)
sim_vel = sim_vel.reshape(-1,1)


sim_t_lst = list(sim_t)

sim_disp_lst = list(sim_disp)

sim_vel_lst = list(sim_vel)

sim_t_lst,sim_disp_lst,sim_vel_lst = zip(*sorted(zip(sim_t_lst,sim_disp_lst,sim_vel_lst)))

sim_t = np.array(sim_t_lst)


sim_disp = np.array(sim_disp_lst)

sim_vel = np.array(sim_vel_lst)

param_dict = MyUtils.get_parameters(param_csvfile,dof=1)

initializer = tf.keras.initializers.he_normal()

model = MyUtils.create_model([1,15,30,60,120,240,120,60,30,15,3],activation='elu',initializer=initializer)


model.load_weights(f'{cwdir}main_model{file_extension}')


output = model(sim_t)

pred_disp = output[:, 0]
pred_vel = output[:, 1]
pred_ff = output[:, 2]


gauss_noise = 0

sim_disp = sim_disp + gauss_noise

sim_vel = sim_vel + gauss_noise

# plt.figure(1,figsize=(15,15))
# plt.plot(sim_t,sim_disp,'.r',sim_t,pred_disp,'-b',linewidth = 2,markersize = 10)
# plt.xlabel('t(s)')
# plt.ylabel('displacement(m)')
# plt.title('Displacement(m) vs time(s)')
# plt.legend(['simulated displacement', 'predicted displacement'])
# plt.savefig(f'Displacement_prediction{file_extension}.svg')

plt.figure(2,figsize=(6,6))
plt.plot(sim_t,sim_F,'-r',sim_t,pred_ff,'--b',linewidth = 2,markersize = 10)
plt.xlabel('time')
plt.ylabel('forcing function')
plt.grid(True, which="both",linestyle = '--')
str_lt_x_dot = r"$\dot{x}_0$"
plt.title(r"$\alpha $ = {} , $\beta $ = {} , $\delta $ = {}, $x_0 $ = {}, {} = {}".format(param_dict['alpha'],
                                                                                       param_dict['beta'],
                                                                                       param_dict['delta'],
                                                                                       round(param_dict['x0'],2),str_lt_x_dot,
                                                                                       round(param_dict['x_dot_0'],2)))
plt.legend(['actual', 'NN pred.'])
# plt.show()
plt.savefig(f'FF{file_extension}.eps',format='eps')

# plt.figure(3,figsize=(15,15))
# plt.plot(sim_t,sim_vel,'.r', sim_t,pred_vel,'-b',linewidth = 2,markersize = 10)
# plt.xlabel('t(s)')
# plt.ylabel('velocity(m/s)')
# plt.title('Velocity (m/s) vs time(s)')
# plt.legend(['simulated velocity', 'predicted velocity'])
# plt.savefig(f'Sim_Vel_vs_Predicted_Vel{file_extension}.svg')
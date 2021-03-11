import time
from pynufft import NUFFT
import numpy
import pandas as pd
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import argparse as args


def image_dimension_speed_test(dim_1, dim_2, nd_cpu, nd_gpu, plot_name, color_label):
    dim_speed = []
    jd_const = []
    count = 0
    for i in range(len(dim_1)):
        for j in range(len(dim_2)):
            for l in range(101):
                Nd = dim_1[i]
                Kd = dim_2[j]
                Jd = (4, 4)
                if count != len(nd_cpu):
                    executionTime = nd_cpu[count] / nd_gpu[count]
                    count += 1
                    if l != 0:
                        dim_speed.append(executionTime)
            std_dev_dim = numpy.std(dim_speed)
            average_speed = sum(dim_speed) / len(dim_speed)
            if i == (len(dim_1) - 1):
                plot_name.errorbar(Nd[0], average_speed, yerr=std_dev_dim, color=f'{color_label[j]}', fmt='-o',
                                   label=f'k_dim = {Kd[0]}, Jd = {Jd[0]}')
            else:
                plot_name.errorbar(Nd[0], average_speed, yerr=std_dev_dim, color=f'{color_label[j]}', fmt='-o')
            print(f'execution time is {average_speed} seconds when Nd = {Nd} and Kd = {Kd} and Jd = {Jd}')
            jd_const.append({"cpu_speed": average_speed, "Nd": Nd, 'Kd': Kd, 'Jd': Jd})
            dim_speed.clear()
    return plot_name, jd_const


def kd_speed_test(dim_1, dim_2, kd_cpu, kd_gpu, plot_name, color_label):
    nd_constant = []
    dim_speed = []
    count = 0
    for i in range(len(dim_1)):
        for j in range(len(dim_2)):
            for l in range(101):
                Kd = dim_1[i]
                Nd = dim_2[j]
                Jd = (4, 4)
                if count != len(kd_cpu):
                    executionTime = kd_cpu[count] / kd_gpu[count]
                    count += 1
                    if l != 0:
                        dim_speed.append(executionTime)
            std_dev_dim = numpy.std(dim_speed)
            average_speed = sum(dim_speed) / len(dim_speed)
            if i == (len(dim_1) - 1):
                plot_name.errorbar(Kd[0], average_speed, yerr=std_dev_dim, color=f'{color_label[j]}', fmt='-o',
                                   label=f'N_dim = {Nd[0]}, Jd = {Jd[0]}')
            else:
                plot_name.errorbar(Kd[0], average_speed, yerr=std_dev_dim, color=f'{color_label[j]}', fmt='-o')
            print(f'execution time is {average_speed} seconds when Kd = {dim_1[i]} and Nd = {dim_2[j]} and '
                  f'Jd = {Jd}')
            nd_constant.append({"cpu_speed": average_speed, "Nd": Nd, 'Kd': Kd, 'Jd': Jd})
            dim_speed.clear()
    return plot_name, nd_constant


def jd_speed_test(dim_1, dim_2, jd_cpu, jd_gpu, plot_name, color_label):
    dim_speed = []
    kd_const = []
    count = 0
    for i in range(len(dim_1)):
        for j in range(len(dim_2)):
            for l in range(101):
                Jd = dim_1[i]
                Nd = (128, 128)
                Kd = dim_2[j]
                if count != len(jd_cpu):
                    executionTime = jd_cpu[count] / jd_gpu[count]
                    count += 1
                    if l != 0:
                        dim_speed.append(executionTime)
            std_dev_dim = numpy.std(dim_speed)
            average_speed = sum(dim_speed) / len(dim_speed)
            if i == (len(dim_1) - 1):
                plot_name.errorbar(Jd[0], average_speed, yerr=std_dev_dim, color=f'{color_label[j]}', fmt='-o',
                                   label=f'N_dim = {Nd[0]}, Kd = {Kd[0]}')
            else:
                plot_name.errorbar(Jd[0], average_speed, yerr=std_dev_dim, color=f'{color_label[j]}', fmt='-o')
            print(f'execution time is {average_speed} seconds when Jd = {dim_1[i]} and Nd = {Nd} and '
                  f'Kd = {dim_2[j]}')
            kd_const.append({"cpu_speed": average_speed, "Nd": Nd, 'Kd': Kd, 'Jd': Jd})
            dim_speed.clear()
    return plot_name, kd_const


img_dimensions = [(64, 64), (128, 128), (256, 256)]
color = ['red', 'green', 'yellow', 'blue', 'orange']
k_d = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
jd = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

nd_cpu = numpy.load('/home/babji/Desktop/forward_/forward_cpu_nd_exe_time.npy')
nd_gpu = numpy.load('/home/babji/Desktop/forward_gpu_/forward_gpu_nd_exe_time.npy')

kd_cpu = numpy.load('/home/babji/Desktop/forward_/forward_cpu_kd_exe_time.npy')
kd_gpu = numpy.load('/home/babji/Desktop/forward_gpu_/forward_gpu_kd_exe_time.npy')

jd_cpu = numpy.load('/home/babji/Desktop/forward_/forward_cpu_jd_exe_time.npy')
jd_gpu = numpy.load('/home/babji/Desktop/forward_gpu_/forward_gpu_jd_exe_time.npy')

# Graph plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig, (ax1) = plt.subplots(1, 1)
ax1, kd_const = image_dimension_speed_test(img_dimensions, k_d, nd_cpu, nd_gpu, ax1, color)
ax1.legend(loc="upper left")
ax1.set_title('Speed with respect to I_dim')
ax1.set_xlabel('Image dimension')
ax1.set_ylabel('Time')

ax2, nd_constant = kd_speed_test(k_d, img_dimensions, kd_cpu, kd_gpu, ax2, color)
ax2.legend(loc="upper right")
ax2.set_title('Speed with respect to kd')
ax2.set_xlabel('kd')
ax2.set_ylabel('Time')

ax3, jd_const = jd_speed_test(jd, k_d, jd_cpu, jd_gpu, ax3, color)
ax3.legend(loc="upper right")
ax3.set_title('Speed with respect to jd')
ax3.set_xlabel('jd')
ax3.set_ylabel('Time')
# fianl=kd_const+nd_constant+jd_const
'''column_names=['cpu_speed','Nd','Kd','Jd']
Nd_constant_df=pd.DataFrame(nd_constant,columns=column_names)
Nd_constant_df.to_csv("/home/babji/Ndcpuforward.csv",index=False)

Kd_constant_df=pd.DataFrame(kd_const,columns=column_names)
Kd_constant_df.to_csv("/home/babji/Kdcpuforward.csv",index=False)

Jd_constant_df=pd.DataFrame(jd_const,columns=column_names)
Jd_constant_df.to_csv("/home/babji/Jdcpuforward.csv",index=False)'''

plt.savefig('/home/babji/Desktop/forward_ratio.png', dpi=100)
plt.show()

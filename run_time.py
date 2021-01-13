import time
from pynufft import NUFFT
import numpy
from matplotlib import pyplot as plt
import argparse as args


def image_dimension_speed_test(dim_1, dim_2, plot_name, color_label):
    dim_speed = []
    for i in range(len(dim_1)):
        for j in range(len(dim_2)):
            for l in range(100):
                A = NUFFT()
                om = numpy.random.randn(10, 2)
                Nd = dim_1[i]
                Kd = dim_2[j]
                Jd = (4, 4)
                A.plan(om, Nd, Kd, Jd)
                x = numpy.random.randn(*Nd)
                startTime = time.time()
                y = A.forward(x)
                executionTime = (time.time() - startTime)
                dim_speed.append(executionTime)

            std_dev_dim = numpy.std(dim_speed)
            average_speed = sum(dim_speed) / len(dim_speed)
            if i == (len(dim_1) - 1):
                plot_name.errorbar(Nd[0], average_speed, yerr=std_dev_dim, color=f'{color_label[j]}', fmt='-o',
                                   label=f'k_dim = {Kd[0]}, Jd = {Jd[0]}')
            else:
                plot_name.errorbar(Nd[0], average_speed, yerr=std_dev_dim, color=f'{color_label[j]}', fmt='-o')
            print(f'execution time is {average_speed} seconds when Nd = {Nd} and Kd = {Kd} and Jd = {Jd}')
            dim_speed.clear()
    return plot_name


def kd_speed_test(dim_1, dim_2, plot_name, color_label):
    dim_speed = []
    for i in range(len(dim_1)):
        for j in range(len(dim_2)):
            for l in range(100):
                A = NUFFT()
                om = numpy.random.randn(10, 2)
                Kd = dim_1[i]
                Nd = dim_2[j]
                Jd = (4, 4)
                A.plan(om, Nd, Kd, Jd)
                x = numpy.random.randn(*Nd)
                startTime = time.time()
                y = A.forward(x)
                executionTime = (time.time() - startTime)
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
            dim_speed.clear()
    return plot_name


def jd_speed_test(dim_1, dim_2, plot_name, color_label):
    dim_speed = []
    for i in range(len(dim_1)):
        for j in range(len(dim_2)):
            for l in range(100):
                A = NUFFT()
                om = numpy.random.randn(10, 2)
                Jd = dim_1[i]
                Nd = (128, 128)
                Kd = dim_2[j]
                A.plan(om, Nd, Kd, Jd)
                x = numpy.random.randn(*Nd)
                startTime = time.time()
                y = A.forward(x)
                executionTime = (time.time() - startTime)
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
            dim_speed.clear()
    return plot_name

img_dimensions = [(64, 64), (128, 128),(256, 256), (512, 512), (1024, 1024)]
color = ['red', 'green', 'yellow', 'blue', 'orange']
k_d = [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
jd = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

# Graph plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig, (ax1) = plt.subplots(1, 1)
ax1 = image_dimension_speed_test(img_dimensions, k_d, ax1, color)
ax1.legend(loc="upper left")
ax1.set_title('Speed with respect to I_dim')
ax1.set_xlabel('Image dimension')
ax1.set_ylabel('Time')

ax2 = kd_speed_test(k_d, img_dimensions, ax2, color)
ax2.legend(loc="upper right")
ax2.set_title('Speed with respect to kd')
ax2.set_xlabel('kd')
ax2.set_ylabel('Time')

ax3 = jd_speed_test(jd, k_d, ax3, color)
ax3.legend(loc="upper right")
ax3.set_title('Speed with respect to jd')
ax3.set_xlabel('jd')
ax3.set_ylabel('Time')

plt.show()






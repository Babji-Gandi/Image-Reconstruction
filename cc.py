from pynufft import NUFFT, helper
import numpy
import time

A = NUFFT(helper.device_list()[0])
A.device
om = numpy.random.randn(10,2)
Nd = (64,64)
Kd = (128,128)
Jd = (6,6)

A.plan(om, Nd, Kd, Jd)
x=numpy.random.randn(*Nd)
image=A.to_device(x)
startTime_1 = time.time()
y = A.forward(x)
executionTime_1 = (time.time() - startTime_1)
print(f'execution time is {executionTime_1} seconds')

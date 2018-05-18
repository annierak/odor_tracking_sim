import scipy
import time

size = 1000000

values = scipy.random.randn(size)
constant = 10.
fill_up = scipy.zeros(size)

start = time.time()
for i in range(len(values)):
    fill_up[i] = values[i]
fill_up = constant*fill_up

end = time.time();print(end-start)

values = scipy.random.randn(size)
constant = 10.
fill_up = scipy.zeros(size)

start = time.time()
for i in range(len(values)):
    fill_up[i] = constant*values[i]

end = time.time();print(end-start)

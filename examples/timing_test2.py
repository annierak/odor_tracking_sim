import time
import scipy

long_sparse_array = scipy.zeros(1000000)
indices_to_fill = scipy.random.choice(range(10000),20,replace=False)
long_sparse_array[indices_to_fill] = scipy.random.randn(20)
replacement_values = scipy.random.choice(range(10),100)

def counting_method(long_sparse_array,replacement_values):
    start = time.time()
    booll = long_sparse_array>0
    num_replacements = sum(booll)
    long_sparse_array[booll] =replacement_values[0:num_replacements]
    end = time.time()
    print('counting method duration: '+str(end-start))
    return long_sparse_array

def blind_method(long_sparse_array,replacement_values):
    start = time.time()
    replacement_counter = 0
    booll = long_sparse_array>0
    for index in range(len(long_sparse_array)):
        if booll[index]:
            long_sparse_array[index] = replacement_values[replacement_counter]
            replacement_counter +=1
    end = time.time()
    print('blind method duration: '+str(end-start))
    return long_sparse_array

new_array0= counting_method(long_sparse_array,replacement_values)

new_array1= blind_method(long_sparse_array,replacement_values)

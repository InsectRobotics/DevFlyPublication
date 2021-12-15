import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

plt.style.use('ggplot')
dataset = 'MNIST'
method = 'BioHash'
hash_length = 16
embedding_size = 20*hash_length
sampling_ratio = 0.05
max_index = 10000

result_path = os.path.join('./results4/' + dataset + '/' + str(hash_length) + '/' + str(embedding_size) + '/' + str(sampling_ratio))
with open(os.path.join(result_path + '/MAPs_per_iteration' + method + str(max_index) + '.pkl'), 'rb') as record_file:
    data = pickle.load(record_file)
pass
print(data)

time_path = os.path.join('./results4/' + dataset + '/models/')
with open(os.path.join(time_path + method + '_time_' + str(embedding_size) + '_' + str(sampling_ratio) + '.pkl'), 'rb') as record_file:
    time = pickle.load(record_file)
pass


time2 = np.zeros([100])
for i in range(100):
    time2[i] = np.sum(time[0: (i + i*100 + 100)])

time3 = np.zeros([101])
time3[0] = 0
time3[1:] = time2
data2 = np.zeros([101])
data2[0] = 0
data2[1:] = data[method]
data = data[method]
plt.plot(time3, data2)


#################### flylshdevelope
method = 'FlylshDevelope'
result_path = os.path.join('./results4/' + dataset + '/' + str(hash_length) + '/' + str(embedding_size) + '/' + str(sampling_ratio))
with open(os.path.join(result_path + '/MAPs_per_iteration' + method + str(max_index) + '.pkl'), 'rb') as record_file:
    data = pickle.load(record_file)
pass

time_path = os.path.join('./results4/' + dataset + '/models/')
with open(os.path.join(time_path + method + '_time_' + str(embedding_size) + '_' + str(sampling_ratio) + '.pkl'), 'rb') as record_file:
    time = pickle.load(record_file)
pass

time2 = np.zeros([80])
for i in range(80):
    time2[i] = np.sum(time[0: (i + i*(np.shape(time)[0] // 80) + (np.shape(time)[0] // 80))])


time3 = np.zeros([81])
time3[0] = 0
time3[1:] = time2
data2 = np.zeros([81])
data2[0] = 0
data2[1:] = data['BioHash(FlylshDevelope)']
data = data['BioHash(FlylshDevelope)']
plt.plot(time3, data2)
val = data2[-1]


#################### flylshdevelopthreshold
method = 'FlylshDevelopThreshold'
result_path = os.path.join('./results4/' + dataset + '/' + str(hash_length) + '/' + str(embedding_size) + '/' + str(sampling_ratio))
with open(os.path.join(result_path + '/MAPs_per_iteration' + method + str(max_index) + '.pkl'), 'rb') as record_file:
    data = pickle.load(record_file)
pass


time_path = os.path.join('./results4/' + dataset + '/models/')
with open(os.path.join(time_path + method + '_time_' + str(embedding_size) + '_' + str(sampling_ratio) + '.pkl'), 'rb') as record_file:
    time = pickle.load(record_file)
pass


time2 = np.zeros([80])
for i in range(80):
    time2[i] = np.sum(time[0: (i + i*(np.shape(time)[0] // 80) + (np.shape(time)[0] // 80))])


time3 = np.zeros([81])
time3[0] = 0
time3[1:] = time2
data2 = np.zeros([81])
data2[0] = 0
data2[1:] = data['BioHash(FlylshDevelopThreshold)']
data = data['BioHash(FlylshDevelopThreshold)']
plt.plot(time3, data2)

plt.axhline(val, linestyle='--', color='orange', linewidth=0.7)
plt.axhline(data2[-1], linestyle='--', color='green', linewidth=0.7)
plt.legend(['BioHash', 'Method 1', 'Method 2'])
plt.xlabel('Time (s)')
plt.ylabel('Mean Average Precision (mAP)')
plt.show()
from lshutils import plot_results, plothlcurve
import os
import numpy as np
import pickle

data_set_name =  'MNIST' #'CIFAR10' #'GLOVE' # 'SIFT' # 'CIFAR10' #'MNIST' # 'CIFAR10' #
max_index = 10000
sampling_ratio = 0.10
ratio = 20
sorted_data = True
shift = True
TIME_OF_RECORDING = '2021-12-09_20-53-32'
keys = ['Fly', 'FlylshDevelope', 'FlylshDevelopThreshold',
        'FlylshDevelopThresholdRandomChoice']
know_name = {'LSH': 'LSH', 'Fly': 'FlyLSH', 'FlylshDevelope': 'Method 1', 'FlylshDevelopThreshold': 'Method 2',
            'FlylshDevelopThresholdRandomChoice': 'Method 3', 'DenseFly':'DenseFly'}
legends = [know_name[x] for x in keys]
result_path = os.path.join("../recording/DevFly", data_set_name, str(sampling_ratio), str(ratio), str(sorted_data),
                           str(shift),  TIME_OF_RECORDING)
with open(os.path.join(result_path, 'all_MAPs' + str(max_index) + '.pkl'), 'rb') as record_file:
    all_MAPs = pickle.load(record_file)
code_dimensions = list(np.array(sorted(all_MAPs.keys())) * ratio)
figure_bokeh, results_stat = plot_results(all_MAPs, x_numbers=code_dimensions, plot_width=800, plot_height=400, name=data_set_name,
                                          keys=keys, legends=legends,
                                          legend_visible=True)
with open(os.path.join(result_path, 'all_retriving_time' + str(max_index) + '.pkl'), 'rb') as record_file:
    all_retriving_time = pickle.load(record_file)
figure_bokeh, results_stat = plot_results(all_retriving_time, x_numbers=code_dimensions,plot_width=800, plot_height=400, name=data_set_name,
                                          keys=keys, legends=legends,
                                          legend_visible=True, curve_ylabel='mean Average Time(s)')
print(results_stat)
result_to_print = {}
if 1:
    hash_lengths = sorted(results_stat.keys())
    for hl in hash_lengths:
        result_to_print[hl] = {'Hash length': hl}
        for k in keys:
            result_to_print[hl][know_name[k]] = '{:2.2%}({:2.2%})'.format(results_stat[hl][k]['mean'], results_stat[hl][k]['stds'])

    fieldnames = ['Hash length']
    fieldnames.extend(legends)

    import csv
    csv_file = os.path.join(result_path, 'all_MAPs' + str(max_index) + '.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for hl in hash_lengths:
            writer.writerow(result_to_print[hl])

    print(csv_file)

result_to_print_2 = {}

hash_lengths = sorted(results_stat.keys())
for hl in hash_lengths:
    result_to_print_2[hl] = {'Hash length': hl}
    for k in keys:
        result_to_print_2[hl][know_name[k]] = '{:2.2%}'.format(results_stat[hl][k]['mean'])

fieldnames = ['Hash length']
fieldnames.extend(legends)

import csv
csv_file_2 = os.path.join(result_path, 'all_MAPs_2_' + str(max_index) + '.csv')
with open(csv_file_2, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for hl in hash_lengths:
        writer.writerow(result_to_print_2[hl])

print(csv_file_2)

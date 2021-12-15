import pickle
import os
import matplotlib.pylab as plt
import numpy as np
from colormaps import cmaps
plt.style.use('ggplot')

cmap = cmaps['viridis']


dataset = ['MNIST', 'CIFAR10', 'GLOVE', 'SIFT10M']
models = ['Fly', 'FlylshDevelope', 'FlylshDevelopThreshold', 'FlylshDevelopThresholdRandomChoice']
mlabels = ['FlyLSH', 'Method 1', 'Method 2', 'Method 3']


hash_lengths = [2, 8, 16, 32]
sampling_ratios = [0.1]
embedding_sizes = [20]
yax = 10*hash_lengths

# embedding_sizes = [300, 600, 1500, 3000, 6000, 12000]
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
print(THIS_FOLDER.replace('plots', ''))
all_data = {}

for set in dataset:
    all_data[set] = {}
    for hash_length in hash_lengths:
        all_data[set][hash_length] = {}
        for embedding_size in embedding_sizes:
            embedding_size = embedding_size * hash_length
            all_data[set][hash_length][embedding_size] = {}
            for sampling_ratio in sampling_ratios:
                my_file = os.path.join('./results6', set, str(hash_length), str(embedding_size), str(sampling_ratio), 'all_MAPs10000.pkl')
                if os.path.exists(my_file):
                    infile = open(my_file, 'rb')
                    all_data[set][hash_length][embedding_size][sampling_ratio] = pickle.load(infile)
                    infile.close()

print(all_data)

colors = {'Fly': "#F71735",
          'FlylshDevelope': "#005066",
          'FlylshDevelopThreshold': "#4062BB",
          'FlylshDevelopThresholdRandomChoice': "#1098F7",
          'DenseFly': "#42B63E",
          'receptive-fields': "#F85E00",
          'BioHash': "#ED6A5A"
          }

cols = ['{}'.format(col) for col in dataset]

fig, axes = plt.subplots(nrows=len(sampling_ratios), ncols=len(dataset), figsize=(12, 3))

pad = 2 # in points

if len(sampling_ratios) > 1:
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', rotation=90, va='center')
else:
    for ax, col in zip(axes[:], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')


fig.tight_layout()
# tight_layout doesn't take these labels into account. We'll need
# to make some room. These numbers are are manually tweaked.
# You could automatically calculate them, but it's a pain.
fig.subplots_adjust(left=0.15, top=0.95)
for d in range(len(dataset)):
    for i in range(len(embedding_sizes)):
        for j in range(len(sampling_ratios)):
            for model_index in range(len(models)):
                y = []
                for hash_length in hash_lengths:
                    if hash_length in all_data[dataset[d]]:
                        if embedding_sizes[i]*hash_length in all_data[dataset[d]][hash_length]:
                            if sampling_ratios[j] in all_data[dataset[d]][hash_length][embedding_sizes[i]*hash_length]:
                                if models[model_index] in all_data[dataset[d]][hash_length][embedding_sizes[i]*hash_length][sampling_ratios[j]]:
                                    y.append(np.mean(all_data[dataset[d]][hash_length][embedding_sizes[i]*hash_length][sampling_ratios[j]][models[model_index]]))
                                else:
                                    y.append(np.nan)
                            else:
                                y.append(np.nan)
                        else:
                            y.append(np.nan)
                    else:
                        y.append(np.nan)
                yax = 10*hash_lengths
                if len(sampling_ratios) > 1:
                    axes[j, d].plot(hash_lengths, y, '|-', linewidth=1.5,
                                    color=colors[models[model_index]],
                                    label=mlabels[model_index])
                else:
                    axes[d].plot(hash_lengths, y, '|-', linewidth=1.5,
                                    color=colors[models[model_index]],
                                    label=mlabels[model_index])
                # axes[j, i].plot(hash_lengths, y, 'bo')

            # axes[j, i].legend()
            # axes[j, i].grid(True)
            if len(sampling_ratios) > 1:
                axes[j, d].set_xticks(hash_lengths)
                axes[j, d].set_ylim(bottom=0, top=0.6)
                axes[j, d].set_xlabel('Hash Length')
                axes[j, d].set_ylabel('Mean Average Precision (mAP)')
            else:
                axes[d].set_xticks(hash_lengths)
                axes[d].set_ylim(bottom=0, top=0.6)
                axes[d].set_xlabel('Number of Active KCs')

            # axes[0,0].set_title('embedding size = {}'.format(embedding_sizes[0]))
if len(sampling_ratios) > 1:
    handles, labels = axes[0, 0].get_legend_handles_labels()
else:
    axes[0].set_ylabel('Mean Average Precision (mAP)')
    handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc='best')
# fig.suptitle(dataset)
# plt.title(dataset)
# fig.setp(ax, ylim=(0, 0.8))
plt.show()

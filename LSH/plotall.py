import pickle
import os
import matplotlib.pylab as plt
import numpy as np
from colormaps import cmaps
# plt.style.use('ggplot')

cmap = cmaps['viridis']


dataset = 'mnist'
models = ['Fly', 'FlylshDevelope', 'FlylshDevelopThreshold', 'FlylshDevelopThresholdRandomChoice']
mlabels = ['FlyLSH', 'Method 1', 'Method 2', 'Method 3']

hash_lengths = [2, 8, 16, 32]
sampling_ratios = [0.01, 0.1]
embedding_sizes = [10, 20, 40, 80]
yax = 10*hash_lengths

# embedding_sizes = [300, 600, 1500, 3000, 6000, 12000]
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
print(THIS_FOLDER.replace('plots', ''))
all_data = {}
for hash_length in hash_lengths:
    all_data[hash_length] = {}
    for embedding_size in embedding_sizes:
        embedding_size = embedding_size * hash_length
        all_data[hash_length][embedding_size] = {}
        for sampling_ratio in sampling_ratios:
            my_file = os.path.join('./results6', dataset, str(hash_length), str(embedding_size), str(sampling_ratio), 'all_MAPs10000.pkl')
            if os.path.exists(my_file):
                infile = open(my_file, 'rb')
                all_data[hash_length][embedding_size][sampling_ratio] = pickle.load(infile)
                infile.close()

print(all_data)


# colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
colors = {'Fly': "#1f78b4",
          'FlylshDevelope': "#33a02c",
          'FlylshDevelopThreshold': "#ff7f00",
          'FlylshDevelopThresholdRandomChoice': "#6a3d9a",
          'DenseFly': "#e31a1c",
          'receptive-fields': "#b15928",
          # 'BioHash':
          }

cols = ['m = {}k'.format(col) for col in embedding_sizes]
rows = ['\u03B1 = {}'.format(row) for row in sampling_ratios]

fig, axes = plt.subplots(nrows=len(sampling_ratios), ncols=len(embedding_sizes), figsize=(12, 8))

pad = 5 # in points

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
    for row in rows:
        axes[0].annotate(row, xy=(0, 0.5), xytext=(-axes[0].yaxis.labelpad - pad, 0),
                    xycoords=axes[0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', rotation=90, va='center')


fig.tight_layout()
# tight_layout doesn't take these labels into account. We'll need
# to make some room. These numbers are are manually tweaked.
# You could automatically calculate them, but it's a pain.
fig.subplots_adjust(left=0.15, top=0.95)
for i in range(len(embedding_sizes)):
    for j in range(len(sampling_ratios)):
        for model_index in range(len(models)):
            y = []
            for hash_length in hash_lengths:
                if hash_length in all_data:
                    if embedding_sizes[i]*hash_length in all_data[hash_length]:
                        if sampling_ratios[j] in all_data[hash_length][embedding_sizes[i]*hash_length]:
                            if models[model_index] in all_data[hash_length][embedding_sizes[i]*hash_length][sampling_ratios[j]]:
                                y.append(np.mean(all_data[hash_length][embedding_sizes[i]*hash_length][sampling_ratios[j]][models[model_index]]))
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
                axes[j, i].plot(hash_lengths, y, '|-',
                                color=colors[models[model_index]],
                                label=mlabels[model_index])
            else:
                axes[i].plot(hash_lengths, y, '|-', linewidth=1,
                                color=colors[models[model_index]],
                                label=mlabels[model_index])
            # axes[j, i].plot(hash_lengths, y, 'bo')
        # axes[j, i].set_xlabel('Hash Length')
        # axes[j, i].set_ylabel('Mean Average Precision (mAP)')
        # axes[j, i].legend()
        # axes[j, i].grid(True)
        if len(sampling_ratios) > 1:
            axes[j, i].set_xticks(hash_lengths)
            axes[j, i].set_ylim(bottom=0, top=0.6)
        else:
            axes[i].set_xticks(hash_lengths)
            axes[i].set_ylim(bottom=0, top=0.6)
        # axes[0,0].set_title('embedding size = {}'.format(embedding_sizes[0]))
if len(sampling_ratios) > 1:
    handles, labels = axes[0, 0].get_legend_handles_labels()
else:
    handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')
# fig.suptitle(dataset)
# plt.title(dataset)
# fig.setp(ax, ylim=(0, 0.8))
plt.show()

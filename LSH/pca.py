from sklearn.decomposition import PCA
import Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap


def compute_pcs(data):
    pca = PCA().fit(data)
    num_pc = pca.n_features_
    pc_list = np.arange(1, num_pc + 1)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return cum

############ mnist
mnist = Dataset.Dataset('mnist', 0, 10000)
mnist = mnist.data['data']
mnist = mnist / 255.0

############ cifar
cifar = Dataset.Dataset('cifar10', 0, 10000)
cifar = cifar.data['data'][:5000]

############ glove
glove = Dataset.Dataset('glove', 0, 10000)
glove = glove.data['data']

############ sift
sift = Dataset.Dataset('sift10m', 0, 10000)
sift = sift.data['data']

############ random
random = Dataset.Dataset('random', 0, 10000)
random = random.data['data']

num_pcs = 128
pca_mnist = PCA().fit(mnist)

# get eigenvalues (variance explained by each PC)
cum_mnist = compute_pcs(mnist)
cum_cifar = compute_pcs(cifar)
cum_glove = compute_pcs(glove)
cum_sift = compute_pcs(sift)
cum_random = compute_pcs(random)


pc_list = np.arange(1, 3072 + 1)
cmnist = np.ones([3072])
cglove = np.ones([3072])
csift = np.ones([3072])
crandom = np.ones([3072])

cmnist[:784] = cum_mnist
cglove[:300] = cum_glove
csift[:128] = cum_sift
crandom[:128] = cum_random

till = 128
plt.rcParams.update({'font.size': 16})
ax = plt.axes(facecolor='#FFFFFF')
ax.set_axisbelow(True)
plt.grid(color='#E6E6E6', linestyle='solid')

for spine in ax.spines.values():
    spine.set_visible(True)

plt.plot(pc_list[:till], cmnist[:till],
         pc_list[:till], cum_cifar[:till],
         pc_list[:till], cglove[:till],
         pc_list[:till], csift[:till],
         pc_list[:till], crandom[:till])

labels = ['MNIST',
          'CIFAR10',
          'GLOVE',
          'SIFT10M',
          'RANDOM']
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Sum of Explained Variance Ratios')
plt.legend(labels)
plt.show()


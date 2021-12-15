
import random
import numpy as np
from scipy.io import loadmat
import pickle, time
import os
from functools import reduce
from bokeh.plotting import figure, output_file, show
from tqdm import tqdm
NUMBER_OF_PROCESSES = 5

class Dataset(object):
    def __init__(self, name, path='./datasets/'):
        self.path = path
        self.name = name.upper()
        if self.name == 'MNIST' or self.name == 'FMNIST':
            self.indim = 784
            try:
                from mnist import MNIST
                mndata = MNIST(os.path.join(path, self.name))
                mndata.gz = True
                images, labels = mndata.load_training()
                self.data = {'images': images, 'labels': labels}
                # self.data = tfds.load('mnist', split='train', shuffle_files=True)
                # self.data = read_data_sets(self.path + self.name)
                # self.data = self.data.train.images
            except OSError as err:
                print(str(err))
                raise ValueError('Try again')
            # ds = tfds.load('mnist', split='train', shuffle_files=True)
        elif self.name == 'CIFAR10':
            self.indim = (32, 32, 3)
            if self.name not in os.listdir(self.path):
                print('Data not in path')
                raise ValueError()
            def unpickle(file):
                import pickle
                with open(file, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                return dict
            # self.data = unpickle(os.path.join(self.path, self.name, 'data_batch_1'))
            self.meta = unpickle(os.path.join(self.path, self.name, "batches.meta"))

        elif self.name == 'GLOVE':
            self.indim = 300
            # self.data = pickle.load(open(self.path + 'glove30k.p', 'rb'))

            def load_glove_model(File, max_index=50000):
                print("Loading Glove Model")
                glove_model = {}
                with open(File, 'r', encoding='utf-8') as f:
                    if max_index != 0:
                        i = 0
                    for line in f:
                        split_line = line.split()
                        word = split_line[0]
                        embedding = np.array(split_line[1:], dtype=np.float64)
                        glove_model[word] = embedding
                        if max_index != 0:
                            i += 1
                            if i >= max_index:
                                break
                print(f"{len(glove_model)} words loaded!")
                return glove_model
            self.data = load_glove_model(os.path.join(self.path, self.name, "glove.6B.300d.txt"))

        elif self.name == 'SIFT':
            self.indim = 128
            # self.data = loadmat(self.path + self.name + '/siftvecs.mat')['vecs']
            def ivecs_read(fname):
                a = np.fromfile(fname, dtype='int32')
                d = a[0]
                return a.reshape(-1, d + 1)[:, 1:].copy()

            def fvecs_read(fname):
                return ivecs_read(fname).view('float32')
            self.data = fvecs_read(os.path.join(self.path, self.name, "siftsmall_learn.fvecs"))
        elif self.name == 'GIST':
            self.indim = 960
            self.data = loadmat(self.path + self.name + '/gistvecs.mat')['vecs']

        elif self.name == 'LMGIST':
            self.indim = 512
            self.data = loadmat(self.path + self.name + '/LabelMe_gist.mat')['gist']

        elif self.name == 'RANDOM':
            self.indim = 128
            self.data = np.random.random(size=(100_000, self.indim))  # np.random.randn(100_000,self.indim)

    def train_batches(self, batch_size=64, sub_mean=False, maxsize=-1):
        if self.name in ['MNIST', 'FMNIST']:
            max_ = self.data.train.images.shape[0] - batch_size if maxsize == -1 else maxsize - batch_size
            for idx in range(0, max_, batch_size):
                batch_x = self.data.train.images[idx:idx + batch_size, :]
                batch_y = self.data.train.labels[idx:idx + batch_size]
                batch_y = np.eye(10)[batch_y]
                if sub_mean:
                    batch_x = batch_x - batch_x.mean(axis=1)[:, None]

                yield batch_x, batch_y

        elif self.name == 'CIFAR10':
            for batch_num in [1, 2, 3, 4, 5]:
                filename = self.path + self.name + '/data_batch_' + str(batch_num)
                with open(filename, mode='rb') as f:
                    data_dict = pickle.load(f, encoding='bytes')
                    features, labels = data_dict[b'data'], data_dict[b'labels']
                for begin in range(0, len(features), batch_size):
                    end = min(begin + batch_size, len(features))
                    yield features[begin:end], labels[begin:end]

        elif self.name in ['GLOVE', 'SIFT', 'LMGIST', 'RANDOM']:
            max_ = self.data.shape[0] - batch_size if maxsize == -1 else maxsize - batch_size
            for idx in range(0, max_, batch_size):
                batch_x = self.data[idx:idx + batch_size, :]
                if sub_mean:
                    batch_x = batch_x - batch_x.mean(axis=1)[:, None]
                yield batch_x, None

    def test_set(self, maxsize=-1, sub_mean=False):
        # maxsize determines how many elements of test set to return
        if self.name in ['MNIST', 'FMNIST']:
            test_x = self.data.test.images[:maxsize]
            test_y = np.eye(10)[self.data.test.labels[:maxsize]]
            if sub_mean:
                test_x = test_x - test_x.mean(axis=1)[:, None]
            return (test_x, test_y)

        elif self.name == 'CIFAR10':
            with open(self.path + self.name + '/test_batch', mode='rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                features, labels = data_dict[b'data'], data_dict[b'labels']
            test_x, test_y = features[:maxsize], labels[:maxsize]
            if sub_mean:
                test_x = test_x - test_x.mean(axis=1)[:, None]
            return test_x, test_y

        elif self.name in ['GLOVE', 'SIFT', 'LMGIST', 'RANDOM']:
            test_x = self.data[:maxsize]
            # test_y=np.eye(10)[self.data.test.labels[:maxsize]]
            if sub_mean:
                test_x = test_x - test_x.mean(axis=1)[:, None]
            return (test_x, None)


class LSH(object):
    def __init__(self, data, hash_length):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        (ratio of PNs that each KC samples from)
        embedding_size: dimensionality of projection space, m
        """
        self.hash_length = hash_length
        self.data = data - np.mean(data, axis=1)[:, None]
        self.weights = np.random.random((data.shape[1], hash_length))
        self.hashes = (self.data @ self.weights) > 0
        self.maxl1distance = 2 * self.hash_length

    def query(self, qidx, nnn, not_olap=False):
        L1_distances = np.sum(np.abs(self.hashes[qidx, :] ^ self.hashes), axis=1)
        # np.sum(np.bitwise_xor(self.hashes[qidx,:],self.hashes),axis=1)
        nnn = min(self.hashes.shape[0], nnn)
        if not_olap:
            no_overlaps = np.sum(L1_distances == self.maxl1distance)
            return no_overlaps

        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)][:nnn]
        # print(L1_distances[NNs]) #an interesting property of this hash is that the L1 distances are always even
        return NNs

    def true_nns(self, qidx, nnn):
        sample = self.data[qidx, :]
        tnns = np.sum((self.data - sample) ** 2, axis=1).argsort()[:nnn + 1]
        tnns = tnns[(tnns != qidx)]
        if nnn < self.data.shape[0]:
            assert len(tnns) == nnn, 'nnn={}'.format(nnn)
        return tnns

    def construct_true_nns(self, indices, nnn):
        all_NNs = np.zeros((len(indices), nnn))
        for idx1, idx2 in enumerate(indices):
            all_NNs[idx1, :] = self.true_nns(idx2, nnn)
        return all_NNs

    def AP(self, predictions, truth):
        assert len(predictions) == len(truth) or len(predictions) == self.hashes.shape[0]
        # removed conversion to list in next line:
        precisions = [len((set(predictions[:idx]).intersection(set(truth[:idx])))) / idx for \
                      idx in range(1, len(truth) + 1)]
        return np.mean(precisions)

    def PR(self, qidx, truth, atindices):
        """truth should be a set"""
        L1_distances = np.sum((self.hashes[qidx, :] ^ self.hashes), axis=1)
        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)]
        # predictions=NNs
        recalls = np.arange(1, len(truth) + 1)
        all_recalls = [len(set(NNs[:idx]) & truth) for idx in atindices]
        # all_recalls.append(len(set(NNs)&truth))
        # all_recalls=[len(set(predictions[:idx])&truth) for idx in range(1,self.hashes.shape[0]+1)]
        # indices=[all_recalls.index(recall) for recall in recalls]
        precisions = [recall / (idx + 1) for idx, recall in zip(atindices, all_recalls)]
        # this_pr=odict({l:(p,r) for l,p,r in zip(atL1,precisions,recalls)})
        return precisions, all_recalls  # (precisions,all_recalls)

    def ROC(self, qidx, truth, atindices):
        """x: False positive rate, y: True positive rate, truth should be a set"""
        L1_distances = np.sum((self.hashes[qidx, :] ^ self.hashes), axis=1)
        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)]
        x, y = [], []
        for idx in atindices:
            ntruepos = len((set(NNs[:idx]) & truth))  # number of positives correctly classified
            nfalseneg = idx - ntruepos  # number of negatives incorrectly classified
            tpr = ntruepos / len(truth)  # positives correctly classified / total positives
            fpr = nfalseneg / (len(NNs) - len(truth))  # negatives incorrectly classified / total negatives
            x.append(fpr)
            y.append(tpr)
        return x, y

    def findmAP_given_true(self, nnn, n_points, all_NNs):
        sample_indices = np.random.choice(self.data.shape[0], n_points)
        self.allAPs = []
        for eidx, didx in enumerate(sample_indices):
            # eidx: enumeration id, didx: index of sample in self.data
            this_nns = self.query(didx, nnn)
            # print(len(this_nns))
            this_AP = self.AP(list(this_nns), list(all_NNs[didx, :]))
            # print(this_AP)
            self.allAPs.append(this_AP)
        return np.mean(self.allAPs)

    def findmAP(self, nnn, n_points):
        start = np.random.randint(low=0, high=self.data.shape[0] - n_points)
        sample_indices = np.random.choice(self.data.shape[0], n_points)
        all_NNs = self.construct_true_nns(sample_indices, nnn)
        self.allAPs = []
        for eidx, didx in enumerate(sample_indices):
            # eidx: enumeration id, didx: index of sample in self.data
            this_nns = self.query(didx, nnn)
            # print(len(this_nns))
            this_AP = self.AP(list(this_nns), list(all_NNs[eidx, :]))
            # print(this_AP)
            self.allAPs.append(this_AP)
        return np.mean(self.allAPs)

    def findZKk(self, n_points):
        """
        ZKk is the number of vectors whose overlap with a specific vector is zero
        """
        sample_indices = np.random.choice(self.data.shape[0], n_points)
        no_overlaps = []
        for eidx, didx in enumerate(sample_indices):
            no_overlaps.append(self.query(didx, -20, not_olap=True))
        return np.mean(no_overlaps)

    def computePRC(self, n_points=1, nnn=200, atindices=None):
        """
        This function calculates precision-recall metrics for model
        """

        def replacenans(x):
            nanidxs = [idx for idx in range(len(x)) if np.isnan(x[idx])]
            notnang = lambda idx: [nidx for nidx in range(idx + 1, len(x)) if nidx not in nanidxs][0]
            notnans = lambda idx: [nidx for nidx in range(idx) if nidx not in nanidxs][-1]
            if len(nanidxs) == 0:
                return x
            else:
                for nanidx in nanidxs:
                    if nanidx == 0:
                        x[nanidx] = x[notnang(nanidx)]
                    else:
                        x[nanidx] = (x[notnang(nanidx)] + x[notnans(nanidx)]) / 2
                return x

        sample_indices = np.random.choice(self.data.shape[0], n_points)
        all_NNs = self.construct_true_nns(sample_indices, nnn)
        self.allprecisions = np.zeros((n_points, len(atindices)))
        self.allrecalls = np.zeros((n_points, len(atindices)))
        # allprs=odict({l:[[],[]] for l in atL1})
        for eidx, didx in enumerate(sample_indices):
            """eidx: enumeration id, didx: index of sample in self.data"""
            # this_nns=self.query(didx,self.hashes.shape[0]) #this is intentionally kept a large number
            this_p, this_r = self.PR(didx, set(all_NNs[eidx, :]), atindices)
            # [allprcs[r].append(p) for p,r in zip(this_p,this_r)]

            self.allprecisions[eidx, :] = this_p
            self.allrecalls[eidx, :] = this_r

        return [self.allprecisions.mean(axis=0),
                self.allrecalls.mean(axis=0)]  # replacenans([np.nanmean(v) for _,v in allprcs.items()])

    def computeROC(self, n_points=1, nnn=200, atindices=None):
        """
        This function calculates receiver operator characteristics (ROC)
        """
        sample_indices = np.random.choice(self.hashes.shape[0], n_points)
        all_NNs = self.construct_true_nns(sample_indices, nnn)
        alltprs = np.zeros((n_points, len(atindices)))
        allfprs = np.zeros((n_points, len(atindices)))
        for eidx, didx in enumerate(sample_indices):
            this_fpr, this_tpr = self.ROC(didx, set(all_NNs[eidx, :]), atindices)
            allfprs[eidx, :] = this_fpr
            alltprs[eidx, :] = this_tpr
        return [allfprs.mean(axis=0), alltprs.mean(axis=0)]

    def create_bins(self):
        if hasattr(self, 'bins'):
            return
        start = time.time()
        self.bins = np.unique(self.hashes, axis=0)
        self.num_bins = self.bins.shape[0]

        assignment = np.zeros(self.hashes.shape[0])
        for idx, _bin in enumerate(self.bins):
            assignment[(self.hashes == _bin).all(axis=1)] = idx
        self.binstopoints = {bin_idx: np.flatnonzero(assignment == bin_idx) for bin_idx in range(self.bins.shape[0])}
        self.pointstobins = {point: int(_bin) for point, _bin in enumerate(assignment)}
        self.timetoindex = time.time() - start

    def query_bins(self, qidx, search_radius=1, order=True):
        if not hasattr(self, 'bins'):
            raise ValueError('Bins for model not created')
        query_bin = self.bins[self.pointstobins[qidx]]
        valid_bins = np.flatnonzero((query_bin[None, :] ^ self.bins).sum(axis=1) <= search_radius)
        all_points = reduce(np.union1d, np.array([self.binstopoints[idx] for idx in valid_bins]))
        if order:
            l1distances = (self.hashes[qidx, :] ^ self.hashes[all_points, :]).sum(axis=1)
            all_points = all_points[l1distances.argsort()]
        return all_points

    def compute_query_mAP(self, n_points, search_radius=1, order=True, nnn=None):

        sample_indices = np.random.choice(self.hashes.shape[0], n_points)
        average_precisions = []
        elapsed = []
        numpredicted = []
        ms = lambda l: (np.mean(l), np.std(l))
        for qidx in sample_indices:
            start = time.time()
            predicted = self.query_bins(qidx, search_radius)

            if nnn is None:
                elapsed.append(time.time() - start)
            else:
                if len(predicted) < nnn:
                    # raise ValueError('Not a good search radius')
                    continue
                elapsed.append(time.time() - start)
                numpredicted.append(len(predicted))

            truenns = self.true_nns(qidx, nnn=len(predicted))
            average_precisions.append(self.AP(predictions=predicted, truth=truenns))
        if nnn is not None:
            if len(average_precisions) < 0.8 * nnn:
                raise ValueError('Not a good search radius')

        return [*ms(average_precisions), *ms(elapsed), *ms(numpredicted)]








class flylsh(LSH):
    def __init__(self, data, hash_length, sampling_ratio, embedding_size, distribution_random=False, shift=True):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.hash_length = hash_length
        self.embedding_size = embedding_size
        K = embedding_size // hash_length
        data_mean = np.mean(data, axis=1)
        if shift:
            self.data = (data - data_mean[:, None])
        else:
            self.data = data

        num_projections = int(sampling_ratio * data.shape[1])
        weights = np.random.random((data.shape[1], embedding_size))
        if distribution_random:
            weights = weights * np.mean(data, axis=0)[:, None]
        yindices = np.arange(weights.shape[1])[None, :]
        xindices = weights.argsort(axis=0)[-num_projections:, :]
        self.weights = np.zeros_like(weights, dtype=np.bool)
        self.weights[xindices, yindices] = True  # sparse projection vectors

        all_activations = (self.data @ self.weights)
        xindices = np.arange(data.shape[0])[:, None]
        yindices = all_activations.argsort(axis=1)[:, -hash_length:]
        self.hashes = np.zeros_like(all_activations, dtype=np.bool)
        # threshold=np.sort(all_activations,axis=1)[:,-hash_length][:,None]
        self.hashes[xindices, yindices] = True  # choose topk activations
        # self.dense_activations=all_activations
        # self.sparse_activations=self.hashes.astype(np.float32)*all_activations #elementwise product
        self.maxl1distance = 2 * self.hash_length
        self.lowd_hashes = all_activations.reshape((-1, hash_length, K)).sum(axis=-1) > 0

    def create_highd_bins(self, d, rounds=1):
        """
        This function implements a relaxed binning for FlyLSH
        This is only one of the many possible implementations for such a scheme
        d: the number of bits to match between hashes for putting them in the same bin
        """
        self.highd_bins = self.hashes[0:1, :]  # initialize hashes to first point
        self.highd_binstopoints, self.highd_pointstobins = {}, {i: [] for i in range(self.hashes.shape[0])}
        for round in range(rounds):
            for hash_idx, this_hash in enumerate(self.hashes):
                overlap = (self.maxl1distance - ((this_hash[None, :] ^ self.highd_bins).sum(axis=1))) >= 2 * d
                # print(overlap.shape)
                if overlap.any():
                    indices = np.flatnonzero(overlap)
                    # indices=indices.tolist()
                    # print(indices)
                    self.highd_pointstobins[hash_idx].extend(indices)
                    for idx in indices:
                        if idx not in self.highd_binstopoints:
                            # print(indices,idx)
                            self.highd_binstopoints[idx] = []
                        self.highd_binstopoints[idx].append(hash_idx)
                else:
                    self.highd_bins = np.append(self.highd_bins, this_hash[None, :], axis=0)
                    bin_idx = self.highd_bins.shape[0] - 1
                    self.highd_pointstobins[hash_idx].append(bin_idx)
                    self.highd_binstopoints[bin_idx] = [hash_idx]

    def create_lowd_bins(self):
        start = time.time()
        self.lowd_bins = np.unique(self.lowd_hashes, axis=0)
        # self.num_bins=self.bins.shape[0]

        assignment = np.zeros(self.lowd_hashes.shape[0])
        for idx, _bin in enumerate(self.lowd_bins):
            assignment[(self.lowd_hashes == _bin).all(axis=1)] = idx
        self.lowd_binstopoints = {bin_idx: np.flatnonzero(assignment == bin_idx) for bin_idx in
                                  range(self.lowd_bins.shape[0])}
        self.lowd_pointstobins = {point: int(_bin) for point, _bin in enumerate(assignment)}
        self.timetoindex = time.time() - start

    def query_lowd_bins(self, qidx, search_radius=1, order=False):
        if not hasattr(self, 'lowd_bins'):
            raise ValueError('low dimensional bins for model not created')
        query_bin = self.lowd_bins[self.lowd_pointstobins[qidx]]
        valid_bins = np.flatnonzero((query_bin[None, :] ^ self.lowd_bins).sum(axis=1) <= 2 * search_radius)
        all_points = reduce(np.union1d, np.array([self.lowd_binstopoints[idx] for idx in valid_bins]))
        if order:
            l1distances = (self.hashes[qidx, :] ^ self.hashes[all_points, :]).sum(axis=1)
            all_points = all_points[l1distances.argsort()]
        return all_points

    def query_highd_bins(self, qidx, order=False):
        if not hasattr(self, 'highd_bins'):
            raise ValueError('high dimensional bins for model not created')
        valid_bins = self.highd_pointstobins[qidx]
        all_points = reduce(np.union1d, np.array([self.highd_binstopoints[idx] for idx in valid_bins]))
        if order:
            l1distances = (self.hashes[qidx, :] ^ self.hashes[all_points, :]).sum(axis=1)
            all_points = all_points[l1distances.argsort()]
        return all_points

    def compute_query_mAP(self, n_points, search_radius=1, order=False, qtype='lowd', nnn=None):
        sample_indices = np.random.choice(self.hashes.shape[0], n_points)
        average_precisions = []
        elapsed = []
        numpredicted = []
        ms = lambda l: (np.mean(l), np.std(l))
        for qidx in sample_indices:
            start = time.time()
            if qtype == 'lowd':
                predicted = self.query_lowd_bins(qidx, search_radius, order)
            elif qtype == 'highd':
                predicted = self.query_highd_bins(qidx, order)
            assert len(predicted) < self.hashes.shape[0], 'All point being queried'

            if nnn is None:
                elapsed.append(time.time() - start)
            else:
                if len(predicted) < nnn:
                    # raise ValueError('Not a good search radius')
                    continue
                elapsed.append(time.time() - start)
                numpredicted.append(len(predicted))

                predicted = predicted[:nnn]

            truenns = self.true_nns(qidx, nnn=len(predicted))
            average_precisions.append(self.AP(predictions=predicted, truth=truenns))
        if nnn is not None:
            if len(average_precisions) < 0.8 * n_points:
                raise ValueError('Not a good search radius')

        return [*ms(average_precisions), *ms(elapsed), *ms(numpredicted)]

    def compute_recall(self, n_points, nnn, sr):
        sample_indices = np.random.choice(self.data.shape[0], n_points)
        recalls = []
        elapsed = []
        numpredicted = []
        for qidx in sample_indices:
            start = time.time()
            predicted = self.query_lowd_bins(qidx, sr)
            if len(predicted) < nnn:
                raise ValueError('Not a good search radius')  # continue

            numpredicted.append(len(predicted))
            rankings = np.sum((self.hashes[predicted, :] ^ self.hashes[qidx, :]), axis=1).argsort()
            predicted = predicted[rankings][:nnn]
            elapsed.append(time.time() - start)
            trueNNs = self.true_nns(qidx, nnn)
            recalls.append(len(set(predicted) & set(trueNNs)) / nnn)
        return [np.mean(recalls), np.std(recalls), np.mean(elapsed), np.std(elapsed), np.mean(numpredicted),
                np.std(numpredicted)]

    def rank_and_findmAP(self, n_points, nnn):
        ms = lambda l: (np.mean(l), np.std(l))
        average_precisions = []
        elapsed = []
        for idx in range(n_points):
            start = time.time()
            average_precisions.append(self.findmAP(nnn, 1))
            elapsed.append(time.time() - start)
        return [*ms(average_precisions), *ms(elapsed)]


class denseflylsh(flylsh):
    def __init__(self, data, hash_length, sampling_ratio, embedding_size, shift=True):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.hash_length = hash_length
        self.embedding_size = embedding_size
        K = embedding_size // hash_length
        data_mean = np.mean(data, axis=1)
        if shift:
            self.data = (data - data_mean[:, None])
        else:
            self.data = data
        self.data = (data - np.mean(data, axis=1)[:, None])
        weights = np.random.random((data.shape[1], embedding_size))
        self.weights = (weights > 1 - sampling_ratio)  # sparse projection vectors
        all_activations = (self.data @ self.weights)
        threshold = 0
        self.hashes = (all_activations >= threshold)  # choose topk activations
        # self.dense_activations=all_activations
        # self.sparse_activations=self.hashes.astype(np.float32)*all_activations #elementwise product
        self.maxl1distance = 2 * self.hash_length
        self.lowd_hashes = all_activations.reshape((-1, hash_length, K)).sum(axis=-1) > 0


colors = {'LSH': 'darkseagreen',  'Fly': 'cornflowerblue',  \
          'DenseFly': 'blue', \
           'FlylshDevelope': 'purple', 'FlylshDevelopThreshold': 'darkorange',
          'FlylshDevelopeRandomChoice':'cyan',
          'FlylshDevelopThresholdRandomChoice':'silver'}


def plot_results(all_results, hash_lengths=None, keys=None, name='data', location='./', metric='mAP', plot_width=800,
                 plot_height=400, legends=None, legend_visible=True, curve_ylabel=None, x_numbers=None):
    if hash_lengths is None:
        hash_lengths = sorted(all_results.keys())
    if x_numbers is None:
        x_numbers = hash_lengths
    if keys is None:
        keys = list(all_results[hash_lengths[0]].keys())

    if legends is None:
        legends = keys
    Lk = len(keys)
    fmt = lambda mk: mk.join([k for k in legends])

    global colors

    if metric == 'mAP':
        if curve_ylabel is None:
            curve_ylabel = 'mean Average Precision (mAP)'
        min_y = 0
        mean = lambda x, n: np.mean(all_results[x][n])
        stdev = lambda x, n: np.std(all_results[x][n])
    elif metric == 'auprc':
        if curve_ylabel is None:
            curve_ylabel = 'Area under precision recall curve'
        min_y = 0
        n_trials = len(all_results[hash_lengths[0]][keys[0]])
        all_precisions = {hl: {k: [all_results[hl][k][i][0] for i in range(n_trials)] for k in keys} for hl in
                          hash_lengths}
        all_recalls = {
            hl: {k: [all_results[hl][k][i][1] / np.max(all_results[hl][k][i][1]) for i in range(n_trials)] for k in
                 keys} for hl in hash_lengths}
        auprc = lambda hl, k, i: np.sum(np.gradient(all_recalls[hl][k][i]) * all_precisions[hl][k][i])
        mean = lambda hl, k: np.mean([auprc(hl, k, i) for i in range(n_trials)])
        stdev = lambda hl, k: np.std(
            [auprc(hl, k, i) for i in range(n_trials)])  # np.std(np.array(all_MAPs[x][n]),axis=0)
    elif metric == 'auroc':
        if curve_ylabel is None:
            curve_ylabel = 'Area under Receiver Operating Characteristic (ROC) curve'
        min_y = 0.5
        n_trials = len(all_results[hash_lengths[0]][keys[0]])
        all_tprs = {hl: {k: [all_results[hl][k][i][1] for i in range(n_trials)] for k in keys} for hl in hash_lengths}
        all_fprs = {
            hl: {k: [all_results[hl][k][i][0] / np.max(all_results[hl][k][i][0]) for i in range(n_trials)] for k in
                 keys} for hl in hash_lengths}

        auroc = lambda hl, k, i: np.sum(np.gradient(all_fprs[hl][k][i]) * all_tprs[hl][k][i])
        mean = lambda hl, k: np.mean([auroc(hl, k, i) for i in range(n_trials)])
        stdev = lambda hl, k: np.std(
            [auroc(hl, k, i) for i in range(n_trials)])  # np.std(np.array(all_MAPs[x][n]),axis=0)

    # p = figure(x_range=[str(h) for h in hash_lengths], title=f'{fmt(",")} on {name}')
    p = figure(x_range=[str(h) for h in x_numbers])
    # p = figure(title=f'{fmt(",")} on {name}')
    delta = 0.5 / (Lk + 1)
    deltas = [delta * i for i in range(-Lk, Lk)][1::2]
    assert len(deltas) == Lk, 'Bad luck'

    x_shift = len(hash_lengths)/2
    x_axes = np.sort(np.array([[0.5+x + d for d in deltas] for x in range(0,  len(hash_lengths))]), axis=None)
    means = [mean(hashl, name) for name, hashl in zip(keys * len(hash_lengths), sorted(hash_lengths * Lk))]
    stds = [stdev(hashl, name) for name, hashl in zip(keys * len(hash_lengths), sorted(hash_lengths * Lk))]

    for i in range(len(hash_lengths)):
        for j in range(Lk):
            p.vbar(x=x_axes[Lk * i + j], width=delta, bottom=0, top=means[Lk * i + j], color=colors[keys[j]],
                   legend=legends[j])

    results_stat = {}
    for i in range(len(hash_lengths)):
        results_stat[hash_lengths[i]] = {}
        for j in range(Lk):
            results_stat[hash_lengths[i]][keys[j]] = {'mean': means[Lk * i + j], 'stds': stds[Lk * i + j]}

    err_xs = [[i, i] for i in x_axes]
    err_ys = [[m - s, m + s] for m, s in zip(means, stds)]
    p.y_range.bounds = (min_y, np.floor(10 * max(means)) / 10 + 0.1)
    p.plot_width = plot_width
    p.plot_height = plot_height
    p.multi_line(err_xs, err_ys, line_width=2, color='black', legend='stdev')
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    p.legend.background_fill_alpha = 0.0
    p.legend.label_text_font_size = '20px'
    p.legend.visible = legend_visible
    p.xaxis.axis_label = 'Code dimension (m)'# 'Hash length (k)/Code length (bits)'
    p.yaxis.axis_label = curve_ylabel
    p.xaxis.major_label_text_font_size = '24px'
    p.yaxis.major_label_text_font_size = '24px'
    p.xaxis.axis_label_text_font_size = '24px'
    p.yaxis.axis_label_text_font_size = '24px'
    output_file(f'{location + fmt("_")}_{name}.html')
    show(p)
    return p, results_stat


def plothlcurve(all_results, hl, name='data', location='./', metric='prc'):
    global colors

    assert hl in all_results.keys(), 'Provide a valid hash length'
    keys = list(all_results[hl].keys())
    n_trials = len(all_results[hl][keys[0]])

    if metric == 'prc':
        all_ys = {k: np.mean([all_results[hl][k][i][0] for i in range(n_trials)], axis=0) for k in keys}
        all_xs = {k: np.mean([all_results[hl][k][i][1] for i in range(n_trials)], axis=0) for k in keys}
        all_xs = {k: all_xs[k] / np.max(all_xs[k]) for k in keys}
        title = f'Precision recall curves for {name}, hash length={hl}'
        xlabel = 'Recall'
        ylabel = 'Precision'
        legend_location = 'top_right'
    elif metric == 'roc':
        all_xs = {k: np.mean([all_results[hl][k][i][0] for i in range(n_trials)], axis=0) for k in keys}
        all_ys = {k: np.mean([all_results[hl][k][i][1] for i in range(n_trials)], axis=0) for k in keys}
        all_xs = {k: all_xs[k] / np.max(all_xs[k]) for k in keys}

        title = f'ROC curves for {name}, hash length={hl}'
        xlabel = 'False Positive rate'
        ylabel = 'True Positive rate'
        legend_location = 'bottom_right'
    auc = lambda k: np.sum(np.gradient(all_xs[k]) * all_ys[k])
    aucs = {k: auc(k) for k in keys}

    p = figure(title=title)
    for k in keys:
        leg = '{}({:.2f})'.format(k, 0.01 * np.floor(100 * np.mean(aucs[k])))
        p.line(all_xs[k], all_ys[k], line_width=2, color=colors[k], legend=leg)

    if metric == 'roc':
        p.line(np.arange(100) / 100.0, np.arange(100) / 100.0, line_width=1, line_dash='dashed', legend='random (0.5)')
        # show random classifier line for ROC metrics

    p.legend.location = legend_location
    p.legend.click_policy = 'hide'
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

    output_file(f'{location}{metric}_{name}_{hl}.html')
    show(p)
    return p


def parse_computed(foldername):
    allfiles = os.listdir(foldername)
    mnames = ['LSH', 'Fly', 'WTA']
    fmlname = {'LSH': 'LSH', 'Fly': 'Fly', 'WTA': 'WTA'}
    # mnames=['lsh','fly','WTA']
    # fmlname={'lsh':'LSH','fly':'Fly','WTA':'WTA'}
    hash_lengths = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    allmaps = {hl: {} for hl in hash_lengths}
    for hl in hash_lengths:
        for mnm in mnames:
            allmaps[hl][fmlname[mnm]] = []
            possible = [f for f in allfiles if mnm + str(hl) + '_' in f]
            for fnm in possible:
                f = open(foldername + fnm, 'r')
                allmaps[hl][fmlname[mnm]].append(float(f.read()))
    return allmaps

class flylsh_quick_retrive(flylsh):
    def query(self, qidx, nnn, not_olap=False):
        ones_indexs = np.array(np.where(self.hashes[qidx, :])).ravel()
        # L1_distances = np.sum(np.abs(self.hashes[qidx, :] ^ self.hashes), axis=1)
        L1_distances = np.sum(np.abs(self.hashes[qidx, ones_indexs] ^ self.hashes[:, ones_indexs]), axis=1)
        # np.sum(np.bitwise_xor(self.hashes[qidx,:],self.hashes),axis=1)
        nnn = min(self.hashes.shape[0], nnn)
        if not_olap:
            no_overlaps = np.sum(L1_distances == self.maxl1distance)
            return no_overlaps

        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)][:nnn]
        # print(L1_distances[NNs]) #an interesting property of this hash is that the L1 distances are always even
        return NNs

class FlylshDevelope(flylsh_quick_retrive):
    def __init__(self, data, hash_length, sampling_ratio, embedding_size, connection_rule='max', shift=True):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m, number of KCs
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.hash_length = hash_length
        self.embedding_size = embedding_size
        K = embedding_size // hash_length
        data_mean = np.mean(data, axis=1)
        if shift:
            self.data = (data - data_mean[:, None])
        else:
            self.data = data

        # self.num_connections_per_dim_target = num_connections_per_dim_target

        num_projections = int(sampling_ratio * data.shape[1])  ## number of post synapse of KC
        yindices = np.arange(embedding_size)[None, :]
        xindices = np.zeros((num_projections, self.embedding_size), dtype=np.int_)
        print(data.shape[0], self.embedding_size)
        choosen_data = np.random.choice(data.shape[0], size=int(self.embedding_size * 1.5), replace=False)
        choosen_data = list(np.sort(choosen_data))
        i = 0
        while i < self.embedding_size:
            # print(i)
            if connection_rule == 'max':
                a_data = self.data[choosen_data.pop()]
                sorted_dims = np.argsort(a_data)
                synapse_index = sorted_dims[-num_projections:]
            elif connection_rule == 'random_choice':
                a_data = data[choosen_data.pop()]
                a_data = np.array(a_data, dtype=float) + 1e-10
                synapse_index = np.random.choice(data.shape[1], size=num_projections, replace=False,
                                                 p=a_data/a_data.sum())
            xindices[:, i] = synapse_index
            if i > 0:
                for j in range(i):
                    # print(i, j)
                    if np.all(np.equal(xindices[:, i], xindices[:, j])):
                        break
                i += 1
            else:
                i += 1

        self.weights = np.zeros((data.shape[1], embedding_size), dtype=np.bool)
        self.weights[xindices, yindices] = True  # sparse projection vectors

        all_activations = (self.data @ self.weights)
        xindices = np.arange(data.shape[0])[:, None]
        yindices = all_activations.argsort(axis=1)[:, -hash_length:]
        self.hashes = np.zeros_like(all_activations, dtype=np.bool)
        # threshold=np.sort(all_activations,axis=1)[:,-hash_length][:,None]
        self.hashes[xindices, yindices] = True  # choose topk activations
        # self.dense_activations=all_activations
        # self.sparse_activations=self.hashes.astype(np.float32)*all_activations #elementwise product
        self.maxl1distance = 2 * self.hash_length
        self.lowd_hashes = all_activations.reshape((-1, hash_length, K)).sum(axis=-1) > 0

    def query(self, qidx, nnn, not_olap=False):
        ones_indexs = np.array(np.where(self.hashes[qidx, :])).ravel()
        # L1_distances_full = np.sum(np.abs(self.hashes[qidx, :] ^ self.hashes), axis=1)
        L1_distances = np.sum(np.abs(self.hashes[qidx, ones_indexs] ^ self.hashes[:, ones_indexs]), axis=1)
        # np.sum(np.bitwise_xor(self.hashes[qidx,:],self.hashes),axis=1)
        nnn = min(self.hashes.shape[0], nnn)
        if not_olap:
            no_overlaps = np.sum(L1_distances == self.maxl1distance)
            return no_overlaps

        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)][:nnn]
        # print(L1_distances[NNs]) #an interesting property of this hash is that the L1 distances are always even
        return NNs

class FlylshDevelopThreshold(flylsh_quick_retrive):
    def __init__(self, data, hash_length, sampling_ratio, embedding_size, threshold=10, limited_occupation=False,
                 connection_rule='max', binary_method="top", connection_threshold=0, shift=True):
        """
        data: Nxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m, number of KCs
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.hash_length = hash_length
        self.embedding_size = embedding_size
        K = embedding_size // hash_length
        # self.data = (data - np.mean(data, axis=1)[:, None])
        data_mean = np.mean(data, axis=1)
        if shift:
            self.data = (data - data_mean[:, None])
        else:
            self.data = data
        # self.num_connections_per_dim_target = num_connections_per_dim_target
        self.weights = np.zeros((data.shape[1], embedding_size), dtype=np.bool)
        self.threshold = threshold
        num_projections = int(sampling_ratio * data.shape[1])  ## number of post synapse of KC
        # yindices = np.arange(embedding_size)[None, :]
        # xindices = np.zeros((num_projections, self.embedding_size), dtype=np.int_)
        # choosen_data = list(np.random.choice(data.shape[0], size=int(self.embedding_size * 1.5), replace=False))
        self.index_of_KC_to_learn = 0
        index_of_data = 0
        num_need_not_to_learn = 0
        round = 1
        number_of_learning_times = np.zeros(self.data.shape[0])
        if limited_occupation:
            self.number_of_occupations = np.zeros(self.data.shape[1])
        while self.index_of_KC_to_learn < self.embedding_size:
            # print(i)
            # a_shifted_data = self.data[np.random.choice(data.shape[0])]
            a_shifted_data = self.data[index_of_data]
            a_origin_data = data[index_of_data]
            if self.index_of_KC_to_learn > 0:
                existing_activity = (a_shifted_data @ self.weights[:, :self.index_of_KC_to_learn])
                if np.all(np.less(existing_activity, self.threshold)) and number_of_learning_times[index_of_data]==0:
                    if connection_rule == 'max':
                        if limited_occupation:
                            a_data_occupied = a_origin_data / (self.number_of_occupations + 1)
                            sorted_dims = np.argsort(a_data_occupied)
                        else:
                            sorted_dims = np.argsort(a_shifted_data)
                        synapse_index = sorted_dims[-num_projections:]
                    elif connection_rule == 'random_choice':
                        if limited_occupation:
                            a_data_occupied = a_origin_data / (self.number_of_occupations + 1)
                            if np.any(a_data_occupied < 0):
                                a_data_occupied = a_data_occupied - np.min(a_data_occupied)
                            temp_data = a_data_occupied + 1e-10
                        else:
                            if np.any(a_origin_data < 0):
                                a_origin_data = a_origin_data - np.min(a_origin_data)
                            temp_data = a_origin_data + 1e-10
                        choice_p = temp_data / temp_data.sum()
                        synapse_index = np.random.choice(data.shape[1], size=num_projections, replace=False,
                                                         p=choice_p)
                    elif connection_rule == 'threshold':
                        if limited_occupation:
                            a_data_occupied = a_origin_data / (self.number_of_occupations + 1)
                            synapse_index = np.where(a_data_occupied > connection_threshold)
                        else:
                            synapse_index = np.where(a_origin_data > connection_threshold)
                    self.weights[synapse_index, self.index_of_KC_to_learn] = True
                    if limited_occupation:
                        self.number_of_occupations += self.weights[:, self.index_of_KC_to_learn]
                    number_of_learning_times[index_of_data] += 1
                    self.index_of_KC_to_learn += 1
                else:
                    num_need_not_to_learn += 1
            else:
                if connection_rule == 'max':
                    sorted_dims = np.argsort(a_shifted_data)
                    synapse_index = sorted_dims[-num_projections:]
                elif connection_rule == 'random_choice':
                    if np.any(a_origin_data < 0):
                        a_origin_data = a_origin_data - np.min(a_origin_data)
                    temp_data = a_origin_data + 1e-10
                    choice_p = temp_data / temp_data.sum()
                    synapse_index = np.random.choice(data.shape[1], size=num_projections, replace=False,
                                                     p=choice_p)
                elif connection_rule == 'threshold':
                    synapse_index = np.where(a_origin_data > connection_threshold)
                self.weights[synapse_index, self.index_of_KC_to_learn] = True
                if limited_occupation:
                    self.number_of_occupations += self.weights[:, self.index_of_KC_to_learn]
                number_of_learning_times[index_of_data] += 1
                self.index_of_KC_to_learn += 1

            index_of_data += 1
            if index_of_data == data.shape[0]:
                self.threshold *= 1.2
                index_of_data = 0
                round += 1
        print(str(round) + ' rounds and ' + str(num_need_not_to_learn) + ' times need not to learn.')



        # self.weights = np.zeros((data.shape[1], embedding_size), dtype=np.bool)
        # self.weights[xindices, yindices] = True  # sparse projection vectors

        all_activations = (self.data @ self.weights)
        xindices = np.arange(data.shape[0])[:, None]
        if binary_method == "top":
            yindices = all_activations.argsort(axis=1)[:, -hash_length:]
        elif binary_method == "threshold":
            yindices = all_activations > 0
        self.hashes = np.zeros_like(all_activations, dtype=np.bool)
        # threshold=np.sort(all_activations,axis=1)[:,-hash_length][:,None]
        self.hashes[xindices, yindices] = True  # choose topk activations
        # self.dense_activations=all_activations
        # self.sparse_activations=self.hashes.astype(np.float32)*all_activations #elementwise product
        self.maxl1distance = 2 * self.hash_length
        self.lowd_hashes = all_activations.reshape((-1, hash_length, K)).sum(axis=-1) > 0

    def query(self, qidx, nnn, not_olap=False):
        ones_indexs = np.array(np.where(self.hashes[qidx, :])).ravel()
        # L1_distances = np.sum(np.abs(self.hashes[qidx, :] ^ self.hashes), axis=1)
        L1_distances = np.sum(np.abs(self.hashes[qidx, ones_indexs] ^ self.hashes[:, ones_indexs]), axis=1)
        # np.sum(np.bitwise_xor(self.hashes[qidx,:],self.hashes),axis=1)
        nnn = min(self.hashes.shape[0], nnn)
        if not_olap:
            no_overlaps = np.sum(L1_distances == self.maxl1distance)
            return no_overlaps

        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)][:nnn]
        # print(L1_distances[NNs]) #an interesting property of this hash is that the L1 distances are always even
        return NNs




def true_nns(data, qidx, nnn):
    sample = data[qidx, :]
    tnns = np.sum((data - sample) ** 2, axis=1).argsort()[:nnn + 1]
    tnns = tnns[(tnns != qidx)]
    print( "The ture nearest neighours are found for qidx {}".format(qidx))
    if nnn < data.shape[0]:
        assert len(tnns) == nnn, 'nnn={}'.format(nnn)
    return tnns


def construct_true_nns(data, indices, nnn, parallel=False, number_of_process=0):
    all_NNs = np.zeros((len(indices), nnn))
    if parallel:
        import multiprocessing
        cup_number = multiprocessing.cpu_count()
        if number_of_process == 0:
            number_of_process = NUMBER_OF_PROCESSES
        with multiprocessing.Pool(processes=number_of_process) as pool:
            all_NNs = pool.starmap(true_nns, ([data, i, nnn] for i in indices))
        all_NNs = np.vstack(all_NNs)
    else:
        for idx1, idx2 in tqdm(enumerate(indices), total=len(indices)):
            all_NNs[idx1, :] = true_nns(data, idx2, nnn)
    return all_NNs


def single_test_result_collector(hash_length, test_index, embedding_size, inputs_, sampling_ratio, nnn, all_NNs,
                                 all_expriments, all_MAPs, threshold, shift, all_retriving_time={}):
    MAPs, retriving_time = single_test(hash_length, test_index, embedding_size, inputs_, sampling_ratio, nnn, all_NNs, all_expriments,
                       threshold, shift)
    for a_key in all_expriments:
        all_MAPs[hash_length][a_key][test_index] = MAPs[a_key]
        all_retriving_time[hash_length][a_key][test_index] = retriving_time[a_key]

def single_test(hash_length, test_index, embedding_size, inputs_, sampling_ratio, nnn, all_NNs, all_expriments,
                threshold=10, shift=True):
    seed = hash_length * test_index * embedding_size * sampling_ratio * nnn
    random.seed(seed)
    np.random.seed(int(seed))
    MAPs = {}
    model = {}
    retriving_time = {}
    functions = {'Fly': flylsh_quick_retrive,  'FlylshDevelope': FlylshDevelope, 'LSH': LSH,
                 'FlylshDevelopThreshold': FlylshDevelopThreshold,
                 'FlylshDevelopeRandomChoice': FlylshDevelope,
                 'FlylshDevelopThresholdRandomChoice': FlylshDevelopThreshold,
                 'DenseFly':denseflylsh}
    for expriment in all_expriments:
        # print(expriment)
        if expriment == 'LSH':
            model[expriment] = functions[expriment](inputs_, hash_length)
        elif expriment == 'FlylshDevelopeRandomChoice':
            model[expriment] = functions[expriment](inputs_, hash_length, sampling_ratio, embedding_size,
                                                    connection_rule='random_choice', shift=shift)
        elif expriment == 'FlylshDevelopThresholdRandomChoice':
            model[expriment] = functions[expriment](inputs_, hash_length, sampling_ratio, embedding_size,
                                                    connection_rule='random_choice', threshold=threshold, shift=shift)
        elif expriment == "FlylshDevelopThreshold":
            model[expriment] = functions[expriment](inputs_, hash_length, sampling_ratio, embedding_size,
                                threshold = threshold, shift=shift)
        else:
            model[expriment] = functions[expriment](inputs_, hash_length, sampling_ratio, embedding_size, shift=shift)
        start_time = time.time()
        MAPs[expriment] = model[expriment].findmAP_given_true(nnn, 1000, all_NNs)
        retriving_time[expriment] = time.time()-start_time
        msg = expriment + ' mean average precision with harsh_length {:d} in test {:d} is equal to {:.4f}. Time to retrieve: {:.4f}'.format(
            hash_length, test_index, MAPs[expriment], retriving_time[expriment])
        print(msg)

    return MAPs, retriving_time


if __name__ == '__main__':
    data_set_name = 'MNIST' # 'CIFAR10' #'GLOVE'# 'SIFT' # 'MNIST' #'CIFAR10' #'FMNIST'
    task = "DevFly" #retrive_time
    TIME_OF_RECORDING = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # data_set = Dataset('MNIST')
    data_set = Dataset(data_set_name)
    # data_set2 = Dataset('CIFAR10')
    input_dim = data_set.indim   # d
    max_index = 1000
    sampling_ratio = 0.10
    nnn = 200  # number of nearest neighbours to compare, 2% of max_index as in paper
    hash_lengths = [2, 8, 16, 24, 32] # [2, 4, 8, 12, 16, 20, 24, 28, 32]
    number_of_tests = 2
    ratio = 20
    sorted_data = True
    shift = False

    import matplotlib.pyplot as plt
    if data_set_name =='CIFAR10':
        inputs_ = data_set.train_batches(batch_size=max_index)
        inputs_ = list(inputs_.__next__())[0]
        im = inputs_[0, :]
        im_r = im[0:1024].reshape(32, 32)
        im_g = im[1024:2048].reshape(32, 32)
        im_b = im[2048:].reshape(32, 32)
        img = np.dstack((im_r, im_g, im_b))
        plt.imshow(img)
        plt.show()
    elif data_set_name == 'FMNIST' or data_set_name == 'MNIST':
        inputs_ = np.array(data_set.data['images'][:max_index])
        labels_ = data_set.data['labels'][:max_index]
        if sorted_data:
            sorted_idx = np.argsort(labels_)
            inputs_ = inputs_[sorted_idx]
            labels_ = np.array(labels_)[sorted_idx]
        plt.imshow(inputs_[0, :].reshape(28, 28))
        plt.show()
    elif data_set_name == 'SIFT':
        inputs_ = data_set.data[:max_index]
    elif data_set_name == 'GLOVE':
        inputs_ = np.array(list(data_set.data.values())[:max_index])

    if data_set_name == 'GLOVE':
        threshold = 1
    else:
        threshold = 10

    all_MAPs = {}
    all_retriving_time = {}
    centered_data = (inputs_ - np.mean(inputs_, axis=1)[:, None])
    all_NNs = construct_true_nns(centered_data, range(inputs_.shape[0]), nnn, parallel=True, number_of_process=5)
    print('All nearest neighbours are found')

    parallel = True
    all_expriments = ['Fly', 'LSH', 'FlylshDevelope', 'LSH','FlylshDevelopThreshold',
                       'FlylshDevelopeRandomChoice', 'FlylshDevelopThresholdRandomChoice', 'DenseFly']
    for hash_length in hash_lengths:  # k
        all_MAPs[hash_length] = {}
        all_retriving_time[hash_length] = {}
        for a_key in all_expriments:
            all_MAPs[hash_length][a_key] = [0 for _ in range(number_of_tests)]
            all_retriving_time[hash_length][a_key] = [0 for _ in range(number_of_tests)]
    if parallel:
        import multiprocessing

        cup_number = multiprocessing.cpu_count()
        for hash_length in hash_lengths:  # k
            embedding_size = int(ratio * hash_length)  # int(10*input_dim) #20k or 10d
            with multiprocessing.Pool(processes=NUMBER_OF_PROCESSES) as pool:
                results = pool.starmap(single_test, (
                    [hash_length, i, embedding_size, inputs_, sampling_ratio, nnn, all_NNs, all_expriments,  threshold, shift]
                    for i in range(number_of_tests)))
            for i in range(number_of_tests):
                for a_key in all_expriments:
                    # if results[i].has_key[a_key]:
                    all_MAPs[hash_length][a_key][i] = results[i][0][a_key]
                    all_retriving_time[hash_length][a_key][i] = results[i][1][a_key]
    else:
        for hash_length in hash_lengths:  # k
            embedding_size = int(ratio * hash_length)  # int(10*input_dim) #20k or 10d
            for test_index in tqdm(range(number_of_tests)):
                single_test_result_collector(hash_length, test_index, embedding_size, inputs_, sampling_ratio, nnn,
                                             all_NNs, all_expriments, all_MAPs, threshold, shift, all_retriving_time)

    print(all_MAPs)

    import pickle

    result_path = os.path.join("../recording/DevFly", data_set_name, str(sampling_ratio), str(ratio),
                               str(sorted_data), str(shift), TIME_OF_RECORDING)
    print(result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(os.path.join(result_path, 'all_MAPs'+str(max_index)+'.pkl'), 'wb') as record_file:
        pickle.dump(all_MAPs, record_file)
    pass
    with open(os.path.join(result_path, 'all_retriving_time'+str(max_index)+'.pkl'), 'wb') as record_file:
        pickle.dump(all_retriving_time, record_file)
    pass
    with open(os.path.join(result_path, 'all_MAPs'+str(max_index)+'.pkl'), 'rb') as record_file:
        all_MAPs = pickle.load(record_file)
    figure_bokeh = plot_results(all_MAPs, plot_width=1600, plot_height=800)
    with open(os.path.join(result_path, 'all_retriving_time'+str(max_index)+'.pkl'), 'rb') as record_file:
        all_retriving_time = pickle.load(record_file)
    figure_bokeh = plot_results(all_retriving_time, plot_width=1600, plot_height=800, curve_ylabel='mean Average Time')

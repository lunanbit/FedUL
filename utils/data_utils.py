import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm


def get_set_sizes(sets, data_len):
    set_size = data_len // sets
    set_sizes = np.ones(sets) * set_size
    return set_sizes


def get_Pi_Multiclass(sets, classnum=10, noniid=False):
    Pi = []

    for i in range(sets):
        # randomly set prior
        this_Pi = np.random.rand(classnum) * 0.9 + 0.1

        this_Pi[i % classnum] *= 10

        this_Pi = this_Pi / np.sum(this_Pi)
        Pi.append(this_Pi)

    Pi = np.array(Pi)

    return Pi


def get_Pi_Multiclass_clientnoniid(sets, classnum=10, noniid=False, clientid=0):
    # def get_Pi_Multiclass(sets, classnum=10, noniid=False):
    Pi = []

    for i in range(sets):
        # randomly set prior
        this_Pi = np.random.rand(classnum) * 0.9 + 0.1

        # dominate step
        if noniid:
            this_Pi = np.random.rand(classnum) * 0.02 + 0.05
            this_Pi[clientid] = np.random.rand(1) * 0.2 + 0.3

            this_Pi = np.random.rand(classnum) * 0.02 + 0.05

        this_Pi = this_Pi / np.sum(this_Pi)
        Pi.append(this_Pi)

    Pi = np.array(Pi)

    return Pi


def get_test_sets_Multiclass(y_test, classnum=10, clientnum=5, clientsize=2000):
    class_idx = []
    for cls in range(classnum):
        this_idx = [i for i, x in enumerate(y_test) if x == cls]
        class_idx.append(this_idx)

    test_clients = ()
    for i in range(clientnum):
        for cls in range(classnum):
            np.random.shuffle(class_idx[cls])

            # uniformly distributed
            n_this = int(clientsize / classnum)

            if cls == 0:
                cur_set = np.array(class_idx[cls][:n_this])
            else:
                cur_set = np.concatenate((cur_set, class_idx[cls][:n_this])).astype(int)

            np.random.shuffle(cur_set)

        test_clients = test_clients + (torch.from_numpy(cur_set),)

    return test_clients


def get_U_sets_Multiclass(bags, y_train, y_indices, bag_sizes, thetas, classnum=10):
    class_idx = []
    for cls in range(classnum):
        this_idx = [y_indices[i] for i, x in enumerate(y_train) if x == cls]
        class_idx.append(this_idx)

    U_sets = ()
    size_bag = []
    # for every bag
    for i in range(bags):
        size_cls = []
        # for every class in a bag
        for cls in range(classnum):
            # shuffle data index list
            np.random.shuffle(class_idx[cls])
            # the number of data selected for this class: prior * setsize
            n_this = int(bag_sizes[i] * thetas[i][cls])

            # concatenate index
            if cls == 0:
                cur_set = np.array(class_idx[cls][:n_this])
            else:
                cur_set = np.concatenate((cur_set, class_idx[cls][:n_this])).astype(int)

            size_cls.append(len(class_idx[cls][:n_this]))

            # shuffle current set
            np.random.shuffle(cur_set)

        # concatenate different class data
        U_sets = U_sets + (torch.from_numpy(cur_set),)

        size_bag.append(np.array(size_cls) / sum(size_cls))

    # calculate priors corr for every U set
    sets_num_count = [len(U_sets[j]) for j in range(len(U_sets))]

    priors_corr = torch.from_numpy(
        np.array([sets_num_count[k] / sum(sets_num_count) for k in range(len(sets_num_count))]))

    bags_pi = np.array(size_bag)

    return U_sets, priors_corr, bags_pi


def get_iid_Pi(clientnum, setnum_perclient, classnum):
    for _ in range(clientnum):
        this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum))
        if _ == 0:
            iid_Pi = this_Pi
        else:
            iid_Pi += this_Pi
    iid_Pi = iid_Pi / torch.sum(iid_Pi, dim=0)

    return iid_Pi


def get_noniid_class_priority(client_num, classnum=10, dominate_rate=0.5):
    priority = []

    for client in range(client_num):
        this_label_shift = np.random.rand(classnum) * 0.1 + 0.45

        this_label_shift[(2 * client) % classnum] *= (4 / (1 - dominate_rate))
        this_label_shift[(2 * client + 1) % classnum] *= (4 / (1 - dominate_rate))

        this_label_shift = this_label_shift / np.sum(this_label_shift)
        priority.append(this_label_shift)

    return priority


def get_class_index(targets, classnum=10):
    indexs = []

    for cls in range(classnum):
        this_index = [index for (index, value) in enumerate(targets) if value == cls]
        indexs.append(this_index)

    return indexs


def noniid_split_dataset(oridata, lengths, classnum=10, dominate_rate=0.95):
    subsets = []
    priority = get_noniid_class_priority(len(lengths), classnum=classnum, dominate_rate=dominate_rate)

    targets = oridata.targets.tolist()
    class_index = get_class_index(targets, classnum=classnum)
    class_count = [0 for _ in range(classnum)]

    for l in range(len(lengths)):
        this_indices = []
        for cls in range(classnum):
            cls_num = int(priority[l][cls] * lengths[l])

            this_indices.extend(class_index[cls][class_count[cls]: class_count[cls] + cls_num])
            class_count[cls] += cls_num

        this_subset = torch.utils.data.Subset(oridata, this_indices)
        subsets.append(this_subset)

    return subsets


# split 10 class MNIST data to different clients and sets
def MNIST_SET_Multiclass(data_path='../data', clientnum=5, setnum_perclient=10, classnum=10, noniid=False):
    all_train_data = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=None)
    test_data = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=None)

    validation_data = all_train_data.data[48000:]
    validation_targets = all_train_data.targets[48000:]
    all_train_data.data = all_train_data.data[:48000]
    all_train_data.targets = all_train_data.targets[:48000]

    # split client bags
    client_train_size = len(all_train_data) // clientnum
    client_test_size = len(test_data) // clientnum
    client_validation_size = len(validation_data) // clientnum
    if noniid:
        client_train_sets = noniid_split_dataset(all_train_data, [client_train_size for _ in range(clientnum)])
    else:
        client_train_sets = torch.utils.data.random_split(all_train_data, [client_train_size for _ in range(clientnum)])

    # get uniformly distributed test data index
    validation_client_idxs = get_test_sets_Multiclass(validation_targets, classnum=classnum, clientnum=clientnum,
                                                      clientsize=client_validation_size)
    test_client_idxs = get_test_sets_Multiclass(test_data.targets, classnum=classnum, clientnum=clientnum,
                                                clientsize=client_test_size)

    client_train_data = []
    client_test_data = []
    client_validation_data = []

    # get Pis, prior test, prior corr
    client_Pi = []
    client_prior_test = []
    client_priors_corr = []

    if not noniid:
        iid_Pi = get_iid_Pi(clientnum, setnum_perclient, classnum)

    print('Spliting U sets for', clientnum, 'clients, each with', setnum_perclient, 'U sets...')
    # for every client
    for n in tqdm(range(clientnum)):
        if noniid:
            this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum))
        if not noniid:
            this_Pi = iid_Pi

        # w/o repeat
        this_set_sizes = get_set_sizes(setnum_perclient, len(client_train_sets[n]))
        this_U_sets, this_priors_corr, this_Pi = get_U_sets_Multiclass(setnum_perclient,
                                                                       client_train_sets[n].dataset.targets[
                                                                           client_train_sets[n].indices],
                                                                       client_train_sets[n].indices,
                                                                       this_set_sizes, this_Pi, classnum=classnum)

        client_Pi.append(torch.from_numpy(this_Pi))
        client_priors_corr.append(this_priors_corr)

        # get prior test
        this_prior_test = torch.from_numpy(np.array([(test_data.targets == m).sum() / float(len(test_data.targets))
                                                     for m in range(classnum)]))
        client_prior_test.append(this_prior_test)

        # set subsets labels, for every set in every client
        client_set_temp_data = None
        client_set_temp_targets = None
        for i in range(setnum_perclient):
            # w/o repeat
            this_set_temp_data = client_train_sets[n].dataset.data[this_U_sets[i]]

            # surrogate label as set index
            this_set_temp_targets = torch.ones(len(this_U_sets[i])) * i

            # concatenate data and labels
            if i == 0:
                client_set_temp_data = this_set_temp_data
                client_set_temp_targets = this_set_temp_targets
            else:
                client_set_temp_data = torch.cat((client_set_temp_data, this_set_temp_data))
                client_set_temp_targets = torch.cat((client_set_temp_targets, this_set_temp_targets))

        # store different clients' data and labels in a dict, for further load
        client_train_data.append({'images': client_set_temp_data, 'labels': client_set_temp_targets})
        client_test_data.append({'images': test_data.data[test_client_idxs[n]],
                                 'labels': test_data.targets[test_client_idxs[n]]})
        client_validation_data.append({'images': validation_data[validation_client_idxs[n]],
                                       'labels': validation_targets[validation_client_idxs[n]]})

    return client_train_data, client_validation_data, client_test_data, client_prior_test, client_priors_corr, client_Pi


# split 10 class CIFAR10 data to different clients and sets
def CIFAR10_SET_Multiclass(data_path='../data', clientnum=5, setnum_perclient=10, classnum=10, noniid=False):
    all_train_data = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transform=None)
    test_data = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=None)
    test_data.targets = np.array(test_data.targets)
    test_data.data = torch.from_numpy(test_data.data).permute(0, 3, 1, 2)
    all_train_data.data = torch.from_numpy(all_train_data.data).permute(0, 3, 1, 2)
    all_train_data.targets = torch.from_numpy(np.array(all_train_data.targets))

    validation_data = all_train_data.data[40000:]
    validation_targets = torch.tensor(all_train_data.targets[40000:])
    all_train_data.data = all_train_data.data[:40000]
    all_train_data.targets = all_train_data.targets[:40000]

    # split client bags
    client_train_size = len(all_train_data) // clientnum
    client_validation_size = len(validation_data) // clientnum
    client_test_size = len(test_data) // clientnum
    if noniid:
        client_train_sets = noniid_split_dataset(all_train_data, [client_train_size for _ in range(clientnum)])
    else:
        client_train_sets = torch.utils.data.random_split(all_train_data, [client_train_size for _ in range(clientnum)])

    # get uniformly distributed test data index
    validation_client_idxs = get_test_sets_Multiclass(validation_targets, classnum=classnum, clientnum=clientnum,
                                                      clientsize=client_validation_size)
    test_client_idxs = get_test_sets_Multiclass(test_data.targets, classnum=classnum, clientnum=clientnum,
                                                clientsize=client_test_size)

    client_train_data = []
    client_test_data = []
    client_validation_data = []

    # get Pis, prior test, prior corr
    client_Pi = []
    client_prior_test = []
    client_priors_corr = []

    print('Spliting U sets for', clientnum, 'clients, each with', setnum_perclient, 'U sets...')
    # for every client
    for n in tqdm(range(clientnum)):
        this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum, noniid=noniid))

        # w/o repeat
        this_set_sizes = get_set_sizes(setnum_perclient, len(client_train_sets[n]))
        this_U_sets, this_priors_corr, this_Pi = get_U_sets_Multiclass(setnum_perclient,
                                                                       client_train_sets[n].dataset.targets[
                                                                           client_train_sets[n].indices],
                                                                       client_train_sets[n].indices,
                                                                       this_set_sizes, this_Pi, classnum=classnum)

        client_Pi.append(torch.from_numpy(this_Pi))
        client_priors_corr.append(this_priors_corr)

        # get prior test
        this_prior_test = None

        client_prior_test.append(this_prior_test)

        # set subsets labels, for every set in every client
        client_set_temp_data = None
        client_set_temp_targets = None
        for i in range(setnum_perclient):
            # w/o repeat
            this_set_temp_data = client_train_sets[n].dataset.data[this_U_sets[i]]

            # surrogate label as set index
            this_set_temp_targets = torch.ones(len(this_U_sets[i])) * i

            # concatenate data and labels
            if i == 0:
                client_set_temp_data = this_set_temp_data
                client_set_temp_targets = this_set_temp_targets
            else:
                client_set_temp_data = torch.cat((client_set_temp_data, this_set_temp_data))
                client_set_temp_targets = torch.cat((client_set_temp_targets, this_set_temp_targets))

        # store different clients' data and labels in a dict, for further load
        client_train_data.append({'images': client_set_temp_data, 'labels': client_set_temp_targets})
        client_test_data.append({'images': test_data.data[test_client_idxs[n]],
                                 'labels': test_data.targets[test_client_idxs[n]]})
        client_validation_data.append({'images': validation_data[validation_client_idxs[n]],
                                       'labels': validation_targets[validation_client_idxs[n]]})

    return client_train_data, client_validation_data, client_test_data, client_prior_test, client_priors_corr, client_Pi


def MNIST_PL(data_path='../data', clientnum=5, setnum_perclient=10, classnum=10, noniid=False):
    all_train_data = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=None)
    test_data = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=None)

    validation_data = all_train_data.data[48000:]
    validation_targets = all_train_data.targets[48000:]
    all_train_data.data = all_train_data.data[:48000]
    all_train_data.targets = all_train_data.targets[:48000]

    # split client bags
    client_train_size = len(all_train_data) // clientnum
    client_validation_size = len(validation_data) // clientnum
    client_test_size = len(test_data) // clientnum
    if noniid:
        client_train_sets = noniid_split_dataset(all_train_data, [client_train_size for _ in range(clientnum)])
    else:
        client_train_sets = torch.utils.data.random_split(all_train_data, [client_train_size for _ in range(clientnum)])

    # get uniformly distributed test data index
    validation_client_idxs = get_test_sets_Multiclass(validation_targets, classnum=classnum, clientnum=clientnum,
                                                      clientsize=client_validation_size)
    test_client_idxs = get_test_sets_Multiclass(test_data.targets, classnum=classnum, clientnum=clientnum,
                                                clientsize=client_test_size)

    client_train_data = []
    client_test_data = []
    client_validation_data = []

    # get Pis, prior test, prior corr
    client_Pi = []
    client_prior_test = []
    client_priors_corr = []

    if not noniid:
        iid_Pi = get_iid_Pi(clientnum, setnum_perclient, classnum)

    print('Spliting U sets for', clientnum, 'clients, each with', setnum_perclient, 'U sets...')
    # for every client
    for n in tqdm(range(clientnum)):
        if noniid:
            this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum))
        if not noniid:
            this_Pi = iid_Pi

        this_labels = torch.max(this_Pi, dim=1).indices

        # w/o repeat
        this_set_sizes = get_set_sizes(setnum_perclient, len(client_train_sets[n]))
        this_U_sets, this_priors_corr, this_Pi = get_U_sets_Multiclass(setnum_perclient,
                                                                       client_train_sets[n].dataset.targets[
                                                                           client_train_sets[n].indices],
                                                                       client_train_sets[n].indices,
                                                                       this_set_sizes, this_Pi, classnum=classnum)

        client_Pi.append(this_Pi)
        client_priors_corr.append(this_priors_corr)

        # get prior test
        this_prior_test = None
        client_prior_test.append(this_prior_test)

        # set subsets labels, for every set in every client
        client_set_temp_data = None
        client_set_temp_targets = None
        for i in range(setnum_perclient):
            # w/o repeat
            this_set_temp_data = client_train_sets[n].dataset.data[this_U_sets[i]]

            # surrogate label as set index
            this_set_temp_targets = torch.ones(len(this_U_sets[i])) * this_labels[i]

            # concatenate data and labels
            if i == 0:
                client_set_temp_data = this_set_temp_data
                client_set_temp_targets = this_set_temp_targets
            else:
                client_set_temp_data = torch.cat((client_set_temp_data, this_set_temp_data))
                client_set_temp_targets = torch.cat((client_set_temp_targets, this_set_temp_targets))

        # store different clients' data and labels in a dict, for further load
        client_train_data.append({'images': client_set_temp_data, 'labels': client_set_temp_targets})
        client_test_data.append({'images': test_data.data[test_client_idxs[n]],
                                 'labels': test_data.targets[test_client_idxs[n]]})
        client_validation_data.append({'images': validation_data[validation_client_idxs[n]],
                                       'labels': validation_targets[validation_client_idxs[n]]})

    return client_train_data, client_validation_data, client_test_data, client_prior_test, client_priors_corr, client_Pi


def CIFAR10_PL(data_path='../data', clientnum=5, setnum_perclient=10, classnum=10, noniid=False):
    all_train_data = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transform=None)
    test_data = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=None)
    test_data.targets = np.array(test_data.targets)
    test_data.data = torch.from_numpy(test_data.data).permute(0, 3, 1, 2)
    all_train_data.data = torch.from_numpy(all_train_data.data).permute(0, 3, 1, 2)
    all_train_data.targets = torch.from_numpy(np.array(all_train_data.targets))

    validation_data = all_train_data.data[40000:]
    validation_targets = torch.tensor(all_train_data.targets[40000:])
    all_train_data.data = all_train_data.data[:40000]
    all_train_data.targets = all_train_data.targets[:40000]

    # split client bags
    client_train_size = len(all_train_data) // clientnum
    client_validation_size = len(validation_data) // clientnum
    client_test_size = len(test_data) // clientnum
    if noniid:
        client_train_sets = noniid_split_dataset(all_train_data, [client_train_size for _ in range(clientnum)])
    else:
        client_train_sets = torch.utils.data.random_split(all_train_data, [client_train_size for _ in range(clientnum)])

    # get uniformly distributed test data index
    validation_client_idxs = get_test_sets_Multiclass(validation_targets, classnum=classnum, clientnum=clientnum,
                                                      clientsize=client_validation_size)
    test_client_idxs = get_test_sets_Multiclass(test_data.targets, classnum=classnum, clientnum=clientnum,
                                                clientsize=client_test_size)

    client_train_data = []
    client_validation_data = []
    client_test_data = []

    # get Pis, prior test, prior corr
    client_Pi = []
    client_prior_test = []
    client_priors_corr = []

    if not noniid:
        iid_Pi = get_iid_Pi(clientnum, setnum_perclient, classnum)

    print('Spliting U sets for', clientnum, 'clients, each with', setnum_perclient, 'U sets...')
    # for every client
    for n in tqdm(range(clientnum)):
        if noniid:
            this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum))
        if not noniid:
            this_Pi = iid_Pi

        this_labels = torch.max(this_Pi, dim=1).indices

        # w/o repeat
        this_set_sizes = get_set_sizes(setnum_perclient, len(client_train_sets[n]))
        this_U_sets, this_priors_corr, this_Pi = get_U_sets_Multiclass(setnum_perclient,
                                                                       client_train_sets[n].dataset.targets[
                                                                           client_train_sets[n].indices],
                                                                       client_train_sets[n].indices,
                                                                       this_set_sizes, this_Pi, classnum=classnum)

        client_Pi.append(this_Pi)
        client_priors_corr.append(this_priors_corr)

        # get prior test
        this_prior_test = None
        client_prior_test.append(this_prior_test)

        # set subsets labels, for every set in every client
        client_set_temp_data = None
        client_set_temp_targets = None
        for i in range(setnum_perclient):
            # w/o repeat
            this_set_temp_data = client_train_sets[n].dataset.data[this_U_sets[i]]

            # surrogate label as set index
            this_set_temp_targets = torch.ones(len(this_U_sets[i])) * this_labels[i]

            # concatenate data and labels
            if i == 0:
                client_set_temp_data = this_set_temp_data
                client_set_temp_targets = this_set_temp_targets
            else:
                client_set_temp_data = torch.cat((client_set_temp_data, this_set_temp_data))
                client_set_temp_targets = torch.cat((client_set_temp_targets, this_set_temp_targets))

        # store different clients' data and labels in a dict, for further load
        client_train_data.append({'images': client_set_temp_data, 'labels': client_set_temp_targets})
        client_test_data.append({'images': test_data.data[test_client_idxs[n]],
                                 'labels': test_data.targets[test_client_idxs[n]]})
        client_validation_data.append({'images': validation_data[validation_client_idxs[n]],
                                       'labels': validation_targets[validation_client_idxs[n]]})

    return client_train_data, client_validation_data, client_test_data, client_prior_test, client_priors_corr, client_Pi


def MNIST_LLP(data_path='../data', clientnum=5, setnum_perclient=10, classnum=10, noniid=False):
    all_train_data = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=None)
    test_data = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=None)

    validation_data = all_train_data.data[48000:]
    validation_targets = all_train_data.targets[48000:]
    all_train_data.data = all_train_data.data[:48000]
    all_train_data.targets = all_train_data.targets[:48000]

    # split client bags
    client_train_size = len(all_train_data) // clientnum
    client_validation_size = len(validation_data) // clientnum
    client_test_size = len(test_data) // clientnum
    if noniid:
        client_train_sets = noniid_split_dataset(all_train_data, [client_train_size for _ in range(clientnum)])
    else:
        client_train_sets = torch.utils.data.random_split(all_train_data, [client_train_size for _ in range(clientnum)])

    # get uniformly distributed test data index
    validation_client_idxs = get_test_sets_Multiclass(validation_targets, classnum=classnum, clientnum=clientnum,
                                                      clientsize=client_validation_size)
    test_client_idxs = get_test_sets_Multiclass(test_data.targets, classnum=classnum, clientnum=clientnum,
                                                clientsize=client_test_size)

    client_train_data = []
    client_test_data = []
    client_validation_data = []

    # get Pis, prior test, prior corr
    client_U_Sets = []
    client_prior_test = []
    client_priors_corr = []

    if not noniid:
        iid_Pi = get_iid_Pi(clientnum, setnum_perclient, classnum)

    print('Spliting U sets for', clientnum, 'clients, each with', setnum_perclient, 'U sets...')
    # for every client
    for n in tqdm(range(clientnum)):
        if noniid:
            this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum))
        if not noniid:
            this_Pi = iid_Pi

        this_labels = torch.max(this_Pi, dim=1).indices

        # w/o repeat
        this_set_sizes = get_set_sizes(setnum_perclient, len(client_train_sets[n]))
        this_U_sets, this_priors_corr, this_Pi = get_U_sets_Multiclass(setnum_perclient,
                                                                       client_train_sets[n].dataset.targets[
                                                                           client_train_sets[n].indices],
                                                                       client_train_sets[n].indices,
                                                                       this_set_sizes, this_Pi, classnum=classnum)

        client_priors_corr.append(this_priors_corr)

        # get prior test
        this_prior_test = None
        client_prior_test.append(this_prior_test)

        # set subsets labels, for every set in every client
        client_set_temp_data = None
        client_set_temp_targets = None

        this_bags = []
        samples_count = 0
        for i in range(setnum_perclient):
            # w/o repeat
            this_set_temp_data = client_train_sets[n].dataset.data[this_U_sets[i]]

            # surrogate label as set index
            this_set_temp_targets = client_train_sets[n].dataset.targets[this_U_sets[i]]

            this_bags.append([_ for _ in range(samples_count, samples_count + len(this_set_temp_targets))])
            samples_count += len(this_set_temp_targets)

            # print(len(this_bags[i]), samples_count)

            this_set_temp_targets = torch.nn.functional.one_hot(this_set_temp_targets, num_classes=classnum)

            # concatenate data and labels
            if i == 0:
                client_set_temp_data = this_set_temp_data
                client_set_temp_targets = this_set_temp_targets
            else:
                client_set_temp_data = torch.cat((client_set_temp_data, this_set_temp_data))
                client_set_temp_targets = torch.cat((client_set_temp_targets, this_set_temp_targets))

        # store different clients' data and labels in a dict, for further load
        client_train_data.append({'images': client_set_temp_data, 'labels': client_set_temp_targets})
        client_test_data.append({'images': test_data.data[test_client_idxs[n]],
                                 'labels': test_data.targets[test_client_idxs[n]]})
        client_validation_data.append({'images': validation_data[validation_client_idxs[n]],
                                       'labels': validation_targets[validation_client_idxs[n]]})
        client_U_Sets.append(this_bags)

    return client_train_data, client_validation_data, client_test_data, client_prior_test, client_priors_corr, client_U_Sets


def CIFAR10_LLP(data_path='../data', clientnum=5, setnum_perclient=10, classnum=10, noniid=False):
    all_train_data = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transform=None)
    test_data = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=None)
    test_data.targets = torch.from_numpy(np.array(test_data.targets))
    test_data.data = torch.from_numpy(test_data.data).permute(0, 3, 1, 2)
    all_train_data.data = torch.from_numpy(all_train_data.data).permute(0, 3, 1, 2)
    all_train_data.targets = torch.from_numpy(np.array(all_train_data.targets))

    validation_data = all_train_data.data[40000:]
    validation_targets = torch.tensor(all_train_data.targets[40000:])
    all_train_data.data = all_train_data.data[:40000]
    all_train_data.targets = all_train_data.targets[:40000]

    # split client bags
    client_train_size = len(all_train_data) // clientnum
    client_validation_size = len(validation_data) // clientnum
    client_test_size = len(test_data) // clientnum
    if noniid:
        client_train_sets = noniid_split_dataset(all_train_data, [client_train_size for _ in range(clientnum)])
    else:
        client_train_sets = torch.utils.data.random_split(all_train_data, [client_train_size for _ in range(clientnum)])

    # get uniformly distributed test data index
    validation_client_idxs = get_test_sets_Multiclass(validation_targets, classnum=classnum, clientnum=clientnum,
                                                      clientsize=client_validation_size)
    test_client_idxs = get_test_sets_Multiclass(test_data.targets, classnum=classnum, clientnum=clientnum,
                                                clientsize=client_test_size)

    client_train_data = []
    client_test_data = []
    client_validation_data = []

    # get Pis, prior test, prior corr
    client_Pi = []
    client_prior_test = []
    client_priors_corr = []
    client_U_Sets = []

    if not noniid:
        iid_Pi = get_iid_Pi(clientnum, setnum_perclient, classnum)

    print('Spliting U sets for', clientnum, 'clients, each with', setnum_perclient, 'U sets...')
    # for every client
    for n in tqdm(range(clientnum)):
        if noniid:
            this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum))
        if not noniid:
            this_Pi = iid_Pi

        this_labels = torch.max(this_Pi, dim=1).indices

        # w/o repeat
        this_set_sizes = get_set_sizes(setnum_perclient, len(client_train_sets[n]))
        this_U_sets, this_priors_corr, this_Pi = get_U_sets_Multiclass(setnum_perclient,
                                                                       client_train_sets[n].dataset.targets[
                                                                           client_train_sets[n].indices],
                                                                       client_train_sets[n].indices,
                                                                       this_set_sizes, this_Pi, classnum=classnum)

        client_Pi.append(torch.from_numpy(this_Pi))
        client_priors_corr.append(this_priors_corr)

        # get prior test
        this_prior_test = None
        client_prior_test.append(this_prior_test)

        # set subsets labels, for every set in every client
        client_set_temp_data = None
        client_set_temp_targets = None

        this_bags = []
        samples_count = 0
        for i in range(setnum_perclient):
            # w/o repeat
            this_set_temp_data = client_train_sets[n].dataset.data[this_U_sets[i]]

            # surrogate label as set index
            this_set_temp_targets = client_train_sets[n].dataset.targets[this_U_sets[i]]

            this_bags.append([_ for _ in range(samples_count, samples_count + len(this_set_temp_targets))])
            samples_count += len(this_set_temp_targets)

            this_set_temp_targets = torch.nn.functional.one_hot(this_set_temp_targets, num_classes=classnum)

            # concatenate data and labels
            if i == 0:
                client_set_temp_data = this_set_temp_data
                client_set_temp_targets = this_set_temp_targets
            else:
                client_set_temp_data = torch.cat((client_set_temp_data, this_set_temp_data))
                client_set_temp_targets = torch.cat((client_set_temp_targets, this_set_temp_targets))

        # store different clients' data and labels in a dict, for further load
        client_train_data.append({'images': client_set_temp_data, 'labels': client_set_temp_targets})
        client_test_data.append({'images': test_data.data[test_client_idxs[n]],
                                 'labels': test_data.targets[test_client_idxs[n]]})
        client_validation_data.append({'images': validation_data[validation_client_idxs[n]],
                                       'labels': validation_targets[validation_client_idxs[n]]})

        client_U_Sets.append(this_bags)

    return client_train_data, client_validation_data, client_test_data, client_prior_test, client_priors_corr, client_U_Sets


# basic dataset class for load data
class BaiscDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(transforms.ToPILImage()(image))

        return image, label


class LLPDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.images = data['images']
        self.labels = data['labels']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label


# split 10 class MNIST data to different clients and sets
def MNIST_UPPER_BOUND(data_path='../data', clientnum=5, setnum_perclient=10, classnum=10, noniid=False):
    all_train_data = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=None)
    test_data = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=None)

    validation_data = all_train_data.data[48000:]
    validation_targets = all_train_data.targets[48000:]
    all_train_data.data = all_train_data.data[:4800]
    all_train_data.targets = all_train_data.targets[:4800]

    # split client bags
    client_train_size = len(all_train_data) // clientnum
    client_validation_size = len(validation_data) // clientnum
    client_test_size = len(test_data) // clientnum
    if noniid:
        client_train_sets = noniid_split_dataset(all_train_data, [client_train_size for _ in range(clientnum)])
    else:
        client_train_sets = torch.utils.data.random_split(all_train_data, [client_train_size for _ in range(clientnum)])

    # get uniformly distributed test data index
    validation_client_idxs = get_test_sets_Multiclass(validation_targets, classnum=classnum, clientnum=clientnum,
                                                      clientsize=client_validation_size)
    test_client_idxs = get_test_sets_Multiclass(test_data.targets, classnum=classnum, clientnum=clientnum,
                                                clientsize=client_test_size)

    client_train_data = []
    client_test_data = []
    client_validation_data = []

    # get Pis, prior test, prior corr
    client_Pi = []
    client_prior_test = []
    client_priors_corr = []

    if not noniid:
        iid_Pi = get_iid_Pi(clientnum, setnum_perclient, classnum)

    print('Spliting U sets for', clientnum, 'clients, each with', setnum_perclient, 'U sets...')
    # for every client
    for n in tqdm(range(clientnum)):
        if noniid:
            this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum))
        if not noniid:
            this_Pi = iid_Pi

        # w/o repeat
        this_set_sizes = get_set_sizes(setnum_perclient, len(client_train_sets[n]))
        this_U_sets, this_priors_corr, this_Pi = get_U_sets_Multiclass(setnum_perclient,
                                                                       client_train_sets[n].dataset.targets[
                                                                           client_train_sets[n].indices],
                                                                       client_train_sets[n].indices,
                                                                       this_set_sizes, this_Pi, classnum=classnum)

        client_Pi.append(torch.from_numpy(this_Pi))
        client_priors_corr.append(this_priors_corr)

        # get prior test
        this_prior_test = torch.from_numpy(np.array([(test_data.targets == m).sum() / float(len(test_data.targets))
                                                     for m in range(classnum)]))
        client_prior_test.append(this_prior_test)

        # set subsets labels, for every set in every client
        client_set_temp_data = None
        client_set_temp_targets = None
        for i in range(setnum_perclient):
            # w/o repeat
            this_set_temp_data = client_train_sets[n].dataset.data[this_U_sets[i]]

            # surrogate label as set index
            this_set_temp_targets = client_train_sets[n].dataset.targets[this_U_sets[i]]

            # concatenate data and labels
            if i == 0:
                client_set_temp_data = this_set_temp_data
                client_set_temp_targets = this_set_temp_targets
            else:
                client_set_temp_data = torch.cat((client_set_temp_data, this_set_temp_data))
                client_set_temp_targets = torch.cat((client_set_temp_targets, this_set_temp_targets))

        # store different clients' data and labels in a dict, for further load
        client_train_data.append({'images': client_set_temp_data, 'labels': client_set_temp_targets})
        client_test_data.append({'images': test_data.data[test_client_idxs[n]],
                                 'labels': test_data.targets[test_client_idxs[n]]})
        client_validation_data.append({'images': validation_data[validation_client_idxs[n]],
                                       'labels': validation_targets[validation_client_idxs[n]]})

    return client_train_data, client_validation_data, client_test_data, client_prior_test, client_priors_corr, client_Pi


def CIFAR10_UPPER_BOUND(data_path='../data', clientnum=5, setnum_perclient=10, classnum=10, noniid=False):
    all_train_data = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transform=None)
    test_data = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=None)
    test_data.targets = np.array(test_data.targets)
    test_data.data = torch.from_numpy(test_data.data).permute(0, 3, 1, 2)
    all_train_data.data = torch.from_numpy(all_train_data.data).permute(0, 3, 1, 2)
    all_train_data.targets = torch.from_numpy(np.array(all_train_data.targets))

    validation_data = all_train_data.data[40000:]
    validation_targets = torch.tensor(all_train_data.targets[40000:])

    all_train_data.data = all_train_data.data[:4000]
    all_train_data.targets = all_train_data.targets[:4000]

    # split client bags
    client_train_size = len(all_train_data) // clientnum
    client_validation_size = len(validation_data) // clientnum
    client_test_size = len(test_data) // clientnum
    if noniid:
        client_train_sets = noniid_split_dataset(all_train_data, [client_train_size for _ in range(clientnum)])
    else:
        client_train_sets = torch.utils.data.random_split(all_train_data, [client_train_size for _ in range(clientnum)])

    # get uniformly distributed test data index
    validation_client_idxs = get_test_sets_Multiclass(validation_targets, classnum=classnum, clientnum=clientnum,
                                                      clientsize=client_validation_size)
    test_client_idxs = get_test_sets_Multiclass(test_data.targets, classnum=classnum, clientnum=clientnum,
                                                clientsize=client_test_size)

    client_train_data = []
    client_test_data = []
    client_validation_data = []

    # get Pis, prior test, prior corr
    client_Pi = []
    client_prior_test = []
    client_priors_corr = []

    if not noniid:
        iid_Pi = get_iid_Pi(clientnum, setnum_perclient, classnum)

    print('Spliting U sets for', clientnum, 'clients, each with', setnum_perclient, 'U sets...')
    # for every client
    for n in tqdm(range(clientnum)):
        if noniid:
            this_Pi = torch.from_numpy(get_Pi_Multiclass(setnum_perclient, classnum=classnum))
        if not noniid:
            this_Pi = iid_Pi

        # w/o repeat
        this_set_sizes = get_set_sizes(setnum_perclient, len(client_train_sets[n]))
        this_U_sets, this_priors_corr, this_Pi = get_U_sets_Multiclass(setnum_perclient,
                                                                       client_train_sets[n].dataset.targets[
                                                                           client_train_sets[n].indices],
                                                                       client_train_sets[n].indices,
                                                                       this_set_sizes, this_Pi, classnum=classnum)

        client_Pi.append(torch.from_numpy(this_Pi))
        client_priors_corr.append(this_priors_corr)

        # get prior test
        this_prior_test = None
        client_prior_test.append(this_prior_test)

        # set subsets labels, for every set in every client
        client_set_temp_data = None
        client_set_temp_targets = None
        for i in range(setnum_perclient):
            # w/o repeat
            this_set_temp_data = client_train_sets[n].dataset.data[this_U_sets[i]]

            # surrogate label as set index
            this_set_temp_targets = client_train_sets[n].dataset.targets[this_U_sets[i]]

            # concatenate data and labels
            if i == 0:
                client_set_temp_data = this_set_temp_data
                client_set_temp_targets = this_set_temp_targets
            else:
                client_set_temp_data = torch.cat((client_set_temp_data, this_set_temp_data))
                client_set_temp_targets = torch.cat((client_set_temp_targets, this_set_temp_targets))

        # store different clients' data and labels in a dict, for further load
        client_train_data.append({'images': client_set_temp_data, 'labels': client_set_temp_targets})
        client_test_data.append({'images': test_data.data[test_client_idxs[n]],
                                 'labels': test_data.targets[test_client_idxs[n]]})
        client_validation_data.append({'images': validation_data[validation_client_idxs[n]],
                                       'labels': validation_targets[validation_client_idxs[n]]})

    return client_train_data, client_validation_data, client_test_data

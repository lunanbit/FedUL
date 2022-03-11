import os
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--noniid', action='store_true', help='noniid sampling')
args = parser.parse_args()

methods = ['FedPL', 'FedLLP', 'FedLLPVAT', 'FedUL', 'FedAvg']
datasets = ['MNIST', 'CIFAR']
seeds = [0, 1, 2]
baseline_sets = [10, 20, 40]
upper_sets = [10]
noniid = str(args.noniid)

datadir = '../logs'

for dataset in datasets:
    print(dataset)
    print('=' * 20)
    for method in methods:
        print(method)
        if method == 'fedavg':
            sets = upper_sets
        else:
            sets = baseline_sets
        for set in sets:
            print('set', set, end=': ')

            this_error = []
            for seed in seeds:
                text_path = os.path.join(datadir, dataset.lower() + '_' + method.lower(),
                                         'fedavg/client5sets' + str(set) + 'seed' + str(
                                             seed) + noniid + 'error_rate.txt')

                if not os.path.exists(text_path):
                    continue

                text_file = open(text_path, 'r')

                data = text_file.readlines()
                final = data[-1]

                # if len(data) < 101:
                #     print('invalid!!!')

                this_error.append(float(final))

            if len(this_error) == 3:
                mean = np.mean(np.array(this_error))
                std = np.std(np.array(this_error))
                print(round(100 * mean, 2), round(100 * std, 2))
            else:
                print()

    print()

import os

methods = ['FedPL', 'FedLLP', 'FedLLPVAT', 'FedUL', 'FedAvg']
datasets = ['MNIST', 'CIFAR']
sets = [10, 20, 40]
seeds = [0, 1, 2]
mnistwdecay = [1e-5, 5e-6, 2e-6]
cifarwdecay = [2e-5, 1e-5, 4e-6]

for dataset in datasets:
    for i, setnum in enumerate(sets):
        for method in methods:
            for seed in seeds:
                if dataset.lower() == 'mnist':
                    command = 'python ../federated/%s_%s.py --setnum %d --wdecay %f --seed %d' % (
                    method.lower(), dataset.lower(), setnum, mnistwdecay[i], seed)
                elif dataset.lower() == 'cifar':
                    command = 'python ../federated/%s_%s.py --setnum %d --wdecay %f --seed %d' % (
                    method.lower(), dataset.lower(), setnum, cifarwdecay[i], seed)

                print(command)
                os.system(command)

for dataset in datasets:
    for seed in seeds:
        if dataset.lower() == 'mnist':
            command = 'python ../federated/fedavg_%s.py --setnum 10 --wdecay %f --seed %d' % (
            dataset.lower(), mnistwdecay[0], seed)
        elif dataset.lower() == 'cifar':
            command = 'python ../federated/fedavg_%s.py --setnum 10 --wdecay %f --seed %d' % (
            dataset.lower(), cifarwdecay[0], seed)

        print(command)
        os.system(command)

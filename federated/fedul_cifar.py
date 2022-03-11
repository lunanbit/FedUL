import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import copy
from nets.models import CIFARModel
import argparse
import numpy as np
import torchvision.transforms as transforms
from utils import data_utils


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def prepare_data(args):
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Get splited data
    if args.classnum == 2:
        client_train_data, client_test_data, prior_test, client_priors_corr, client_Pi = \
            data_utils.CIFAR10_SET(
                data_path="../data",
                clientnum=args.clientnum,
                setnum_perclient=args.setnum)
    else:
        client_train_data, client_validation_data, client_test_data, prior_test, client_priors_corr, client_Pi = \
            data_utils.CIFAR10_SET_Multiclass(
                data_path="../data",
                clientnum=args.clientnum,
                setnum_perclient=args.setnum,
                noniid=args.noniid)

    # get dataloaders
    train_loaders = []
    validation_loaders = []
    test_loaders = []

    for i, this_client_data in enumerate(client_train_data):
        this_train_set = data_utils.BaiscDataset(client_train_data[i], transform=train_transform)
        this_validation_set = data_utils.BaiscDataset(client_validation_data[i], transform=test_transform)
        this_test_set = data_utils.BaiscDataset(client_test_data[i], transform=test_transform)

        train_loaders.append(torch.utils.data.DataLoader(this_train_set, batch_size=args.batch, shuffle=True,
                                                         drop_last=True, num_workers=3))
        validation_loaders.append(
            torch.utils.data.DataLoader(this_validation_set, batch_size=args.batch * 5, shuffle=False))
        test_loaders.append(torch.utils.data.DataLoader(this_test_set, batch_size=args.batch * 5, shuffle=False,
                                                        num_workers=2))

    return train_loaders, validation_loaders, test_loaders, prior_test, client_priors_corr, client_Pi


def L1_Regularization(model):
    L1_reg = 0
    for param in model.parameters():
        L1_reg += torch.sum(torch.abs(param))

    return L1_reg


def train(args, model, train_loader, optimizer, loss_fun, client_num, device, Pi, priors_corr, prior_test):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    noises = {}
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        priors_corr = priors_corr.to(device).float()
        Pi = Pi.to(device).float()

        output = model(x, Pi, priors_corr, prior_test)

        loss = loss_fun(output, y) + L1_Regularization(model) * args.wdecay
        loss.backward()
        loss_all += loss.item()

        optimizer.step()

    return loss_all / len(train_iter)


def test(model, test_loader, loss_fun, device, classnum=10):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    noises = {}

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).float()
            target = target.to(device).long()

            output = model.predict(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
            total += target.size(0)

    test_error = (total - correct) / total

    return test_error


def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Device:', device, '\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=0., help='learning rate')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--iters', type=int, default=100, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='fedavg', help='fedavg')
    parser.add_argument('--save_path', type=str, default='../checkpoint/cifar', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')

    parser.add_argument('--clientnum', type=int, default=5, help='client number')
    parser.add_argument('--setnum', type=int, default=10, help='set number per client has')

    parser.add_argument('--classnum', type=int, default=10, help='class num')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--noniid', action='store_true', help='noniid sampling')

    args = parser.parse_args()
    print(args)

    setup_seed(args.seed)

    exp_folder = 'cifar_fedul'

    args.save_path = os.path.join(args.save_path, exp_folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path,
                             args.mode + 'client' + str(args.clientnum) + 'sets' + str(args.setnum) + 'seed' + str(
                                 args.seed) + str(args.noniid))

    # server model and ce loss
    server_model = CIFARModel(class_num=args.classnum).to(device)
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders, validation_loaders, test_loaders, prior_test, client_priors_corr, client_Pi = prepare_data(args)

    print('\nData prepared, start training...\n')

    # federated setting
    client_num = args.clientnum
    clients = ['client' + str(_) for _ in range(1, client_num + 1)]
    client_weights = [1 / client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        this_test_error = []
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss = test(server_model, test_loader, loss_fun, device, classnum=args.classnum)
            this_test_error.append(test_loss)
            print(' {:<8s}| Error Rate: {:.2f} %'.format(clients[test_idx], test_loss * 100.))
        print('Best Test Error: {:.2f} %'.format(100. * sum(this_test_error) / len(this_test_error)))

        exit(0)

    best_test_error = 1.
    training_loss_log = []
    error_rate_log = []

    # start training
    for a_iter in range(args.iters):
        # record training loss and test error rate
        this_test_error = []
        this_train_loss = []

        optimizers = [optim.Adam(params=models[idx].parameters(), lr=args.lr)
                      for idx in range(client_num)]

        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + 1 + a_iter * args.wk_iters))

            for client_idx in range(client_num):
                model, train_loader, optimizer = models[client_idx], train_loaders[client_idx], optimizers[client_idx]
                train_loss = train(args, model, train_loader, optimizer, loss_fun, client_num, device,
                                   client_Pi[client_idx], client_priors_corr[client_idx], prior_test)
                print(' {:<8s}| Train Loss: {:.4f}'.format(clients[client_idx], train_loss))

                this_train_loss.append(train_loss)

        # aggregation
        server_model, models = communication(args, server_model, models, client_weights)

        # start testing
        for test_idx, test_loader in enumerate(validation_loaders):
            test_loss = test(models[test_idx], test_loader, loss_fun, device, classnum=args.classnum)
            this_test_error.append(test_loss)
            print(' {:<8s}| Error Rate: {:.2f} %'.format(clients[test_idx], test_loss * 100.))

        print()

        # error rate after this communication
        this_test_error = sum(this_test_error) / len(this_test_error)
        if this_test_error < best_test_error:
            best_test_error = this_test_error

            # Save checkpoint
            print(' Saving checkpoints to {}'.format(SAVE_PATH))
            torch.save({
                'server_model': server_model.state_dict(),
                'a_iter': a_iter,
            }, SAVE_PATH)

        # Best Validation Error Rate
        print(' Best Validation Error Rate: {:.2f} %, Current Validation Error Rate: {:.2f} %\n'.format(
            best_test_error * 100.,
            this_test_error * 100.
        ))

        training_loss_log.append(sum(this_train_loss) / len(this_train_loss))
        error_rate_log.append(this_test_error)

        if not os.path.exists(os.path.join('../logs/cifar_fedul', args.mode)):
            os.makedirs(os.path.join('../logs/cifar_fedul', args.mode))

    print(' Start final testing\n')
    checkpoint = torch.load(SAVE_PATH)
    server_model.load_state_dict(checkpoint['server_model'])
    this_test_error = []
    for test_idx, test_loader in enumerate(test_loaders):
        test_loss = test(server_model, test_loader, loss_fun, device, classnum=args.classnum)
        this_test_error.append(test_loss)
        print(' {:<8s}| Error Rate: {:.2f} %'.format(clients[test_idx], test_loss * 100.))
    print(' Best Test Error: {:.2f} %'.format(100. * sum(this_test_error) / len(this_test_error)))

    error_rate_log.append(sum(this_test_error) / len(this_test_error))
    # save record
    np.savetxt(os.path.join('../logs/cifar_fedul', args.mode, 'client' + str(args.clientnum) +
                            'sets' + str(args.setnum) + 'seed' + str(args.seed) + str(args.noniid) + 'train_loss.txt'),
               training_loss_log, newline="\r\n")
    np.savetxt(os.path.join('../logs/cifar_fedul', args.mode, 'client' + str(args.clientnum) +
                            'sets' + str(args.setnum) + 'seed' + str(args.seed) + str(args.noniid) + 'error_rate.txt'),
               error_rate_log, newline="\r\n")

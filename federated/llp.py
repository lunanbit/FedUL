import torch
from torch.utils.data import Sampler


class BagMiniBatch:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.reset()

    def reset(self):
        self.bags = []
        self.bag_sizes = []
        self.targets = []  # store proportion labels

    def append(self, x, y):
        assert x.size(0) == y.size(0)
        self.targets.append(torch.mean(y, dim=0))
        if self.n_samples > 0:
            index = torch.randperm(x.size(0))[:self.n_samples]
            x = x[index]
            y = y[index]
        self.bags.append((x, y))
        self.bag_sizes.append(y.size(0))

    def __iter__(self):
        for item in zip(self.bag_sizes, self.targets):
            yield item

    @property
    def total_size(self):
        return sum(self.bag_sizes)

    @property
    def max_bag_size(self):
        return max(self.bag_sizes)

    @property
    def num_bags(self):
        return len(self.bag_sizes)


class BagSampler(Sampler):
    def __init__(self, bags, num_bags=-1):
        """
        params:
            bags: shape (num_bags, num_instances), the element of a bag
                  is the instance index of the dataset
            num_bags: int, -1 stands for using all bags
        """
        self.bags = bags
        if num_bags == -1:
            self.num_bags = len(bags)
        else:
            self.num_bags = num_bags
        assert 0 < self.num_bags <= len(bags)

    def __iter__(self):
        indices = torch.randperm(self.num_bags)
        for index in indices:
            yield self.bags[index]

    def __len__(self):
        return len(self.bags)

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision


class DigitModel(nn.Module):
    def __init__(self, class_num=10):
        super(DigitModel, self).__init__()
        self.class_num = class_num

        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, class_num)

    def forward(self, x, Pi, priors_corr, prior_test):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)

        g = torch.softmax(x, dim=1)
        x = self.QfunctionMulticlass(g, Pi, priors_corr)

        return x

    def QfunctionMulticlass(self, g, Pi, priors_corr):
        pi_ita = torch.mm(Pi, g.permute(1, 0))
        rou_pi_ita = torch.matmul(priors_corr, pi_ita)

        pi_corr = pi_ita.permute(1, 0) * priors_corr.unsqueeze(0)
        output = (pi_corr.permute(1, 0) / rou_pi_ita).permute(1, 0)

        return output

    def predict(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)

        g = torch.softmax(x, dim=1)

        return g


class ResNetFc(nn.Module):
    def __init__(self):
        super(ResNetFc, self).__init__()
        model_resnet = torchvision.models.resnet18(pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu

        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        self.avgpool = model_resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class CIFARModel(nn.Module):
    def __init__(self, class_num=10):
        super(CIFARModel, self).__init__()
        self.class_num = class_num

        self.backbone = ResNetFc()

        self.clf = nn.Linear(512, class_num)

    def forward(self, x, Pi, priors_corr, prior_test):
        x = self.backbone(x)
        x = self.clf(x)

        g = torch.softmax(x, dim=1)
        x = self.QfunctionMulticlass(g, Pi, priors_corr)

        return x

    def QfunctionMulticlass(self, g, Pi, priors_corr):
        pi_ita = torch.mm(Pi, g.permute(1, 0))
        rou_pi_ita = torch.matmul(priors_corr, pi_ita)

        pi_corr = pi_ita.permute(1, 0) * priors_corr.unsqueeze(0)
        output = (pi_corr.permute(1, 0) / rou_pi_ita).permute(1, 0)

        return output

    def predict(self, x):
        x = self.backbone(x)
        x = self.clf(x)

        g = torch.softmax(x, dim=1)

        return g

    def server_forward(self, x):
        x = self.backbone(x)
        x = self.clf(x)

        return x


class PLMNISTModel(nn.Module):
    def __init__(self, class_num=10):
        super(PLMNISTModel, self).__init__()
        self.class_num = class_num

        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, class_num)

    def forward(self, x, Pi, priors_corr, prior_test):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)

        return x

    def predict(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)

        x = torch.softmax(x, dim=1)

        return x


class PLCIFARModel(nn.Module):
    def __init__(self, class_num=10):
        super(PLCIFARModel, self).__init__()
        self.class_num = class_num

        self.backbone = ResNetFc()

        self.clf = nn.Linear(512, class_num)

    def forward(self, x, Pi, priors_corr, prior_test):
        x = self.backbone(x)
        x = self.clf(x)

        return x

    def predict(self, x):
        x = self.backbone(x)
        x = self.clf(x)

        g = torch.softmax(x, dim=1)

        return g

    def server_forward(self, x):
        x = self.backbone(x)
        x = self.clf(x)

        return x


class LLPMNISTModel(nn.Module):
    def __init__(self, class_num=10):
        super(LLPMNISTModel, self).__init__()
        self.class_num = class_num

        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(512, class_num)

    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)
        x = self.dropout(x)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)
        x = self.dropout(x)

        x = func.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x

    def predict(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)

        x = torch.softmax(x, dim=1)

        return x


class LLPResNetFc(nn.Module):
    def __init__(self):
        super(LLPResNetFc, self).__init__()
        model_resnet = torchvision.models.resnet18(pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu

        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        self.avgpool = model_resnet.avgpool

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def predict(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class LLPCIFARModel(nn.Module):
    def __init__(self, class_num=10):
        super(LLPCIFARModel, self).__init__()
        self.class_num = class_num

        self.backbone = LLPResNetFc()

        self.clf = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.backbone(x)
        x = self.clf(x)

        return x

    def predict(self, x):
        x = self.backbone.predict(x)
        x = self.clf(x)

        g = torch.softmax(x, dim=1)

        return g


class UpperDigitModel(nn.Module):
    def __init__(self, class_num=10):
        super(UpperDigitModel, self).__init__()
        self.class_num = class_num

        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, class_num)

    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)

        x = torch.softmax(x, dim=1)

        return x

    def predict(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)

        g = torch.softmax(x, dim=1)

        return g


if __name__ == '__main__':
    model = CIFARModel()
    print(model)

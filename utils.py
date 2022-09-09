import os
import random
import pickle
from typing import Optional, Sequence

import torch
import torchvision
import numpy as np
import scipy.special
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from collections import namedtuple
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
from tqdm import tqdm

import models.resnet_cifar10 as resnet
import models.wide_resnet as wrn
import models.densenet as dense_net


""" Setup """
def seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

"""Augmentation tool"""
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

""" Dataset Transforms """
# cifar 10
cifar10_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    Cutout(n_holes=1, length=6)
])

cifar10_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# cifar 100
cifar100_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2675, 0.2565, 0.2761)),
    Cutout(n_holes=1, length=4)
])

cifar100_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2675, 0.2565, 0.2761)),
])

# ImageNet
imagenet_transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Cutout(n_holes=1, length=32)
        ])

imagenet_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

""" Dataset """
def load_dataset(data, split, batch_size=1000, image_size=224, aug=True):
    """
    :param data:
    :param split:
    :param batch_size:
    :param image_size: only used for ImageNet
    :param aug: data augmentation
    :return:
    """
    if data == 'CIFAR-10':
        if split == 'train':
            return get_cifar10_train_loader(batch_size=batch_size, aug=aug)
        elif split == 'val':
            return get_cifar10_val_loader(batch_size=batch_size, aug=aug)
        elif split == 'test':
            return get_cifar10_test_loader(batch_size=batch_size)
    elif data == 'CIFAR-100':
        if split == 'train':
            return get_cifar100_train_loader(batch_size=batch_size, aug=aug)
        elif split == 'val':
            return get_cifar100_val_loader(batch_size=batch_size, aug=aug)
        elif split == 'test':
            return get_cifar100_test_loader(batch_size=batch_size)
    elif data == 'ImageNet':
        if split == 'val':
            return get_imagenet_val_loader(batch_size=batch_size, aug=aug)
        elif split == 'test':
            return get_imagenet_test_loader(batch_size=batch_size)

# cifar10 train dataset
def get_cifar10_train_loader(batch_size, aug=True):
    if aug is True:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=cifar10_transform_train)
    else:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=cifar10_transform_test)
    trainset.data = trainset.data[:45000]
    trainset.targets = trainset.targets[:45000]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    return trainloader

# cifar10 validation dataset
def get_cifar10_val_loader(batch_size, aug=True):
    if aug is True:
        valset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=cifar10_transform_train)
    else:
        valset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=cifar10_transform_test)
    valset.data = valset.data[45000:]
    valset.targets = valset.targets[45000:]
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8)
    return valloader

# cifar10 test dataset
def get_cifar10_test_loader(batch_size):
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=cifar10_transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return testloader

# cifar100 train dataset
def get_cifar100_train_loader(batch_size, aug=True):
    if aug is True:
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=cifar100_transform_train)
    else:
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=cifar100_transform_test)
    trainset.data = trainset.data[:45000]
    trainset.targets = trainset.targets[:45000]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    return trainloader

# cifar 100 validation dataset
def get_cifar100_val_loader(batch_size, aug=True):
    if aug is True:
        valset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=cifar100_transform_train)
    else:
        valset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=cifar100_transform_test)
    valset.data = valset.data[45000:]
    valset.targets = valset.targets[45000:]
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8)
    return valloader

# cifar100 test dataset
def get_cifar100_test_loader(batch_size):
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=cifar100_transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return testloader

# imagenet dataset: 25k val images
def get_imagenet_val_loader(batch_size, aug=True):
    if aug is True:
        dataset = ImageFolder(root=os.path.join('./data/imagenet', 'val'),
                              transform=imagenet_transform_train)
    else:
        dataset = ImageFolder(root=os.path.join('./data/imagenet', 'val'),
                              transform=imagenet_transform_test)
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# imagenet dataset: 25k test images
def get_imagenet_test_loader(batch_size):
    dataset = ImageFolder(root=os.path.join('./data/imagenet', 'test'),
                          transform=imagenet_transform_test)
    return torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)



""" Metric """
# entropy
class Entropy(nn.Module):
    """
    Calbulates the Entropy of model prediction from logits

    """
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, logits):
        ent = F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)
        return -1.0 * ent.sum() / logits.size()[0]

# Expected Calibration Error
class ECE(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class SCE(nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(SCE, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce

class AdaptiveECE(nn.Module):
    """Compute Adaptive ECE"""
    def __init__(self, n_bins=15):
        super(AdaptiveECE, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


""" Testing Tool """
def test_model(model, loader, n_bins=15, temp=None, device_ids=[]):
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    model.cuda()
    ece_criterion = ECE(n_bins=n_bins).cuda()
    ent_criterion = Entropy().cuda()
    sce_criterion = SCE(n_bins=n_bins).cuda()
    aece_criterion = AdaptiveECE(n_bins=n_bins).cuda()

    # collect all the logits and labels for the test data set
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for data in tqdm(loader):
            # load input datas and its labels
            inputs, labels = data[0].cuda(), data[1].cuda()

            # inference
            logits = model(inputs)

            logits_list.append(logits)
            labels_list.append(labels)

        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

    # Calculate ECE on test data set
    _, predicted = torch.max(logits.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    acc = correct / total
    # print(correct, total)
    print('- Accuracy: %f' % acc)
    before_temperature_ece = ece_criterion(logits, labels).item()
    before_temperature_ent = ent_criterion(logits).item()
    before_sce_result = sce_criterion(logits, labels).item()
    before_aece_result = aece_criterion(logits, labels).item()
    print('- ECE: %f' % before_temperature_ece)
    print('- Entropy: %f' % before_temperature_ent)
    print('- SCE: %f' % before_sce_result)
    print('- Adaptive ECE: %f' % before_aece_result)

    if temp is not None:
        after_temperature_ece = ece_criterion(logits/temp, labels).item()
        after_temperature_ent = ent_criterion(logits/temp).item()
        after_sce_result = sce_criterion(logits/temp, labels).item()
        after_aece_result = aece_criterion(logits/temp, labels).item()
        print('After Temperature Scaling')
        print('- ECE: %f' % after_temperature_ece)
        print('- Entropy: %f' % after_temperature_ent)
        print('- SCE: %f' % after_sce_result)
        print('- Adaptive ECE: %f' % after_aece_result)

def get_logits_and_labels(model, loader):
    model.eval()
    model.cuda()
    # collect all the logits and labels for the test data set
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for data in loader:
            # load input datas and its labels
            inputs, labels = data[0].cuda(), data[1].cuda()

            # inference
            logits = model(inputs)

            # list appned
            logits_list.append(logits)
            labels_list.append(labels)
        # concat
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

    return logits, labels

def accuracy(out, target):
    return out.max(dim=1)[1] == target


""" Model """
def model_classes(data):
    if data == 'CIFAR-10':
        return 10
    elif data == 'CIFAR-100':
        return 100
    elif 'ImageNet':
        return 1000

def load_model(name, data, pretrained=True):
    if name == 'ResNet-110' and data == 'CIFAR-100':
        return get_resnet110_cifar100(pretrained=pretrained)
    elif name == 'ResNet-110' and data == 'CIFAR-10':
        return get_resnet110_cifar10()
    elif name == 'WideResNet-40-10' and data == 'CIFAR-100':
        return get_wide_resnet_40_10_cifar100()
    elif name == 'DenseNet-121' and data == 'CIFAR-100':
        return get_densenet_121_cifar100()
    elif name == 'ResNet-101' and data == 'ImageNet':
        return get_resnet101_imagenet()
    elif name == 'ViT-B16' and data == 'ImageNet':
        return get_vit_b16_imagenet()
    elif name == 'ConvNeXt-T' and data == 'ImageNet':
        return get_convnext_tiny_imagenet()
    elif name == 'EfficientNet-B0' and data == 'ImageNet':
        return get_efficientnet_b0_imagenet()
    elif name == 'WideResNet-50-2' and data == 'ImageNet':
        return get_wide_resnet50_2_imagent()
    elif name == 'RegNetY-400MF' and data == 'ImageNet':
        return get_regnet_y_400mf()


def get_resnet110_cifar100(pretrained=True):
    if pretrained is True:
        model = resnet.resnet110(num_classes=100)
        model_path = './saved_models/resnet110-cifar100-7415.pt'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        print('load model successfully')
    else:
        model = resnet.resnet110(num_classes=100)
    return model

def get_resnet110_cifar10():
    model = resnet.resnet110(num_classes=10)
    model_path = 'saved_models/resnet110-cifar10-9325.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print('load model successfully')
    return model

def get_wide_resnet_40_10_cifar100():
    model = wrn.wideresnet()
    model_path = 'saved_models/wide_resnet_40_10.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print('load model successfully')
    return model

def get_densenet_121_cifar100():
    model = dense_net.densenet121()
    model_path = 'saved_models/densenet121.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print('load model successfully')
    return model

def get_resnet101_imagenet():
    model = models.resnet101(pretrained=True)
    print('load model successfully')
    return model

def get_vit_b16_imagenet():
    model = models.vit_b_16(pretrained=True)
    print('load model successfully')
    return model

def get_convnext_tiny_imagenet():
    model = models.convnext_tiny(pretrained=True)
    print('load model successfully')
    return model

def get_efficientnet_b0_imagenet():
    model = models.efficientnet_b0(pretrained=True)
    print('load model successfully')
    return model

def get_wide_resnet50_2_imagent():
    model = models.wide_resnet50_2(pretrained=True)
    print('load model successfully')
    return model

def get_regnet_y_400mf():
    model = models.regnet_y_400mf(pretrained=True)
    print('load model successfully')
    return model

""" Loss Function """
class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num=10, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        epsilon = 1.e-7
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)
        probs = torch.clamp(probs, max=1-epsilon, min=epsilon)
        log_p = probs.log()

        batch_loss = -alpha*((1-probs)**self.gamma)*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

""" Tool """
def save_logits_labels(model, loader, model_name, data_name, split):
    logits, labels = get_logits_and_labels(model=model, loader=loader)
    print("Saving logits")
    name = model_name + '_' + data_name + '_' + split
    with open("logits_%s.p" % name, "wb") as f:
        pickle.dump((logits.cpu().numpy(), labels.cpu().numpy()), f)
    print("Done")

def unpickle_probs(fname):
    # Read and open the file
    with open(fname, 'rb') as f:
        logits, labels = pickle.load(f)
    probs = scipy.special.softmax(logits, 1)
    return probs, labels


import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable

from utils import ECE, Entropy, SCE, AdaptiveECE, FocalLoss


class ModelWithNeuralClamping(nn.Module):
    """
        A decorator, which wraps a model with neural clamping
        Neural Clamping: Joint Input Perturbation and Temperature Scaling for Neural Network Calibration

        :param model: A classification neural network. (nn.Module)
                      Output of the neural network should be the classification logits
        :param lambda_0: hyperparameter of loss function [default=1]
        :param lambda_1: hyperparameter of loss function [default=1]
        :param image_size: input image size [default=(3, 32, 32)]
        :param init_scale: initial scale for input perturbation [float, default=0.01]
        :param dropout: use dropout on input perturbation or not during training process [float, default=0]
        :param init_temp: initial value for output temperature [float, default=1]
        :param num_classes: number of classes [int, default=10]
        :param device_ids: if use multi gpu, hint the wrapper with gpu id list.

        Loss Function:
            lambda_0 * NLL/FocalLoss + lambda_1 * regularization
    """
    def __init__(self, model,
                 lambda_0=1, lambda_1=1, image_size=(3, 32, 32), init_scale=0.01, dropout_p=0.0, init_temp=1,
                 num_classes=10, device_ids=[]):
        super(ModelWithNeuralClamping, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=dropout_p)
        self.perturbation = nn.Parameter(torch.randn(size=image_size)*init_scale, requires_grad=True)
        self.perturbation.requires_grad = True
        self.temperature = torch.tensor(torch.ones(1) * init_temp)
        self.temperature = nn.Parameter(self.temperature.requires_grad_())
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.num_classes = num_classes
        self.device_ids = device_ids
        if len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model.cuda()
        self.perturbation.cuda()
        self.temperature.cuda()

    def forward(self, input):
        """
        :param input: input tensor
        :return: logits that inference with adversarial parameter
        """
        logits = self.model(input + self.perturbation)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        return logits / self.temperature

    def train_NC(self, val_loader, epoch, lr=0.001, momentum=0.9, nesterov=False, verbose=True,
                 focal_loss=False, gamma=1, adamW=False):
        """
        Tune the Neural Clamping of the model (using the validation set).
        We're going to set it to optimize NLL/FocalLoss + Reg.
        :param val_loader: validation set loader [DataLoader]
        :param epoch: training epoch [int]
        :param lr: learning rate for optimize adversarial parameter [float]
        :param momentum: momentum value for SGD optimizer [float]
        :param nesterov: nesterov or not [bool]
        """
        self.model.eval()
        self.cuda()
        self.perturbation.cuda()
        self.perturbation.requires_grad = True

        nll_criterion = nn.CrossEntropyLoss().cuda()
        if focal_loss is True:
            nll_criterion = FocalLoss(class_num=self.num_classes, gamma=gamma).cuda()
        ece_criterion = ECE().cuda()
        ent_criterion = Entropy().cuda()
        optimizer = optim.SGD([self.perturbation, self.temperature], lr=lr, momentum=momentum, nesterov=nesterov)
        if adamW is True:
            optimizer = optim.AdamW([self.perturbation, self.temperature], lr=lr)

        loss_list = []
        nll_list = []
        reg_list = []
        ece_list = []
        ent_list = []
        temp_list = []

        for i in range(epoch):
            if verbose is False:
                print('Training Epochs:', i + 1, '/', epoch)

            optimizer.zero_grad()

            logits_list = []
            labels_list = []
            batch_loss = []
            batch_nll = []
            batch_reg = []

            for data in val_loader:
                inputs, labels = data[0].cuda(), data[1].cuda()

                # forward pass
                logits = self.temperature_scale(self.model(inputs + self.dropout(self.perturbation)))

                with torch.no_grad():
                    logits_list.append(logits)
                    labels_list.append(labels)

                # loss
                nll_loss = nll_criterion(logits, labels)
                batch_nll.append(nll_loss.item())

                reg_loss = torch.norm(self.perturbation)
                batch_reg.append(reg_loss.item())

                loss = self.lambda_0 * nll_loss + self.lambda_1 * reg_loss
                batch_loss.append(loss.item())

                # back prop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                # these used for calculating ece
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda()
                # ece
                ece = ece_criterion(logits, labels)
                # entropy
                ent = ent_criterion(logits)
                # append information
                loss_list.append(np.mean(batch_loss))
                nll_list.append(np.mean(batch_nll))
                reg_list.append(np.mean(batch_reg))
                ece_list.append(ece.item())
                ent_list.append(ent.item())
                temp_list.append(self.temperature.item())

            if verbose is True:
                print(i+1, '/', epoch, 'Training')
                print('Loss: %.4f' % np.mean(batch_loss))
                print('-NLL: %.4f' % np.mean(batch_nll))
                print('-REG: %.4f' % np.mean(batch_reg))
                print('-ECE: %.4f' % ece)
                print('-Entropy: %.4f' % ent.item())
                print('-TEMP: %.4f' % self.temperature.item())

        return {"loss": loss_list, "nll": nll_list, "reg": reg_list,
                "ece": ece_list, 'ent': ent_list, "temp": temp_list}

    def test_with_NC(self, test_loader, n_bins=15, return_result=False, focal_loss=False, gamma=0):
        self.model.eval()
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        if focal_loss is True:
            nll_criterion = FocalLoss(class_num=self.num_classes, gamma=gamma).cuda()
        ece_criterion = ECE(n_bins=n_bins).cuda()
        ent_criterion = Entropy().cuda()
        sce_criterion = SCE(n_bins=n_bins).cuda()
        aece_criterion = AdaptiveECE(n_bins=n_bins).cuda()

        # collect all the logits and labels for the test data set
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data in test_loader:
                # load input datas and its labels
                inputs, labels = data[0].cuda(), data[1].cuda()

                # inference
                logits = self.temperature_scale(self.model(inputs + self.perturbation))

                logits_list.append(logits)
                labels_list.append(labels)

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate ECE on test data set
        after_loss = nll_criterion(logits, labels).item()
        after_adv_ece = ece_criterion(logits, labels).item()
        after_adv_ent = ent_criterion(logits).item()
        print('Below information is based on testing data set')
        _, predicted = torch.max(logits.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        acc = correct / total
        print('Accuracy: %.4f' % acc)
        print('Loss: ', after_loss)
        print('ECE: %.4f' % after_adv_ece)
        print('Entropy: %.4f' % after_adv_ent)
        sce_result = sce_criterion(logits, labels).item()
        print('SCE: %.4f' % sce_result)
        aece_result = aece_criterion(logits, labels).item()
        print('Adaptive ECE: %.4f' % aece_result)
        if return_result is True:
            return after_adv_ent, after_adv_ece, aece_result, sce_result, after_loss

    def test_ece_with_only_perturbation(self, test_loader, n_bins=15, return_result=False, focal_loss=False, gamma=0):
        self.model.eval()
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        if focal_loss is True:
            nll_criterion = FocalLoss(class_num=self.num_classes, gamma=gamma).cuda()
        ece_criterion = ECE(n_bins=n_bins).cuda()
        ent_criterion = Entropy().cuda()
        sce_criterion = SCE(n_bins=n_bins).cuda()
        aece_criterion = AdaptiveECE(n_bins=n_bins).cuda()

        # collect all the logits and labels for the test data set
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data in test_loader:
                # load input datas and its labels
                inputs, labels = data[0].cuda(), data[1].cuda()

                # inference
                logits = self.model(inputs + self.perturbation)

                logits_list.append(logits)
                labels_list.append(labels)

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate ECE on test data set
        after_loss = nll_criterion(logits, labels).item()
        after_adv_ece = ece_criterion(logits, labels).item()
        after_adv_ent = ent_criterion(logits).item()
        print('Below information is based on testing data set')
        _, predicted = torch.max(logits.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        acc = correct / total
        print('Accuracy: %.4f' % acc)
        print('Loss: ', after_loss)
        print('ECE w/ adv. parameter: %.4f' % after_adv_ece)
        print('Entropy w/ adv. parameter: %.4f' % after_adv_ent)
        sce_result = sce_criterion(logits, labels).item()
        print('SCE w/ adv. parameter: %.4f' % sce_result)
        aece_result = aece_criterion(logits, labels).item()
        print('Adaptive ECE w/ adv. parameter: %.4f' % aece_result)
        if return_result is True:
            return after_adv_ent, after_adv_ece, aece_result, sce_result, after_loss

    def test_ece_with_only_temperature(self, test_loader, n_bins=15, return_result=False, focal_loss=False, gamma=0):
        self.model.eval()
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        if focal_loss is True:
            nll_criterion = FocalLoss(class_num=self.num_classes, gamma=gamma).cuda()
        ece_criterion = ECE(n_bins=n_bins).cuda()
        ent_criterion = Entropy().cuda()
        sce_criterion = SCE(n_bins=n_bins).cuda()
        aece_criterion = AdaptiveECE(n_bins=n_bins).cuda()

        # collect all the logits and labels for the test data set
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data in test_loader:
                # load input datas and its labels
                inputs, labels = data[0].cuda(), data[1].cuda()

                # inference
                logits = self.temperature_scale(self.model(inputs))

                logits_list.append(logits)
                labels_list.append(labels)

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate ECE on test data set
        after_loss = nll_criterion(logits, labels).item()
        after_adv_ece = ece_criterion(logits, labels).item()
        after_adv_ent = ent_criterion(logits).item()
        print('Below information is based on testing data set')
        _, predicted = torch.max(logits.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        acc = correct / total
        print('Accuracy: %.4f' % acc)
        print('Loss: ', after_loss)
        print('ECE w/ adv. parameter: %.4f' % after_adv_ece)
        print('Entropy w/ adv. parameter: %.4f' % after_adv_ent)
        sce_result = sce_criterion(logits, labels).item()
        print('SCE w/ adv. parameter: %.4f' % sce_result)
        aece_result = aece_criterion(logits, labels).item()
        print('Adaptive ECE w/ adv. parameter: %.4f' % aece_result)
        if return_result is True:
            return after_adv_ent, after_adv_ece, aece_result, sce_result, after_loss

    def set_delta_init(self, val_loader):
        self.model.eval()
        self.cuda()
        self.perturbation.cuda()
        self.perturbation.requires_grad = True

        grad_list = []

        for data in val_loader:
            # load input datas and its labels
            inputs, labels = data[0].cuda(), data[1].cuda()
            inputs = Variable(inputs, requires_grad=True)

            # inference
            logits = self.temperature_scale(self.model(inputs + self.perturbation))
            # loss = nll_criterion(logits, labels).cuda()
            grad = torch.autograd.grad(outputs=logits, inputs=inputs, grad_outputs=torch.ones_like(logits).cuda())
            grad = grad[0]
            grad_list.append(grad)

        grad = torch.cat(grad_list)
        grad = torch.mean(grad, dim=0)
        self.perturbation = nn.Parameter(grad, requires_grad=True)
        print('set delta initial value successfully')
        return None


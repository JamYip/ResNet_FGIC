import torch
import numpy as np
from utils import *


def train(train_loader, model, criterion, optimizer):
    train_loss = 0
    correct = 0
    total = 0
    idx = 0
    model.train()
    for i, data in enumerate(train_loader):
        idx = i
        img, label = data
        input = img.to(device)
        label = torch.Tensor(label).type(torch.int64).to(device)

        output = model(input)
        loss = criterion(output, label)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data).cpu().sum()
        if i % 50 == 0:

            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                i, train_loss / (i + 1), 100. * float(correct) / total, correct, total))

    train_acc = 100. * float(correct) / total
    train_loss = train_loss / (idx + 1)

    return train_acc, train_loss


def test(test_loader, model, criterion):
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    model.eval()
    for i, data in enumerate(test_loader):
        idx = i
        img, label = data
        input = img.to(device)
        label = label.to(device)

        output = model(input)
        loss = criterion(output, label)

        test_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += predicted.eq(label.data).cpu().sum()

        if i % 50 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            i, test_loss / (i + 1), 100. * float(correct) / total, correct, total))

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_loss

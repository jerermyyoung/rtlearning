# training on Clothing1M

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
from model import *
from data import *
import torch.nn.functional as F
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int)
args = parser.parse_args()
# torch.cuda.set_device(args.device)
batch_size = 256
# batch_size = 32
num_classes = 14

CE = nn.CrossEntropyLoss().cuda(device=args.device)

data_root = '/data/clothing1m/'
train_dataset = Clothing(root=data_root, img_transform=train_transform, train=True, valid=False, test=False)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 32)
valid_dataset = Clothing(root=data_root, img_transform=test_transform, train=False, valid=True, test=False)
valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False, num_workers = 32)
test_dataset = Clothing(root=data_root, img_transform=test_transform, train=False, valid=False, test=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 32)


def train(train_loader, model, optimizer, criterion=CE):
    model.train()
    total_loss = 0.0

    # for i, (idx, input, target) in enumerate(tqdm(train_loader)):
    for i, (idx, input, target) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            break
        input = torch.autograd.Variable(input.cuda(device=args.device))
        target = torch.autograd.Variable(target.cuda(device=args.device))

        output = model(input)
        loss = criterion(output, target)
        total_loss = (i*total_loss+loss.detach().item())/(i+1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    # for i, (idx, input, target) in enumerate(tqdm(test_loader)):
    for i, (idx, input, target) in enumerate(test_loader):
        input = torch.Tensor(input).cuda(device=args.device)
        target = torch.autograd.Variable(target).cuda(device=args.device)

        total += target.size(0)
        output = model(input)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    return accuracy


def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1)
    y_onehot = torch.FloatTensor(target.size(0), num_classes)
    y_onehot.zero_()
    targets = targets.cpu()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda(device=args.device)
    mat = y_onehot @ outputs
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)


def main_ce():
    # model_ce = resnet50().cuda(device=args.device)
    model_ce = resnet50(pretrained=True)
    # model_ce = torch.nn.DataParallel(model_ce, device_ids=[args.device,args.device+1,args.device+2,args.device+3,args.device+4,args.device+5])
    model_ce = torch.nn.DataParallel(model_ce)
    best_ce_acc = 0

    for epoch in range(10):
        print("epoch=", epoch)
        learning_rate = 0.01
        if epoch >= 5:
            learning_rate = 0.001

        optimizer_ce = torch.optim.SGD(model_ce.parameters(), momentum=0.9, weight_decay=1e-3, lr=learning_rate)

        print("traning model_ce...")
        train_loss = train(train_loader=train_loader, model=model_ce, optimizer=optimizer_ce)
        print('train_loss =', train_loss)
        print("validating model_ce...")
        valid_acc = test(model=model_ce, test_loader=valid_loader)
        print('valid_acc =', valid_acc)
        if valid_acc > best_ce_acc:
            best_ce_acc = valid_acc
            torch.save(model_ce, './model_ce')

    model_ce = torch.load('./model_ce')
    test_acc = test(model=model_ce, test_loader=test_loader)
    print('model_ce_final_test_acc=', test_acc)

def main_dmi():
    # model_dmi = torch.load('./model_ce')
    model_dmi = resnet50().cuda(device=args.device)
    model_dmi = torch.nn.DataParallel(model_dmi, device_ids=[args.device,args.device+1,args.device+2,args.device+3,args.device+4,args.device+5])
    model_dict = torch.load('./model_ce').module.state_dict()
    model_dmi.module.load_state_dict(model_dict)
    # model_dmi.to(args.device)
    best_acc = 0

    for epoch in range(10):
        print("epoch=", epoch)
        learning_rate = 1e-6
        if epoch >= 5:
            learning_rate = 5e-7

        optimizer_dmi = torch.optim.SGD(model_dmi.parameters(), momentum=0.9, weight_decay=1e-3, lr=learning_rate)

        print("traning model_dmi...")
        train(train_loader=train_loader, model=model_dmi, optimizer=optimizer_dmi, criterion=DMI_loss)
        print("validating model_dmi...")
        valid_acc = test(model=model_dmi, test_loader=valid_loader)
        print('valid_acc=', valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model_dmi, './model_dmi')

    model_ce = torch.load('./model_dmi')
    test_acc = test(model=model_ce, test_loader=test_loader)
    print('model_dmi_final_test_acc=', test_acc)


if __name__ == '__main__':
    main_ce()
    # main_dmi()

# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import MLP
import argparse, sys
import numpy as np
import datetime
import shutil

from loss import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric]', default='pairflip')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--n_iter', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=200)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

# load dataset
num_classes=10
args.epoch_decay_start = 200
args.n_epoch = 200
train_dataset = MNIST(root='./data/',
                            download=True,  
                            train=True, 
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                     )
    
test_dataset = MNIST(root='./data/',
                           download=True,  
                           train=False, 
                           transform=transforms.ToTensor(),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate
                    )
    
if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan=np.ones(args.n_epoch,dtype=float)*learning_rate
'''
alpha_plan[:int(args.n_epoch*0.5)] = [learning_rate] * int(args.n_epoch*0.5)
alpha_plan[int(args.n_epoch*0.5):int(args.n_epoch*0.75)] = [learning_rate*0.1] * int(args.n_epoch*0.25)
alpha_plan[int(args.n_epoch*0.75):] = [learning_rate*0.01] * int(args.n_epoch*0.25)
'''
alpha_plan[80:]=learning_rate-learning_rate*np.arange(args.n_epoch-80,dtype=float)/(args.n_epoch-80)
beta1_plan = [mom1] * args.n_epoch
beta1_plan[80:]=[mom2] * (args.n_epoch-80)
# print(alpha_plan,beta1_plan)

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['momentum']=beta1_plan[epoch] # Only change beta1
        
save_dir = args.result_dir +'/mnist/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
model_str='mnist_rand_coteaching_'+args.noise_type+'_'+str(args.noise_rate)+("-%s.txt" % args.seed)
txtfile=save_dir+"/"+model_str

# Data Loader (Input Pipeline)
print('loading dataset...')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           num_workers=args.num_workers,
                                           drop_last=True,
                                           shuffle=True)
    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          num_workers=args.num_workers,
                                          drop_last=True,
                                          shuffle=False)

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2, rate_schedule):
    print('Training %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits1=model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2
        rate_schedule[epoch]=min(rate_schedule[epoch],0.99)
        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4f, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f' 
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.item(), loss_2.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2

def black_box_function(opt_param):
    mean_pure_ratio1=0
    mean_pure_ratio2=0

    print('building model...')
    cnn1 = MLP(n_outputs=num_classes)
    cnn1.cuda()
    print(cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    
    cnn2 = MLP(n_outputs=num_classes)
    cnn2.cuda()
    print(cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)
    
    # rate_schedule=opt_param[0]*(1-np.exp(-opt_param[2]*np.power(np.arange(args.n_epoch,dtype=float),opt_param[1])))+(1-opt_param[0])*(1-1/np.power((opt_param[4]*np.arange(args.n_epoch,dtype=float)+1),opt_param[3]))-np.power(np.arange(args.n_epoch,dtype=float)/args.n_epoch,opt_param[5])*opt_param[6]
    rate_schedule=opt_param[0]*(1-np.exp(-opt_param[2]*np.power(np.arange(args.n_epoch,dtype=float),opt_param[1])))\
+(1-opt_param[0])*opt_param[7]*(1-1/np.power((opt_param[4]*np.arange(args.n_epoch,dtype=float)+1),opt_param[3]))\
+(1-opt_param[0])*(1-opt_param[7])*(1-np.log(1+opt_param[8])/np.log(1+opt_param[8]+opt_param[9]*np.arange(args.n_epoch,dtype=float)))\
-np.power(np.arange(args.n_epoch,dtype=float)/args.n_epoch,opt_param[5])*opt_param[6]\
-np.log(1+np.power(np.arange(args.n_epoch,dtype=float),opt_param[11]))/np.log(1+np.power(args.n_epoch,opt_param[11]))*opt_param[10]
    print('Schedule:',rate_schedule,opt_param)
    
    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + ' ' + str(rate_schedule[epoch]) + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2, rate_schedule)
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + ' ' + str(rate_schedule[epoch]) + "\n")

    return (test_acc1+test_acc2)/200

def main():
     
    np.random.seed(args.seed)
    cur_acc=0
    max_acc=0
    num_param=12
    cur_param=np.zeros(num_param)
    max_pt=np.zeros(num_param)
    for iii in range(args.n_iter):
        for jjj in range(args.n_samples):
            for kkk in range(num_param):
                cur_param[kkk]=np.random.beta(1,1)
            cur_param[2]*=0.5
            cur_param[4]*=0.5
            cur_param[9]*=0.5
            cur_param[5]/=0.5
            cur_param[6]*=0.5
            cur_param[11]/=0.5
            cur_param[10]*=0.5
            cur_acc=black_box_function(cur_param)
            if max_acc<cur_acc:
                max_acc=cur_acc
                max_pt=cur_param.copy()
    '''
    rate_schedule=np.ones(args.n_epoch)*forget_rate
    rate_schedule[:10]=np.arange(10,dtype=float)/10*forget_rate
    # rate_schedule[10:]=np.arange(args.n_epoch-10,dtype=float)/(args.n_epoch-10)*forget_rate+forget_rate
    rate_schedule=np.zeros(args.n_epoch)
    print(rate_schedule)
    '''
    mean_pure_ratio1=0
    mean_pure_ratio2=0

    print('building model...')
    cnn1 = MLP(n_outputs=num_classes)
    cnn1.cuda()
    print(cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    
    cnn2 = MLP(n_outputs=num_classes)
    cnn2.cuda()
    print(cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + ' ' + str(rate_schedule[epoch]) + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2, rate_schedule)
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
        # save results
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + ' ' + str(rate_schedule[epoch]) + "\n")

if __name__=='__main__':
    main()

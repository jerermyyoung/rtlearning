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
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--n_iter', type=int, default=5)
parser.add_argument('--n_samples', type=int, default=4)
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
torch.cuda.set_device(args.device)
batch_size = 256
val_size = 32
num_classes = 14

CE = nn.CrossEntropyLoss().cuda(device=args.device)

data_root = '/home/yanghansi/clothing1m/'
train_dataset = Clothing(root=data_root, img_transform=train_transform, train=True, valid=False, test=False)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 32)
valid_dataset = Clothing(root=data_root, img_transform=test_transform, train=False, valid=True, test=False)
valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = val_size, shuffle = False, num_workers = 32)
test_dataset = Clothing(root=data_root, img_transform=test_transform, train=False, valid=False, test=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = val_size, shuffle = False, num_workers = 32)


# def train(train_loader, model, optimizer, criterion=CE):
def train(train_loader, model, optimizer, forget_rate, loss_rate, corr_rate, criterion=CE):
    model.train()

    # for i, (idx, input, target) in enumerate(tqdm(train_loader)):
    for i, (idx, input, target) in enumerate(train_loader):
        if idx.size(0) != batch_size:
            break
        input = torch.autograd.Variable(input.cuda(device=args.device))
        target = torch.autograd.Variable(target.cuda(device=args.device))

        output = model(input)
        # loss = criterion(output, target)
        temp = F.one_hot(target,num_classes)
        pred_1 = torch.tensor(output.tolist(),requires_grad=False)
        pred_1 = pred_1.cuda()
    # pred_2 = torch.tensor(y_2.tolist(),requires_grad=False)
    # pred_2 = pred_2.cuda()
        t_1 = torch.argmax((1-loss_rate)*temp+loss_rate*pred_1,dim=1)
        t_1 = t_1.cuda()
    # t_2 = torch.argmax((1-forget_rate)*temp+forget_rate*pred_1,dim=1)
    # t_2 = t_2.cuda()
        
        loss_1 = F.cross_entropy(output, target, reduce = False)
        ind_1_sorted = torch.argsort(loss_1).cuda(device=args.device)
        loss_1_sorted = loss_1[ind_1_sorted]
        # print(ind_1_sorted.shape,loss_1_sorted)
        '''
    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = torch.argsort(loss_2).cuda()
    ind_2_cum = ind_2_sorted.cpu() 
    # ind_2_cum = ind_2_sorted 
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_cum[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_cum[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
        
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))
        ind_1_update=ind_1_sorted[:num_remember]
        loss_1_update = F.cross_entropy(output[ind_1_update], target[ind_1_update],reduce = False)
        # print(loss_1_update.shape,num_remember)
        '''
        remember_rate = 1 - max(forget_rate,corr_rate)
        throw_rate = 1 - min(forget_rate,corr_rate)
        num_remember = int(remember_rate * len(loss_1_sorted))
        num_throw = int(throw_rate * len(loss_1_sorted))

        ind_1_update=ind_1_sorted[:num_remember]
        # ind_2_update=ind_2_sorted[:num_remember]
        ind_1_midate=ind_1_sorted[num_remember:num_throw]
        # ind_2_midate=ind_2_sorted[num_remember:num_throw]
        ind_1_downdate=ind_1_sorted[num_throw:]
        # ind_2_downdate=ind_2_sorted[num_throw:]
    # exchange
        if len(ind_1_midate)==0:
            loss_1_update = F.cross_entropy(output[ind_1_update], target[ind_1_update], reduction='none') 
#        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
        else:
            loss_1_update = torch.cat((F.cross_entropy(output[ind_1_update], target[ind_1_update], reduction='none'),F.cross_entropy(output[ind_1_midate], t_1[ind_1_midate], reduction='none')))
#        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update]) + F.cross_entropy(y_2[ind_1_midate], t_1[ind_1_midate])
#        loss = criterion(output, t_1)
        loss = torch.sum(loss_1_update)/num_throw
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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


def main_ce(opt_param):
    model_ce = resnet50().cuda(device=args.device)
    model_ce = torch.nn.DataParallel(model_ce, device_ids=[args.device,args.device+1,args.device+2,args.device+3])
    # model_ce = torch.nn.DataParallel(model_ce, device_ids=[args.device,args.device+1])
    best_ce_acc = 0
    '''
    rate_schedule=opt_param[0]*(1-np.exp(-opt_param[2]*np.power(np.arange(args.n_epoch,dtype=float),opt_param[1])))\
+(1-opt_param[0])*opt_param[7]*(1-1/np.power((opt_param[4]*np.arange(args.n_epoch,dtype=float)+1),opt_param[3]))\
+(1-opt_param[0])*(1-opt_param[7])*(1-np.log(1+opt_param[8])/np.log(1+opt_param[8]+opt_param[9]*np.arange(args.n_epoch,dtype=float)))\
-np.power(np.arange(args.n_epoch,dtype=float)/args.n_epoch,opt_param[5])*opt_param[6]\
-np.log(1+np.power(np.arange(args.n_epoch,dtype=float),opt_param[11]))/np.log(1+np.power(args.n_epoch,opt_param[11]))*opt_param[10]
    '''
    rate_schedule=opt_param[0]*(1-np.exp(-opt_param[2]*np.power(np.arange(args.n_epoch,dtype=float),opt_param[1])))\
+(1-opt_param[0])*(1-1/np.power((opt_param[4]*np.arange(args.n_epoch,dtype=float)+1),opt_param[3]))\
-np.power(np.arange(args.n_epoch,dtype=float)/args.n_epoch,opt_param[5])*opt_param[6]

    loss_schedule=opt_param[7]*(1-np.exp(-opt_param[9]*np.power(np.arange(args.n_epoch,dtype=float),opt_param[8])))\
+(1-opt_param[7])*(1-1/np.power((opt_param[11]*np.arange(args.n_epoch,dtype=float)+1),opt_param[10]))\
-np.power(np.arange(args.n_epoch,dtype=float)/args.n_epoch,opt_param[12])*opt_param[13]

    corr_schedule=opt_param[14]*(1-np.exp(-opt_param[16]*np.power(np.arange(args.n_epoch,dtype=float),opt_param[15])))\
+(1-opt_param[14])*(1-1/np.power((opt_param[18]*np.arange(args.n_epoch,dtype=float)+1),opt_param[17]))\
-np.power(np.arange(args.n_epoch,dtype=float)/args.n_epoch,opt_param[19])*opt_param[20]
    print('Schedule:',rate_schedule,loss_schedule,corr_schedule,opt_param)

    for epoch in range(args.n_epoch):
        print("epoch=", epoch)
        learning_rate = 0.01
        if epoch >= 5:
            learning_rate = 0.001

        optimizer_ce = torch.optim.SGD(model_ce.parameters(), momentum=0.9, weight_decay=1e-3, lr=learning_rate)

        print("traning model_ce...")
        train(train_loader=train_loader, model=model_ce, optimizer=optimizer_ce, forget_rate=rate_schedule[epoch],loss_rate=loss_schedule[epoch],corr_rate=corr_schedule[epoch])
        print("validating model_ce...")
        valid_acc = test(model=model_ce, test_loader=valid_loader)
        print('valid_acc=', valid_acc)
        if valid_acc > best_ce_acc:
            best_ce_acc = valid_acc
            torch.save(model_ce, './model_ce_co')

    model_ce = torch.load('./model_ce_co')
    test_acc = test(model=model_ce, test_loader=test_loader)
    print('model_ce_final_test_acc=', test_acc)
    return best_ce_acc

def main():
    np.random.seed(args.seed)
    cur_acc=0
    max_acc=0
    num_param=7*3
    cur_param=np.zeros(num_param)
    max_pt=np.zeros(num_param)
    for iii in range(args.n_iter):
        for jjj in range(args.n_samples):
            for kkk in range(num_param):
                cur_param[kkk]=np.random.beta(1,1)
            cur_param[2]*=0.2
            cur_param[4]*=0.2
            cur_param[5]*=2
            cur_param[6]*=0.01
            # cur_param[9]*=0.5
            # cur_param[11]/=0.5
            # cur_param[10]*=0.5

            cur_param[9]*=0.2
            cur_param[11]*=0.2
            cur_param[12]*=2
            cur_param[13]*=0.01
            # cur_param[9]*=0.5
            # cur_param[11]/=0.5
            # cur_param[10]*=0.5

            cur_param[16]*=0.2
            cur_param[18]*=0.2
            cur_param[19]*=2
            cur_param[20]*=0.01
            # cur_param[9]*=0.5
            # cur_param[11]/=0.5
            # cur_param[10]*=0.5
            cur_acc=main_ce(cur_param)
            if max_acc<cur_acc:
                max_acc=cur_acc
                max_pt=cur_param.copy()

if __name__ == '__main__':
    main()

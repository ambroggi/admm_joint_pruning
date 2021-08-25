from __future__ import print_function

import argparse
import torch
import torch.nn.functional as F
from optimizer2 import PruneAdam, SGD
from model2 import LeNet, AlexNet, CifarNet
from utils2 import regularized_nll_loss, admm_loss, \
    initialize_Z_and_U, update_X, update_V, update_Z1, update_Z1_l1, \
    update_Z2, update_Z3, update_U1, update_U2, update_U3, \
    print_convergence, print_prune, apply_filter, apply_prune, apply_l1_prune, \
    adjust_learning_rate
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

import torchvision.models as models


def train(args, model, device, train_loader, test_loader, optimizer):
    for name, param in model.named_parameters():
        if name[:1]=="v":
            param.requires_grad = False
    for epoch in range(args.num_pre_epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        print('Pre epoch: {}'.format(epoch + 1))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = regularized_nll_loss(args, model, output, target)
            loss.backward()
            optimizer.step()
        test(args, model, device, test_loader)


def prune_admm(args, model, device, train_loader, test_loader, optimizer):
    for name, param in model.named_parameters():
            if name[:1]=="v":
                param.requires_grad = True
    Z1, Z2, Z3, U1, U2, U3 = initialize_Z_and_U(model)
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        print('Epoch: {}'.format(epoch + 1))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = admm_loss(args, device, model, Z1, U1, Z2, U2, Z3, U3, output, target)
            loss.backward()
            optimizer.step()     
        X = update_X(model)
        Z1 = update_Z1_l1(X, U1, args) if args.l1 else update_Z1(X, U1, args)
        U1 = update_U1(U1, X, Z1)
        V = update_V(model)
        Z2 = update_Z2(V, Z2, U2, args)
        Z3 = update_Z3(V, U3)
        U2 = update_U2(U2, V, Z2)
        U3 = update_U3(U3, V, Z3)
        print_convergence(model, X, Z1)
        test(args, model, device, test_loader)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    topk=(1,5)
    maxk=max(topk)
    correct_1=0
    correct_5=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            
            _, pred = torch.topk(output, maxk, dim=1, largest=True, sorted=True)
            pred= pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred)) 
            
            correct_1+=correct[:topk[0]].float().sum().item()
            correct_5+=correct[:topk[1]].float().sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, top1_Accuracy: {}/{} ({:.0f}%), top5_Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_1, len(test_loader.dataset), 100. * correct_1 / len(test_loader.dataset),
        correct_5, len(test_loader.dataset), 100. * correct_5 / len(test_loader.dataset)))
  

def retrain(args, model, mask, device, train_loader, test_loader, optimizer):
    for epoch in range(args.num_re_epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)
        print('Re epoch: {}'.format(epoch + 1))
        model.train()
        for name, param in model.named_parameters():
            if name[:1]=="v":
                param.requires_grad = False
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.prune_step(mask)
        test(args, model, device, test_loader)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--net', type=str, default='VGG16',choices=["Lenet","CifarNet","AlexNet","VGG16"],
                        metavar="N", help='CNN network')
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "cifar10"],
                        metavar='D', help='training dataset (mnist or cifar10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--percent', type=list, default=[0.048, 0.52],
                        metavar='P', help='weight pruning percentage (default: [0.2,0.4])')
    parser.add_argument('--alpha', type=float, default=5e-4, metavar='L',
                        help='l2 norm weight (default: 5e-4)')
    parser.add_argument('--rho', type=float, default=1.5e-3, metavar='R',
                        help='cardinality weight (default: 1e-2)')
    parser.add_argument('--l1', default=False, action='store_true',
                        help='prune weights with l1 regularization instead of cardinality')
    parser.add_argument('--l2', default=False, action='store_true',
                        help='apply l2 regularization')
    parser.add_argument('--num_pre_epochs', type=int, default=3, metavar='P',
                        help='number of epochs to pretrain (default: 3)')
    parser.add_argument('--num_epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_re_epochs', type=int, default=3, metavar='R',
                        help='number of epochs to retrain (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, metavar='E',
                        help='momentum (default: 0.9)')
    parser.add_argument('--sgd_decay', type=float, default=5e-4, metavar='E',
                        help='weight_decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.dataset == "mnist":
        model=LeNet().to(device)
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    elif args.dataset == "cifar10" and args.net =="CifarNet":
        model=CifarNet().to(device)
        args.batch_size=64
        args.num_pre_epochs = 25
        args.num_epochs = 25
        args.num_re_epochs = 25
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.batch_size, **kwargs)
    
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                      (0.24703233, 0.24348505, 0.26158768))
                             ])), shuffle=True, batch_size=args.test_batch_size, **kwargs)
    
    
    elif args.dataset == "imagenet" and args.net =="Alexnet":
        model=AlexNet().to(device)
        args.lr=1e-3
        args.batch_size=256
        args.num_epochs = 25
        args.num_re_epochs = 15
        
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])
        trainset = torchvision.datasets.ImageNet('imagenet', split='train', download=None, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        
        testset = torchvision.datasets.ImageNet('imagenet', split='val', download=None, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    

    optimizer = SGD(model.named_parameters(), lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.sgd_decay)
    train(args, model, device, train_loader, test_loader, optimizer)
    
    prune_admm(args, model, device, train_loader, test_loader, optimizer)
      
    apply_filter(model,device,args)
    mask = apply_l1_prune(model, device, args) if args.l1 else apply_prune(model, device, args)

    print_prune(model)
    test(args, model, device, test_loader)
    
    optimizer = SGD(model.named_parameters(), lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.sgd_decay)
    retrain(args, model, mask, device, train_loader, test_loader, optimizer)

if __name__ == "__main__":
    main()
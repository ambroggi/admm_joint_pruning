import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.v1=nn.Parameter(torch.ones(1*20))
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.v2=nn.Parameter(torch.ones(20*50))
        self.fc1 = nn.Linear(4*4*50, 500)
        self.v3=nn.Parameter(torch.ones(500))
        self.fc2 = nn.Linear(500, 10)
        self.v4=nn.Parameter(torch.ones(10))
        

    def forward(self, x):
        x = F.conv2d(x, torch.diag(self.v1).mm(self.conv1.weight.view(1*20,5*5)).view_as(self.conv1.weight), self.conv1.bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.conv2d(x, torch.diag(self.v2).mm(self.conv2.weight.view(20*50,5*5)).view_as(self.conv2.weight), self.conv2.bias)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.linear(x, torch.diag(self.v3).mm(self.fc1.weight), bias=self.fc1.bias)
        x = F.relu(x)
        x = F.linear(x, torch.diag(self.v4).mm(self.fc2.weight), bias=self.fc2.bias)
        return F.log_softmax(x, dim=1)


class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.pruning_layers = []
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def add_addm_v_layers(self):
        count = 1
        for module in [self.conv1, self.conv2, self.fc1, self.fc2]:
            print(module)
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                self.pruning_layers.append(PostMutablePruningLayer(module))
                self.pruning_layers[-1].para.to(self.fc1.weight.device)
                # model.pruning_layers[-1].para.data = torch.rand_like(model.pruning_layers[-1].para.data)
                self.register_parameter(f"v{count}", self.pruning_layers[-1].para)
                count += 1

    def remove_addm_v_layers(self, keep):
        count = 1
        # test = [x.para.data for x in model.pruning_layers]
        while len(self.pruning_layers) > 0:
            self.__setattr__(f"v{count}", None)
            pruning_layer = self.pruning_layers.pop(0)
            pruning_layer.remove()
            count += 1



class PostMutablePruningLayer():
    def __init__(self, module: torch.nn.Module, register_parameter=True):
        if isinstance(module, torch.nn.Linear):
            self.para = torch.nn.Parameter(torch.ones(module.out_features, device=module.weight.device))
        elif isinstance(module, torch.nn.Conv2d):
            self.para = torch.nn.Parameter(torch.ones(module.out_channels, device=module.weight.device))
        else:
            print(f"Soft Pruning for Module type {module._get_name()} not implemented yet")
        self.module = module

        if register_parameter:
            self.paramiter = True
            module.register_parameter(f"v_{module._get_name()}", self.para)
        else:
            self.paramiter = False
        self.remove_hook = module.register_forward_hook(self)

    def __call__(self, module: torch.nn.Module, args: list[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        # if isinstance(module, torch.nn.Linear):
        #     return (output - module.bias[None, :]) * self.para[None, :] + module.bias[None, :]
        # elif isinstance(module, torch.nn.Conv2d):
        #     return (output - module.bias[None, :, None, None]) * self.para[None, :, None, None] + module.bias[None, :, None, None]
        if isinstance(module, torch.nn.Linear):
            return output * self.para[None, :]
        elif isinstance(module, torch.nn.Conv2d):
            return output * self.para[None, :, None, None]
        else:
            print("Soft Pruning Layer Failed")
            return output

    def remove(self, update_weights: bool = True):
        self.remove_hook.remove()
        if update_weights:
            w: torch.nn.Parameter = self.module.__getattr__("weight")
            self.module.__getattr__("weight").permute(*torch.arange(w.ndim - 1, -1, -1)).data *= self.para.data
            self.module.__getattr__("bias").permute(*torch.arange(self.module.__getattr__("bias").ndim - 1, -1, -1)).data *= self.para.data

        if self.paramiter:
            self.module.__setattr__(f"v_{self.module._get_name()}", None)
        del self.para


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1=nn.Conv2d(3, 64, kernel_size=5)
        self.v1 = nn.Parameter(torch.ones(3*64))
        #self.bn_conv1 = nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64, 64, kernel_size=5,padding=2)
        self.v2 = nn.Parameter(torch.ones(64*64))
        #self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(2304, 384)
        self.v3 = nn.Parameter(torch.ones(384))
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(384, 192)
        self.v4 = nn.Parameter(torch.ones(192))
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(192, 10)
        self.v5 = nn.Parameter(torch.ones(10))
    
    def forward(self, x):
        x = F.conv2d(x, torch.diag(self.v1).mm(self.conv1.weight.view(3*64,5*5)).view_as(self.conv1.weight), self.conv1.bias)
        #x = self.bn_conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.conv2d(x, torch.diag(self.v2).mm(self.conv2.weight.view(64*64,5*5)).view_as(self.conv2.weight), self.conv2.bias, padding=2)
        #x = self.bn_conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)
        x = x.view(-1, 2304)
        x = F.linear(x, torch.diag(self.v3).mm(self.fc1.weight), bias=self.fc1.bias)
        x = self.drop1(x)
        x = F.relu(x)
        x = F.linear(x, torch.diag(self.v4).mm(self.fc2.weight), bias=self.fc2.bias)
        x = self.drop2(x)
        x = F.relu(x)
        x = F.linear(x, torch.diag(self.v5).mm(self.fc3.weight), bias=self.fc3.bias)
        return F.log_softmax(x, dim=1)

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.v1 = nn.Parameter(torch.ones(3*64))
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.v2 = nn.Parameter(torch.ones(64*192))
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.v3 = nn.Parameter(torch.ones(192*384))
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.v4 = nn.Parameter(torch.ones(384*256))
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.v5 = nn.Parameter(torch.ones(256*256))
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.v6 = nn.Parameter(torch.ones(4096))
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.v7 = nn.Parameter(torch.ones(4096))
        self.fc3 = nn.Linear(4096, 1000)
        self.v8 = nn.Parameter(torch.ones(1000))

    def forward(self, x):
        x = F.conv2d(x,torch.transpose(self.v1*torch.transpose(self.conv1.weight.view(3*64,11*11),0,1),0,1).view_as(self.conv1.weight), self.conv1.bias, stride=4, padding=2)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.conv2d(x,torch.transpose(self.v2*torch.transpose(self.conv2.weight.view(64*192,5*5),0,1),0,1).view_as(self.conv2.weight), self.conv2.bias, padding=2)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)
        x = F.conv2d(x,torch.transpose(self.v3*torch.transpose(self.conv3.weight.view(192*384,3*3),0,1),0,1).view_as(self.conv3.weight), self.conv3.bias, padding=1)
        x = F.relu(x)
        x = F.conv2d(x,torch.transpose(self.v4*torch.transpose(self.conv4.weight.view(384*256,3*3),0,1),0,1).view_as(self.conv4.weight), self.conv4.bias, padding=1)
        x = F.relu(x)
        x = F.conv2d(x,torch.transpose(self.v5*torch.transpose(self.conv5.weight.view(256*256,3*3),0,1),0,1).view_as(self.conv5.weight), self.conv5.bias, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)        
        x = F.adaptive_avg_pool2d(x, (6,6))
        x = x.view(-1, 256*6*6)
        x = self.drop1(x)
        x = F.linear(x, torch.transpose(self.v6*torch.transpose(self.fc1.weight,0,1),0,1), bias=self.fc1.bias)
        x = F.relu(x)
        x = self.drop2(x)
        x = F.linear(x, torch.transpose(self.v7*torch.transpose(self.fc2.weight,0,1),0,1), bias=self.fc2.bias)
        x = F.relu(x)
        x = F.linear(x, torch.transpose(self.v8*torch.transpose(self.fc3.weight,0,1),0,1), bias=self.fc3.bias)
        return F.log_softmax(x, dim=1)
import torch
import torch.nn.functional as F
import numpy as np

import osqp
import scipy.sparse as sparse


def regularized_nll_loss(args, model, output, target):
    index = 0
    loss = F.nll_loss(output, target)
    if args.l2:
        for name, param in model.named_parameters():
            if name.split('.')[-1] == "weight" and (name[:4] == "conv" or name[:2] == "fc"):
                loss += args.alpha * param.norm()
                index += 1
    return loss


def admm_loss(args, device, model, Z1, U1, Z2, U2, Z3, U3, output, target):
    w_idx = 0
    v_idx = 0
    loss = F.nll_loss(output, target)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (name[:4] == "conv" or name[:2] == "fc"):
            u = U1[w_idx].to(device)
            z = Z1[w_idx].to(device)
            loss += args.rho / 2 * (param - z + u).norm()
            if args.l2:
                loss += args.alpha * param.norm()
            w_idx += 1
        elif name[:1]=="v":
            u2 = U2[v_idx].to(device)
            z2 = Z2[v_idx].to(device)
            u3 = U3[v_idx].to(device)
            z3 = Z3[v_idx].to(device)
            loss += args.rho /2 *((param - z2  + u2).norm()+(param - z3 + u3).norm())
            v_idx += 1
    return loss


def initialize_Z_and_U(model):
    Z1 = ()
    Z2 = ()
    Z3 = ()
    U1 = ()
    U2 = ()
    U3 = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (name[:4] == "conv" or name[:2] == "fc"):
            Z1 += (param.detach().cpu().clone(),)
            U1 += (torch.zeros_like(param).cpu(),)
        elif name[:1]=="v":
            Z2 += (param.detach().cpu().clone(),)
            Z3 += (param.detach().cpu().clone(),)
            U2 += (torch.zeros_like(param).cpu(),)
            U3 += (torch.zeros_like(param).cpu(),)
                
    return Z1, Z2, Z3, U1, U2, U3

def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (name[:4] == "conv" or name[:2] == "fc"):
            X += (param.detach().cpu().clone(),)
    return X

def update_V(model):
    V = ()
    for name, param in model.named_parameters():
        if name[:1]=="v":
            V += (param.detach().cpu().clone(),)
    return V

def update_Z1(X, U1, args):
    new_Z = ()
    idx = 0
    #Z = torch.cat(tuple(i.flatten()+j.flatten() for i, j in zip(X, U1)))
    for x, u in zip(X, U1):
        z = x + u
        pcen = np.percentile(abs(z), 100*(1-args.percent[idx]))
        under_threshold = abs(z) < pcen
        z.data[under_threshold] = 0
        new_Z += (z,)
        idx += 1
    return new_Z

def update_Z1_l1(X, U1, args):
    new_Z = ()
    delta = args.alpha / args.rho
    for x, u in zip(X, U1):
        z = x + u
        new_z = z.clone()
        if (z > delta).sum() != 0:
            new_z[z > delta] = z[z > delta] - delta
        if (z < -delta).sum() != 0:
            new_z[z < -delta] = z[z < -delta] + delta
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z += (new_z,)
    return new_Z


def update_Z2(V, U2, args):
    new_Z=()
    idx=0
    for v, u in zip(V, U2):
        prob = osqp.OSQP()
        P=sparse.csc_matrix(sparse.eye(len(v)))
        #P=sparse.csc_matrix(np.eye(len(z)))
        q= (- v - u).numpy()
        A=sparse.csc_matrix(sparse.vstack([sparse.eye(len(v)), np.ones([len(v),1]).transpose()]))
        #A=sparse.csc_matrix(np.vstack([np.eye(len(z)), np.ones([len(z),1]).transpose()]))
        lb=np.vstack([np.zeros([len(v),1]), args.k[idx]*np.ones([1,1])])
        ub=np.vstack([np.ones([len(v),1]),  args.k[idx]*np.ones([1,1])])
        
        prob.setup(P, q, A, lb, ub, alpha=1.0, verbose=False)
        res=prob.solve()
        new_z=res.x
        #new_z=new_z.reshape(len(z),1)
        new_Z += (torch.from_numpy(new_z),)
        idx +=1
    return new_Z


def PSp(x):
    m=x.size
    l=np.ones(m)
    t0 = (x-l/2)*np.sqrt(m)/(2*np.linalg.norm(x-l/2, 2))
    out = l/2 + t0
    return out

def update_Z3(V, U3):
    new_Z = ()
    for v, u in zip(V, U3):
        z = PSp((v + u).numpy())
        new_Z += (torch.from_numpy(z),)
    return new_Z


def update_U1(U1, X, Z1, args):
    new_U = ()
    for u, x, z in zip(U1, X, Z1):
        new_u = u + args.rho*(x - z)
        new_U += (new_u,)
    return new_U

def update_U2(U2, V, Z2, args):
    new_U = ()
    for u, v, z in zip(U2, V, Z2):
        new_u = u + args.rho*(v - z)
        new_U += (new_u,)
    return new_U

def update_U3(U3, V, Z3, args):
    new_U = ()
    for u, v, z in zip(U3, V, Z3):
        new_u = u + args.rho*(v - z)
        new_U += (new_u,)
    return new_U


def prune_filter(weight,device,number):
    weight_numpy = weight.detach().cpu().numpy()
    num=np.sort(abs(weight_numpy))[-number]
    mask = torch.Tensor(abs(weight_numpy)>=num).to(device)
    return mask

def apply_filter(model,device,args):
    V=()
    v_idx = 0
    V_idx = 0
    for name, param in model.named_parameters():
        if name[:1] =="v":
            mask = prune_filter(param, device, args.k[v_idx])
            V += (mask,)
            param.data=mask
            v_idx += 1
    for name, param in model.named_parameters():
        if len(param.shape)==4 and name.split('.')[-1] == "weight":
            param.data=torch.transpose(V[V_idx]*torch.transpose(param.view(param.shape[0]*param.shape[1],param.shape[2]*param.shape[3]),0,1),0,1).view_as(param)
            V_idx += 1
        elif len(param.shape)==2 and name.split('.')[-1] == "weight":
            param.data=torch.transpose(V[V_idx]*torch.transpose(param,0,1),0,1)
            #param.data=torch.transpose(V[V_idx]*torch.transpose(param.view(param.shape[1],param.shape[0]),0,1),0,1).view_as(param)
            V_idx += 1


def prune_weight(weight, device, percent):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    pcen=np.percentile(abs(weight_numpy),100*(1-percent))
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask


def prune_l1_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask


def apply_prune(model, device, args):
    # returns dictionary of non_zero_values' indices
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (name[:4] == "conv" or name[:2] == "fc"):
            mask = prune_weight(param, device, args.percent[idx])
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask
    

def apply_l1_prune(model, device, args):
    delta = args.alpha / args.rho
    print("Apply Pruning based on percentile")
    dict_mask = {}
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (name[:4] == "conv" or name[:2] == "fc"):
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
    return dict_mask



def print_convergence(model, X, Z1):
    idx = 0
    print("normalized norm of (weight - projection)")
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight" and (name[:4] == "conv" or name[:2] == "fc"):
            x, z = X[idx], Z1[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
            idx += 1


def print_prune(model):
    prune_v, total_v = 0, 0
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name[:1] =="v":
            print("[at parameter {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100*(abs(param)==0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
            total_v += param.numel()
            prune_v += (param != 0).sum().item()
    print("total nonzero v_parameter after pruning: {} / {} ({:.4f}%)".
          format(prune_v, total_v,
                 100 * (total_v - prune_v) / total_v))

    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and (name[:4] == "conv" or name[:2] == "fc"):
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
            total_param += param.numel()
            prune_param += (param != 0).sum().item()
    print("total nonzero weights after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

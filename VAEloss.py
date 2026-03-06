import torch.nn as nn
import torch

# Loss function with nn.BCEWithLogitsLoss
ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')  # Sum over batch and seq
mse_loss_fn = nn.MSELoss(reduction='mean')
mse_loss_sum = nn.MSELoss(reduction='sum')

def sigmoid_anneal(epoch, max_weight, max_epochs, steepness):
    '''
    Handles a sigmoid method annealling of the KL. 
    
    :param epoch: the training step. 
    :param max_weight: the max weight of the KL that we need to reach by the end of training. 
    :param max_epochs: maximum no of epochs. 
    :param steepness: how fast the increasing of KL should happen. 
    '''
    midpoint = max_epochs / 2  # Set midpoint at 15 epochs
    return max_weight / (1 + torch.exp(-steepness * (torch.tensor(epoch) - midpoint)))

def vae_dnn_loss(recon_x, x, prop, prediction, mask, logvar, mu, KL_weight, dnn_weight):
    '''
    Evaluates the total loss. The loss is averaged over the batch size. 
    
    :param recon_x: reconstructed tensor - decoder's output. 
    :param x: input tensor. 
    :param prop: target property of the SMILES. 
    :param prediction: predicted property value from DNN. 
    :param mask: bool tensor that keep tracks of the SMILES with properties. 
    :param logvar: logvar of the latent space. 
    :param mu: mean of the latent distribution. 
    :param KL_weight: KL weight in the total loss. alpha parameter in the paper. 
    :param dnn_weight: the DNN loss weight. 
    '''
    # Cross entropy loss (for reconstruction)
    recon_loss = ce_loss_fn(recon_x, x)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # sums over the batch of data, excludes sum over the seq len
    kl_loss/= x.size(0)

    # checks if there is an empty batch 
    num_valid = mask.float().sum()
    if num_valid > 0:
        regress_loss = mse_loss_fn(prediction, prop)
        return recon_loss + KL_weight * kl_loss + dnn_weight * regress_loss
    else:
        print('we found a size 0 tensor !!!')
        return recon_loss + KL_weight * kl_loss
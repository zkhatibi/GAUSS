import torch
from rdkit import Chem
import torch.nn as nn
import numpy as np


def local_perturbation(seed, scale):
    """
    Applies Local Perturbation (LP) to a given latent space point.
    
    Args:
    - seed (torch.Tensor): The original latent vector (shape: [batch_size, 128])
    - scale (float): Scaling factor for the perturbation (default: 1.0)
    
    Returns:
    - z_new (torch.Tensor): The perturbed latent vector (shape: [batch_size, 128])
    """
    latent_dim = seed.size()
    sigma = torch.full(latent_dim, fill_value=scale)  # Generate artificial standard deviation
    epsilon_local = torch.randn_like(seed)  # Sample different noise per dimension
    z_new = seed + sigma * epsilon_local  # Apply local perturbation
    
    return z_new

def make_canon(string):
    if Chem.CanonSmiles(string) is not None:
        return Chem.CanonSmiles(string)
    
def get_string_prop(latentz_point, model, idx_to_char):
    z_norm = model.normalize_latent(latentz_point)
    prediction = model.regressor(z_norm)
    new_smile = get_string(latentz_point, model, idx_to_char)
    return new_smile, prediction, z_norm

def get_string(latentz_point, model, idx_to_char):
    recon = model.decode(latentz_point.unsqueeze(0))
    recon = nn.Softmax(dim=2)(recon)
    recon = torch.argmax(recon, dim=2)
    new_smile = ''.join([idx_to_char[idx.item()] for idx in recon[0]]).replace(' ','')
    return new_smile

def lp_sampling(no_of_samples, seed, variance, canon_smiles, idx_to_char, model):
    gen_smiles = []
    for _ in range(no_of_samples):
        z_perturbed = local_perturbation(seed, scale=variance)
        new_smile = get_string(z_perturbed, model, idx_to_char)
        gen_smiles.append(new_smile)
    valid = [string for string in gen_smiles if Chem.MolFromSmiles(string) and string !='']
    novel = [string for string in valid if make_canon(string) not in canon_smiles]
    return valid, novel

def lp_sampling_w_prop(no_of_samples, seed, variance, canon_smiles, idx_to_char, model):

    gen_smiles = [] # generated smiles 
    repeat_dict = {}
    
    for _ in range(no_of_samples):
        z_perturbed = local_perturbation(seed, scale=variance)
        new_smile, prediction, _ = get_string(z_perturbed, model, idx_to_char)
        if new_smile in repeat_dict:
            repeat_dict.update({new_smile:repeat_dict[new_smile]+1})
        else:
            repeat_dict.update({new_smile:1})
        gen_smiles.append([new_smile, prediction])
    seed_string, _, _= get_string(seed, model, idx_to_char)
    print(f'The seed string is: {seed_string}')
    print(f'The strings repeated as follows: {repeat_dict}')
    gen_dict = {smile:value for smile, value in gen_smiles}
    valid = [[string,value] for string, value in gen_smiles if Chem.MolFromSmiles(string) and string !='']
    novel = [[string,value] for string, value in valid if make_canon(string) not in canon_smiles]
    return gen_dict, valid, novel


def slerp_sampling(seed1, seed2, steps_len, idx_to_char, smiles, model):
    omega = torch.arccos(torch.inner(seed1,seed2)/torch.inner(torch.norm(seed1),torch.norm(seed2)))

    steps = np.linspace(0,1,steps_len)
    z_sampled = []
    idx = []
    gen_smiles = []
    with torch.no_grad():  
        for i,t in enumerate(steps):
            z = torch.sin((1-t) * omega) * seed1 /  torch.sin(omega) + torch.sin(t * omega) * seed2 / torch.sin(omega)
            z_sampled.append(z)
            recon = model.decode(z.unsqueeze(0))
            recon = nn.Softmax(dim=2)(recon)
            recon = torch.argmax(recon, dim=2)
            new_smile = ''.join([idx_to_char[idx.item()] for idx in recon[0]]).replace(' ','')
            if new_smile not in gen_smiles:
                gen_smiles.append(new_smile)
                idx.append(i)
    gen_dict = {smile:idx[i] for i,smile in enumerate(gen_smiles)}
    valid_indices = [gen_dict[string] for string in gen_smiles if Chem.MolFromSmiles(string)]
    valid_smiles = [string for string in gen_smiles if Chem.MolFromSmiles(string)]
    novel_indices = [gen_dict[string] for string in valid_smiles if string not in smiles]

    print(len(valid_indices), len(novel_indices))
    return valid_smiles, novel_indices

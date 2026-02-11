import torch
from rdkit import Chem
import torch.nn as nn
import numpy as np

def test_loop(model, spiltted_data, idx_to_char, model_checkpoint_path, out_dir):
    '''
    Measures the model success in terms of how many of the decoded SMILES are identical to the original SMILES. 
    
    :param model: name of the model. 
    :param spiltted_data: provides the splitted data using a dataloader. 
    :param idx_to_char: index to char dictionary.  
    :param model_checkpoint_path: path to model state and parameters. 
    :param out_dir: output directory. 
    '''
    # load in the model parameters. 
    model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        for batch_data in spiltted_data.test_loader:
            x, y, mask = batch_data
            target = torch.argmax(x, dim=2)
            # the model receives both the smiles and coordinates. 
            recon, _, _, _, _, _= model(x, mask, y[:,:6])
            recon = nn.Softmax(dim=2)(recon)
            recon = torch.argmax(recon, dim=2)
    smiles_org = []
    smiles_recon = []

    for i, mol in enumerate (target):
        smiles_org.append(''.join([idx_to_char[idx.item()] for idx in mol]).replace(' ',''))
        smiles_recon.append(''.join([idx_to_char[idx.item()] for idx in recon[i]]).replace(' ',''))

    # how many of the SMILES are actual molecules. check with RDKit. 
    valid_smiles = [string for string in smiles_recon if Chem.MolFromSmiles(string)]
    # how many of the valid SMILES are identical to the original encoded SMILES. 
    correct_smiles = [string for it, string in enumerate(smiles_recon) if string == smiles_org[it]]

    print(f'Percentage of correctly reconstructed molecules: {100*len(correct_smiles)/len(target)}, Percentage of valid molecules: {100*len(valid_smiles)/len(target)}')

    if valid_smiles:
        print('Some examples of the valid structures:')
        [print(string) for string in valid_smiles[:50]]
    if correct_smiles:
        print('Some examples of the correct structures:')
        [print(string) for string in correct_smiles[:50]]
    
    np.savetxt(out_dir+'stat.txt', [100*len(correct_smiles)/len(target), 100*len(valid_smiles)/len(target)])

def gen_latentZ(model, spiltted_data, model_checkpoint_path, output_path):
    '''
    Outputs the generated latent vectors of the full set for post analysis and visulization using PCA. 
    
    :param model: model name. 
    :param spiltted_data: the dataloader that encompassed the splitted data. 
    :param model_checkpoint_path: path to model check point. 
    :param output_path: output directory. 
    '''
    print('calculating the fullset now ')

    model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True))
    model.eval()

    latent_norm = []
    latent_space = []
    latent_masked =[]
    target = []
    predicion = []
    latent_norm_masked = []

    with torch.no_grad():
        for batch_data in spiltted_data.full_loader:
            x, y, mask = batch_data
            _, y_pred, _, _, z_norm, z = model(x, mask, y[:,:6])
            latent_space.append(z)
            latent_norm.append(z_norm)
            latent_norm_masked.append(z_norm[mask])
            latent_masked.append(z[mask])
            target.append(y[mask, 6:])
            predicion.append(y_pred)

    latent_norm = torch.cat(latent_norm, dim=0)
    latent_norm_masked = torch.cat(latent_norm_masked, dim=0)
    latent_masked = torch.cat(latent_masked, dim=0)
    latent_space = torch.cat(latent_space, dim=0)
    target = torch.cat(target, dim=0)
    predicion = torch.cat(predicion, dim=0)

    latent_norm = latent_norm.numpy()
    latent_norm_masked = latent_norm_masked.numpy()
    latent_masked = latent_masked.numpy()
    latent_space = latent_space.numpy()
    target = target.numpy() 
    predicion = predicion.numpy() 

    print('Writting the latentZ space points and target properties in the output directory...')
    np.savetxt(output_path+'VAE_latent_norm.txt', latent_norm, delimiter=' ')
    np.savetxt(output_path+'VAE_latent_masked.txt', latent_masked, delimiter=' ')
    np.savetxt(output_path+'VAE_latent_norm_masked.txt', latent_norm_masked, delimiter=' ')
    np.savetxt(output_path+'VAE_latent.txt', latent_space, delimiter=' ')
    np.savetxt(output_path+'VAE_target.txt', target, delimiter=' ')
    np.savetxt(output_path+'VAE_pred.txt', predicion, delimiter=' ')
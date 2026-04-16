from one_hot_extract import * 
from data_split import * 
from model_architecture import *
from train_loop import * 
from test_loop import * 
import torch.optim as optim 
import sys
import time 
import os 
import shutil

start_time = time.time()


def reset_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  
    os.makedirs(dir_path)        
    print(f"Setting the output directory: {dir_path}")


# Training and testing:
if __name__ == '__main__':

    train, annealling = True, False   
    out_dir = './results/' # set the output directory 
    dataset_path = './data/VAE_129k_DB.txt' # set the datset path 
    VAE_model = 'VAE_XYZ' # set the GAUSS flavour, 'VAE_LOPROP' or 'VAE_XYZ'
    model_checkpoint_path = f'./data/{VAE_model}_best_state.pt' # if test, set the model parameters path 
    reset_directory(out_dir)

    smiles, char_to_idx, idx_to_char, nchars, max_len, prop, mask = define_dict_w_mask(dataset_path)
    data = smiles_to_hot(smiles=smiles, seq_len=max_len, unique_chars=nchars, char_to_idx=char_to_idx)
    spiltted_data = MaskedDatasetSplitter(data, prop, mask, percentile=.99, batch_size=100)

    # set the hyperparameters
    num_data, seq_len, input_dim = torch.tensor(data).size()
    epochs = 1
    learning_rate = 1e-3
    latent_dim = 32 # latent space should be strictly 32 
    KLD_weight = 1e-2

    # Initialize model, optimizer, and data
    model = GAUSS(input_dim=input_dim, latent_dim=latent_dim, seq_len = seq_len, mode=VAE_model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # set a scheduler to monitor the loss and modify the learning rate automatically 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = 'min',        # 'min' for loss
        factor = .5,        # new_lr = lr * factor
        patience = 0
    )

    if train:
        print('The calculation has started ...')
        training_loop_w_prop(model, optimizer, scheduler, epochs, spiltted_data, KLD_weight, annealling, VAE_model, out_dir)
    else:
        print('Reading the model checkpoint file to retrieve the model parameters...')
        print('The calculation has started ...')
        test_VAE_recon(model, spiltted_data, idx_to_char, model_checkpoint_path, out_dir) # test the performance 
        test_VAE_properties(model, spiltted_data, model_checkpoint_path, out_dir)


end_time = time.time()
hours = int((end_time-start_time)/3600)
minutes = int(((end_time-start_time)%3600)/60)
print(f'Calculation took {hours} hrs and {minutes} mins.')
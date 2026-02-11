from one_hot_extract import * 
from data_split import * 
from model_architecture import *
from training_loop import * 
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
    DB_size = 147
    out_dir = f'/media/zahra/zahraext/storage/DATA_jupyter/mol_proj/results_n_data/VAE+DNN_results/new/size_{DB_size}/'
    reset_directory(out_dir)

    KD_file_path = '/media/zahra/zahraext/storage/DATA_jupyter/mol_proj/results_n_data/qm9star_DB/unique_DB/DB_wout_xyz/'
    smiles, char_to_idx, idx_to_char, nchars, max_len, prop, mask = define_dict_w_mask(KD_file_path+f"VAE_{DB_size}k_DB.txt")
    data = smiles_to_hot(smiles=smiles, seq_len=max_len, unique_chars=nchars, char_to_idx=char_to_idx)
    spiltted_data = MaskedDatasetSplitter(data, prop, mask, percentile=.99, batch_size=100)
    VAE_model = 'VAE_GRU_2L_nochunk_v3_w_DNN'

    # Hyperparameters
    num_data, seq_len, input_dim = torch.tensor(data).size()
    epochs = 50
    learning_rate = 1e-3
    latent_dim = 32
    KLD_weight = 1e-2
    print('The calculation has started ...')
    print(f"The latent space dim is {latent_dim} and the KL weight is: {KLD_weight}")

    # Initialize model, optimizer, and data
    model = VAE_GRU_2L_nochunk_v3_w_DNN(input_dim=input_dim, latent_dim=latent_dim, seq_len = seq_len)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Define a scheduler to monitor the loss and modify the learning rate automatically 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = 'min',        # Use 'min' for loss, 'max' for accuracy
        factor = .5,        # Multiplier for LR. new_lr = lr * factor
        patience = 0
    )

    if train:
        training_loop_w_prop(model, optimizer, scheduler, epochs, spiltted_data, KLD_weight, annealling, VAE_model, out_dir)
    else:
        print('Reading the model checkpoint file to retrive the model parameters...')
        model_checkpoint_path = f'/media/zahra/zahraext/storage/DATA_jupyter/mol_proj/results_n_data/VAE+DNN_results/learning_curve_xyz/size_{DB_size}/{VAE_model}_best_state.pt'
        test_loop(model, spiltted_data, idx_to_char, model_checkpoint_path, out_dir)
        gen_latentZ(model, spiltted_data, model_checkpoint_path, out_dir)


end_time = time.time()
hours = int((end_time-start_time)/3600)
minutes = int(((end_time-start_time)%3600)/60)
print(f'Calculation took {hours} hrs and {minutes} mins.')
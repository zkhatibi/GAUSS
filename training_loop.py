import torch
from VAEloss_annealling_sum import * 
from copy import deepcopy
    
def training_loop_w_prop(model, optimizer, scheduler, epochs, spiltted_data, KLD_weight, annealling, model_name, out_dir):
    '''
    Trains the model using the training data. 
    
    :param model: loads the model architecture. 
    :param optimizer: model optimizer, has control over the learning rate through a scheduler. 
    :param scheduler: adjusts the learning rate by observing the losses over different epochs. 
    :param epochs: no of the steps in training. 
    :param train_loader: training set.  
    :param KLD_weight: the KL prefactor, adjusts the strength of the KL term in loss function. 
    :param annealling: controls the KL strength through the epochs to ensure a smooth transition of the posterior to prior. 
    :param model_name: model name. 
    '''
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_points = 0
        best_loss = float('inf')
        best_model_state = None
        
        for batch_data in spiltted_data.train_loader:
            # y contains both the traget KDs and the structural info like SPP and MPP. 
            # the first 6 element of y are the structral info and the last 7 values are the KDs 
            x, y, mask = batch_data
            batch_size, _, input_dim = x.size()

            # Forward pass
            recon, y_pred, mu, logvar, _, _ = model(x, mask, y[:,:6])
            recon = recon.reshape(-1, input_dim)  # Logits: (batch_size * seq_len, input_dim)
            target = torch.argmax(x, dim=2).reshape(-1)  # return the indices of chars in the string: (batch_size * seq_len)
            
            # Compute loss
            if annealling:
                KL_weight = sigmoid_anneal(epoch=epoch, max_weight=KLD_weight, max_epochs=30, steepness=0.4)
            else:
                KL_weight = KLD_weight 

            # loss = vae_loss(recon, target, logvar, mu, KL_weight)  # vae loss 
            loss = vae_dnn_loss(recon, target, y[mask, 6:], y_pred, mask, logvar, mu, KL_weight, dnn_weight=1)
            
            # Update tracking variables
            train_loss += loss.detach().item() * batch_size
            total_points += batch_size 
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Average loss per character
        avg_loss = train_loss / total_points

        # update the best loss value - the model params for the best loss is going to be saved for test run and sampling. 
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = deepcopy(model.state_dict()) 
        

        # Validation set loss:
        model.eval()
        val_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for batch_data in spiltted_data.test_loader:
                x, y, mask = batch_data
                batch_size, _, _ = x.size()
                recon, y_pred, mu, logvar, _, _ = model(x, mask, y[:,:6])
                recon = recon.reshape(-1, input_dim)  # Logits: (batch_size * seq_len, input_dim)
                target = torch.argmax(x, dim=2).reshape(-1)  # return the indices of chars in the string: (batch_size * seq_len)
                loss = vae_dnn_loss(recon, target, y[mask, 6:], y_pred, mask, logvar, mu, KL_weight, dnn_weight)
                val_loss += loss.detach().item() * batch_size
                total_val_samples += batch_size 
                avg_val_loss = val_loss / total_val_samples
            
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg val loss: {avg_val_loss:.4f}")
        history.append([avg_loss, avg_val_loss]) # records the losses 
        scheduler.step(avg_loss)  # LR adjustment


    with open(out_dir+'loss.txt', 'w') as file: 
        for line in history:
            file.write(' '.join(map(str,line)) +'\n')

    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss.item(), 
    #         'scheduler_state_dict': scheduler.state_dict()
    #     }, out_dir+f'/{model_name}_cp.pt')
    #     print(f"Checkpoint saved at epoch {epoch + 1}")

    # # torch.save(model.state_dict(), out_dir+f'/{model_name}_weight_n_biases.pt')
    torch.save(best_model_state, out_dir+f'/{model_name}_best_state.pt')

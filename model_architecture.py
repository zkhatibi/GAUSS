import torch.nn as nn
import torch

# VAE model:
class GAUSS(nn.Module):
    def __init__(self, input_dim, latent_dim, seq_len):
        '''
        The main architecture of the VAE model. It includes an encoder, a decoder and a property predicter.
        
        :param input_dim: The length of the char dictionary. 
        :param latent_dim: The latent vector dimension.  
        :param seq_len: The length of the largest SMILES in the dataset that is computed automatically.  
        '''
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        hidden_dim = int((latent_dim*2+64)) 
        hidden_dim2 = int((latent_dim+64)) 
        self.mean = nn.Parameter(torch.ones(latent_dim))
        self.sigma = nn.Parameter(torch.ones(latent_dim))
        
        # GRU Encoder
        self.encoder_gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True) # encapsulates the squence length in one 
        self.encoder_gru2 = nn.GRU(2*hidden_dim, hidden_dim2, batch_first=True, bidirectional=True) # encapsulates the squence length in one 
        self.encoder_gru3 = nn.GRU(2*hidden_dim2, latent_dim, batch_first=True, bidirectional=True)
        self.encoder_dense_mu = nn.Linear(2*latent_dim, latent_dim)
        self.encoder_dense_logvar = nn.Linear(2*latent_dim, latent_dim)
        self.encoder_dropout = nn.Dropout(p=0.2)
        
        # GRU Decoder
        self.decoder_gru1 = nn.GRU(latent_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder_gru2 = nn.GRU(2*hidden_dim, input_dim, batch_first=True, bidirectional=True)
        self.decoder_dense = nn.Linear(2*input_dim, input_dim)
        self.decoder_dropout = nn.Dropout(p=0.2)

        self.regressor_XYZ = RegressorXYZ(latent_dim)
        self.regressor_LOPROP = RegressorLOPROP(latent_dim)

    def normalize_latent(self, z):
        return (z-self.mean)/(1e-14 + self.sigma)

    def encode(self, x):
        out, _ = self.encoder_gru1(x) # returns the last slice of the hidden state with the dim of (1,batch_size,hidden_dim)
        out = self.encoder_dropout(out)
        out, _ = self.encoder_gru2(out)
        out = self.encoder_dropout(out)
        _, h = self.encoder_gru3(out)
        h_cat = torch.cat((h[0,:,:], h[1,:,:]), dim=-1)
        mu = self.encoder_dense_mu(h_cat).squeeze(0)
        logvar = self.encoder_dense_logvar(h_cat).squeeze(0)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick to ensure differentiability
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from normal distribution
        return mu + eps * std
    
    def decode(self, z):
        h = z.unsqueeze(1)  # Adds a dimension for sequence length ->  (batch_size, 1, latent_dim)
        h = h.repeat(1, self.seq_len, 1)  # Repeat along the sequence length (28 rows) -> (batch_size, sequence_length=28, latent_dim)
        out, _ = self.decoder_gru1(h)
        out = self.decoder_dropout(out)
        recon, _ = self.decoder_gru2(out) # generates tensors of dim : (batch_size, sequence_length=28, input_dim)
        recon = self.decoder_dense(recon)
        return recon  
    
    def forward(self, x, mask, xp, mode):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # if z.size(-1) !=32:
        #     print('we have a batch of one smile')
        z = z.reshape([-1,self.latent_dim]) # this and next line takes care of possible batches of one ligand
        recon = self.decode(z)
        z_norm = self.normalize_latent(z)
        if mode == 'VAE_XYZ':
            xp = xp.reshape([-1, 6])
            masked_xp = torch.zeros_like(xp)
            masked_xp[mask] = xp[mask]
            z_input = torch.cat((z_norm, masked_xp), dim=1) # concats the xyz info at the end of latent vector
            y = self.regressor_XYZ(z_input[mask])
        elif mode == 'VAE_LOPROP':
            y = self.regressor_LOPROP(z_norm[mask])
        return recon, y, mu, logvar, z_norm, z
    
class RegressorXYZ(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 6, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.005),
            nn.Linear(latent_dim, 8),
            nn.SiLU(),
            nn.Linear(8, 7)
        )

    def forward(self, z):
        return self.net(z)
    
class RegressorLOPROP(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(latent_dim, 4)
        )

    def forward(self, z):
        return self.net(z)
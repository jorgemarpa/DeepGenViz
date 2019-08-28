import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.tcn import TemporalConvNet


class VAE_TCN(nn.Module):
    def __init__(self, latent_dim, seq_len, kernel_size, hidden_dim, nlevels,
                 n_feats=3, dropout=0., return_norm=True, transpose=False,
                 latent_mode='repeat', con_dim=0):
        super(VAE_TCN, self).__init__()

        self.latent_dim = latent_dim
        self.return_norm = return_norm
        self.seq_len = seq_len
        ## all layers have same number of hidden units,
        ## different units per layers are possible by specifying them
        ## as a list, e.g. [32,16,8]
        self.num_channels = [hidden_dim] * nlevels
        self.latent_mode = latent_mode

        self.latent_convt1 = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, latent_dim,
                               kernel_size=seq_len, dilation=1),
            nn.ReLU())
        self.latent_linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * seq_len),
            nn.ReLU())

        self.enc_tcn = TemporalConvNet(n_feats,
                                       self.num_channels,
                                       kernel_size,
                                       dropout=dropout)

        self.enc_linear1 = nn.Linear(seq_len * hidden_dim + con_dim,
                                     latent_dim)
        self.enc_linear2 = nn.Linear(seq_len * hidden_dim + con_dim,
                                     latent_dim)

        if not transpose:
            self.dec_tcn = TemporalConvNet(latent_dim + con_dim,
                                           self.num_channels,
                                           kernel_size,
                                           dropout=dropout)
        else:
            self.dec_tcn = TemporalConvNet(latent_dim + con_dim,
                                           self.num_channels,
                                           kernel_size,
                                           dropout=dropout,
                                           transpose=True)
        self.dec_linear = nn.Linear(self.num_channels[-1], n_feats)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def encoder(self, x, c=None):
        eh1 = self.enc_tcn(x.transpose(1, 2)).transpose(1, 2)
        eh1 = eh1.flatten(start_dim=1)
        if c is not None:
            eh1 = torch.cat([eh1, c], dim=1)
        mu = self.enc_linear1(eh1)#.mean(-2)
        logvar = self.enc_linear2(eh1)#.mean(-2)
        return mu, logvar


    def decoder(self, z, c=None):
        if c is not None:
            z = torch.cat([z, c], dim=1)
        if self.latent_mode == 'linear':
            z_ = self.latent_linear(z).unsqueeze(-1).view(-1,
                                                          self.seq_len,
                                                          self.latent_dim)
        elif self.latent_mode == 'convt':
            z_ = self.latent_convt1(z.unsqueeze(-1)).transpose(1, 2)
        elif self.latent_mode == 'repeat':
            z_ = z.unsqueeze(1).repeat(1, self.seq_len, 1)

        dh1 = self.dec_tcn(z_.transpose(1, 2)).transpose(1, 2)
        xhat = self.dec_linear(dh1)
        if self.return_norm:
            xhat = self.sigmoid(xhat)
        return xhat


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad=False)
        return mu + eps*std


    def forward(self, x, c=None):
        mu, logvar = self.encoder(x, c=c)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z, c=c)
        return xhat, mu, logvar, z



class VAE_LSTM(nn.Module):
    def __init__(self, latent_dim, seq_len, hidden_dim, n_layers,
                 rnn='LSTM', n_feats=3, dropout=0., return_norm=True,
                 latent_mode='repeat'):
        super(VAE_LSTM, self).__init__()

        self.latent_dim = latent_dim
        self.return_norm = return_norm
        self.seq_len = seq_len
        self.latent_mode = latent_mode

        self.latent_convt1 = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, latent_dim,
                               kernel_size=seq_len, dilation=1),
            nn.ReLU())
        self.latent_linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * seq_len),
            nn.ReLU())

        ## batch_first --> [batch, seq, feature]
        if rnn == 'LSTM':
            self.enc_lstm = nn.LSTM(n_feats, hidden_dim, n_layers,
                                    batch_first=True, dropout=dropout,
                                    bidirectional=False)
            self.dec_lstm = nn.LSTM(latent_dim, hidden_dim, n_layers,
                                    batch_first=True, dropout=dropout,
                                    bidirectional=False)
        elif rnn == 'GRU':
            self.enc_lstm = nn.GRU(n_feats, hidden_dim, n_layers,
                                   batch_first=True, dropout=dropout,
                                   bidirectional=False)
            self.dec_lstm = nn.GRU(latent_dim, hidden_dim, n_layers,
                                   batch_first=True, dropout=dropout,
                                   bidirectional=False)

        self.enc_linear1 = nn.Linear(seq_len * hidden_dim, latent_dim)
        self.enc_linear2 = nn.Linear(seq_len * hidden_dim, latent_dim)


        self.dec_linear = nn.Linear(hidden_dim, n_feats)

        self.init_weights()

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def init_weights(self):
        for name, param in self.enc_lstm.named_parameters():
          if 'bias' in name:
             nn.init.normal_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        for name, param in self.dec_lstm.named_parameters():
          if 'bias' in name:
             nn.init.normal_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)



    def encoder(self, x):
        eh1, h_state = self.enc_lstm(x)
        eh1 = eh1.flatten(start_dim=1)
        mu = self.enc_linear1(eh1)
        logvar = self.enc_linear2(eh1)
        return mu, logvar


    def decoder(self, z):
        if self.latent_mode == 'linear':
            z_ = self.latent_linear(z).unsqueeze(-1).view(-1,
                                                          self.seq_len,
                                                          self.latent_dim)
        elif self.latent_mode == 'convt':
            z_ = self.latent_convt1(z.unsqueeze(-1)).transpose(1, 2)
        elif self.latent_mode == 'repeat':
            z_ = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        dh1, h_state = self.dec_lstm(z_)
        xhat = self.dec_linear(dh1)
        if self.return_norm:
            xhat = self.sigmoid(xhat)
        return xhat


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std, requires_grad=False)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)
        return xhat, mu, logvar, z

import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
import copy

import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

# Prepare CIFAR data
from pl_bolts.datamodules import CIFAR10DataModule
os.makedirs("temp",exist_ok = True)
datamodule = CIFAR10DataModule('temp')
mean = torch.tensor(datamodule.default_transforms().transforms[1].mean)
std = torch.tensor(datamodule.default_transforms().transforms[1].std)
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

# Save model by checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim = 512, latent_dim = 256,img_dim = (3,32,32) ):
        super().__init__()
        self.img_dim = img_dim
        self.save_hyperparameters()

        #encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim = latent_dim,
            input_height = img_dim[2],
            first_conv = False,
            maxpool1 = False
        )

        # Distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1,2,3))

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl=(log_qzx-log_pz).sum(-1)
        return kl

    def forward(self, z):
        img = self.decoder(z)
        img = img.view(img.size(0), *self.img_dim)

    def training_step(self, batch, batch_idx):
        # encoder
        x,_ =batch
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        #sampling
        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        #decoder
        x_hat = self.decoder(z)

        # reconstruction loss
        # P(x|z)
        recon_loss = self.gaussian_likelihood(x_hat,self.log_scale,x)

        #kl
        kl = self.kl_divergence(z, mu, std)

        #elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        # log sampling images
        sample_images = np.transpose(unnormalize(x_hat), (1, 2, 0))
        grid =torchvision.utils.make_grid(sample_images)
        self.logger.experiment.add_image("generated images", grid, 0)
        
        return elbo

#pl.seed_everything(1234)
print('Initializing VAE class')
vae = VAE()

# Init ModelCheckpoint callback, monitoring 'elbo'
#checkpoint_callback = ModelCheckpoint(monitor="elbo")
# saves a file like: my/path/sample-cifar10-epoch=02-elbo=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor="elbo",
    #dirpath="my/path/",
    filename="sample-cifar10-{epoch:02d}-{elbo:.2f}",
    save_top_k=3,
    mode="min",
)
# Add your callback to the callbacks list
trainer = pl.Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=10,
logger=TensorBoardLogger("lightning_logs/", name="resnet"),
callbacks=[checkpoint_callback])

trainer.fit(vae, datamodule) 

#tensorboard --logdir lightning_logs


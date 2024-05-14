import torch
from torch import nn
from ssqueezepy import issq_stft


def flatting(two_d, img_size, batch_size):
    signals = torch.empty((batch_size, img_size), device=torch.device('cuda:0'), dtype=torch.float32)
    for idx, img in enumerate(two_d):
        my_img = img[0] + img[1] * 1j
        my_sig = issq_stft(my_img)
        signals[idx, :] = my_sig
    return signals


def vae_loss(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return [reproduction_loss, KLD]


class Couple_loss(nn.Module):
    def __init__(self):
        super(Couple_loss, self).__init__()

    def forward(self, mean, log_var, output_rec, output_clas, target_rec, target_clas, batch_size, img_size, weight):
        output_sig = flatting(output_rec, img_size, batch_size)
        target_sig = flatting(target_rec, img_size, batch_size)

        my_clas_loss = nn.functional.cross_entropy(output_clas, target_clas)
        my_vae_loss_sig, my_vae_loss_kld = vae_loss(target_sig, output_sig, mean, log_var)

        return weight[0] * my_vae_loss_sig + weight[1] * my_vae_loss_kld + weight[2] * my_clas_loss

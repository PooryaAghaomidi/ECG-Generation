import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from ssqueezepy import ssq_stft
from loss.MSE import mse_loss
from optimizer.Adam import adam_opt
from model.vae import VAEModel
from model.encoder import VAE_Encoder
from model.decoder import VAE_Decoder
from torch.utils.tensorboard import SummaryWriter


class TrainAutoencoder:
    def __init__(self, device, seed, shape, data_path, my_loss, my_opt, lr, batch_size, epoch, saved_path):
        if torch.cuda.is_available() and device == 'gpu':
            self.device = "cuda"
        else:
            raise ValueError('GPU is not available')

        generator = torch.Generator(device=self.device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        encoder = VAE_Encoder().to(self.device)
        decoder = VAE_Decoder().to(self.device)
        self.model = VAEModel(encoder, decoder, shape, self.device, generator)

        data = np.load(data_path)
        self.train = data[:int(len(data) * 0.8), :]
        self.test = data[int(len(data) * 0.8):int(len(data) * 0.9), :]
        self.val = data[int(len(data) * 0.9):, :]

        self.steps_per_epoch = len(self.train) // batch_size
        self.steps_per_test = len(self.test) // batch_size
        self.steps_per_val = len(self.val) // batch_size

        if my_loss == 'mse':
            self.loss = mse_loss()
        else:
            raise ValueError('Invalid loss function')

        if my_opt == 'adam':
            self.opt = adam_opt(self.model, lr)
        else:
            raise ValueError('Invalid optimizer')

        self.saved_path = saved_path
        self.batch_size = batch_size
        self.shape = shape
        self.epoch = epoch

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('checkpoints/vae_trainer_{}'.format(self.timestamp))

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i in tqdm(range(self.steps_per_epoch)):
            start = i * self.batch_size
            Xbatch = self.train[start:start + self.batch_size, :128]
            # labels = self.train[start:start + self.batch_size, 128]
            # labels = torch.tensor(labels, dtype=torch.float32, device=self.device)

            inputs = np.empty((self.batch_size, *self.shape))
            for idx, signal in enumerate(Xbatch):
                Twxo, TF, *_ = ssq_stft(signal, n_fft=256)
                inputs[idx, 0, :, :] = TF.real[0:128, :]
                inputs[idx, 1, :, :] = TF.imag[0:128, :]
                v = inputs[idx, :, :, :]
                inputs[idx, :, :, :] = (v - v.min()) / (v.max() - v.min())

            self.opt.zero_grad()

            input_image_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            labels = input_image_tensor
            outputs = self.model(input_image_tensor)

            loss = self.loss(outputs, labels)
            loss.backward()

            self.opt.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / self.steps_per_epoch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * self.steps_per_epoch + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def training(self):
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(self.epoch):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, self.writer)

            running_vloss = 0.0
            self.model.eval()

            with torch.no_grad():
                for i in range(self.steps_per_val):
                    start = i * self.batch_size
                    Vbatch = self.val[start:start + self.batch_size, :128]
                    # vlabels = self.val[start:start + self.batch_size, 128]
                    # vlabels = torch.tensor(vlabels, dtype=torch.float32, device=self.device)

                    vinputs = np.empty((self.batch_size, *self.shape))
                    for idx, signal in enumerate(Vbatch):
                        Twxo, TF, *_ = ssq_stft(signal, n_fft=256)
                        vinputs[idx, 0, :, :] = TF.real[0:128, :]
                        vinputs[idx, 1, :, :] = TF.imag[0:128, :]
                        v = vinputs[idx, :, :, :]
                        vinputs[idx, :, :, :] = (v - v.min()) / (v.max() - v.min())

                    input_image_tensor = torch.tensor(vinputs, dtype=torch.float32, device=self.device)
                    vlabels = input_image_tensor
                    voutputs = self.model(input_image_tensor)

                    vloss = self.loss(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            self.writer.add_scalars('Training vs. Validation Loss',
                                    {'Training': avg_loss, 'Validation': avg_vloss},
                                    epoch_number + 1)
            self.writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), self.saved_path)

            epoch_number += 1

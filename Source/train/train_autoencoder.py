import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from ssqueezepy import ssq_stft
from torch.nn import functional as F
from optimizer.Adam import adam_opt
from model.vae import VAEModel
from model.encoder import VAE_Encoder
from model.decoder import VAE_Decoder
from model.fconnected import Fully_connected
from torch.utils.tensorboard import SummaryWriter
from loss import reconstruction_loss, classification_loss, MSE

# torch.autograd.set_detect_anomaly(True)


class TrainAutoencoder:
    def __init__(self, device, seed, shape, data_path, los, my_opt, lr, batch_size, epoch,
                 saved_path, weight):
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
        fully_connected = Fully_connected().to(self.device)
        self.model = VAEModel(encoder, decoder, fully_connected, shape, self.device, generator)

        data = np.load(data_path)
        self.train = data[:int(len(data) * 0.8), :]
        self.test = data[int(len(data) * 0.8):int(len(data) * 0.9), :]
        self.val = data[int(len(data) * 0.9):, :]

        self.steps_per_epoch = len(self.train) // batch_size
        self.steps_per_test = len(self.test) // batch_size
        self.steps_per_val = len(self.val) // batch_size

        if los == 'mse':
            self.loss = MSE.mse_loss()
        elif los == 'customized':
            self.loss = reconstruction_loss.Couple_loss()
        else:
            raise ValueError('Invalid reconstruction loss function')

        if my_opt == 'adam':
            self.opt = adam_opt(self.model, lr)
        else:
            raise ValueError('Invalid optimizer')

        self.saved_path = saved_path
        self.batch_size = batch_size
        self.weight = [weight['signal'], weight['KLD'], weight['class']]
        self.shape = shape
        self.epoch = epoch

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('checkpoints/vae_trainer_{}'.format(self.timestamp))

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        running_acc = 0.
        last_acc = 0.

        for i in tqdm(range(self.steps_per_epoch)):
            start = i * self.batch_size
            Xbatch = self.train[start:start + self.batch_size, :128]
            c_labels = self.train[start:start + self.batch_size, 128]
            c_labels = F.one_hot(torch.tensor(c_labels, dtype=torch.int64, device=self.device), num_classes=5).float()

            inputs = np.empty((self.batch_size, *self.shape))
            for idx, signal in enumerate(Xbatch):
                Twxo, TF, *_ = ssq_stft(signal, n_fft=256)
                inputs[idx, 0, :, :] = TF.real[0:128, :]
                inputs[idx, 1, :, :] = TF.imag[0:128, :]
                v = inputs[idx, :, :, :]
                inputs[idx, :, :, :] = (v - v.min()) / (v.max() - v.min())

            self.opt.zero_grad()

            input_image_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            r_labels = input_image_tensor
            mean, stdev, output_c, output_r = self.model(input_image_tensor)
            loss = self.loss(mean, stdev, output_r, output_c, r_labels, c_labels, self.batch_size, self.shape[1], self.weight)
            loss.backward()

            self.opt.step()

            running_loss += loss.item()
            _, predicted_output = torch.max(output_c, 1)
            _, predicted_target = torch.max(c_labels, 1)
            running_acc += (predicted_output == predicted_target).sum().item()
            if i % 500 == 499:
                last_loss = running_loss / self.steps_per_epoch
                last_acc = running_acc / (i*2)
                print('\n')
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                print('  batch {} accuracy: {}'.format(i + 1, last_acc))
                tb_x = epoch_index * self.steps_per_epoch + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar('Accuracy/train', last_acc, tb_x)
                running_loss = 0.

        return last_loss, last_acc

    def training(self):
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(self.epoch):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train(True)
            avg_loss, avg_acc = self.train_one_epoch(epoch_number, self.writer)

            running_vloss = 0.0
            running_vacc = 0.0
            self.model.eval()

            with torch.no_grad():
                for i in range(self.steps_per_val):
                    start = i * self.batch_size
                    Vbatch = self.val[start:start + self.batch_size, :128]
                    c_vlabels = self.val[start:start + self.batch_size, 128]
                    c_vlabels = F.one_hot(torch.tensor(c_vlabels, dtype=torch.float32, device=self.device),
                                          num_classes=5).float()

                    vinputs = np.empty((self.batch_size, *self.shape))
                    for idx, signal in enumerate(Vbatch):
                        Twxo, TF, *_ = ssq_stft(signal, n_fft=256)
                        vinputs[idx, 0, :, :] = TF.real[0:128, :]
                        vinputs[idx, 1, :, :] = TF.imag[0:128, :]
                        v = vinputs[idx, :, :, :]
                        vinputs[idx, :, :, :] = (v - v.min()) / (v.max() - v.min())

                    input_image_tensor = torch.tensor(vinputs, dtype=torch.float32, device=self.device)
                    r_vlabels = input_image_tensor
                    vmean, vstdev, voutput_c, voutput_r = self.model(input_image_tensor)
                    vloss = self.loss(vmean, vstdev, voutput_r, voutput_c, r_vlabels, c_vlabels, self.batch_size,
                                      self.shape[1], self.weight)

                    running_vloss += vloss
                    _, predicted_output = torch.max(voutput_c, 1)
                    _, predicted_target = torch.max(c_vlabels, 1)
                    running_vacc += (predicted_output == predicted_target).sum().item()

            avg_vloss = running_vloss / (i + 1)
            avg_vacc = running_vacc / (i + 1)
            print('\n')
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print('ACCURACY train {} valid {}'.format(avg_acc, avg_vacc))

            self.writer.add_scalars('Training vs. Validation loss',
                                    {'Training_loss': avg_loss, 'Validation_loss': avg_vloss},
                                    epoch_number + 1)
            self.writer.add_scalars('Training vs. Validation accuracy',
                                    {'Training_accuracy': avg_acc, 'Validation_accuracy': avg_vacc},
                                    epoch_number + 1)
            self.writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), self.saved_path)

            epoch_number += 1

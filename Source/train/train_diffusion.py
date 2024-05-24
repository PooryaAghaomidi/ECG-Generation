import torch
import numpy as np
from loss import MSE
from tqdm import tqdm
from model.clip import CLIP
from datetime import datetime
from model.vae import VAEModel
from utils import model_converter
from optimizer.Adam import adam_opt
from model.unet import DiffusionModel
from model.encoder import VAE_Encoder
from model.decoder import VAE_Decoder
from transformers import CLIPTokenizer
from scheduler.sampler import DDPMSampler
from ssqueezepy import ssq_stft
from torch.utils.tensorboard import SummaryWriter


def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class TrainDiffusion:
    def __init__(self,
                 device,
                 seed,
                 shape,
                 data_path,
                 los,
                 my_opt,
                 lr,
                 batch_size,
                 epoch,
                 saved_path,
                 ckpt_path,
                 vae_path,
                 tokenizier_path,
                 merg_path,
                 weight,
                 steps,
                 sampler_name):

        if torch.cuda.is_available() and device == 'gpu':
            self.device = "cuda"
        else:
            raise ValueError('GPU is not available')

        generator = torch.Generator(device=self.device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        self.tokenizer = CLIPTokenizer(tokenizier_path, merges_file=merg_path)

        state_dict = model_converter.load_from_standard_weights(ckpt_path, self.device)
        self.clip = CLIP().to(self.device)
        self.clip.load_state_dict(state_dict['clip'], strict=True)

        my_encoder = VAE_Encoder().to(self.device)
        my_decoder = VAE_Decoder().to(self.device)
        # fully_connected = Fully_connected().to(device)
        vae_model = VAEModel(my_encoder, my_decoder, shape, device, generator)
        vae_model.load_state_dict(torch.load(vae_path))

        self.encoder = vae_model.encoder
        self.decoder = vae_model.decoder

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.model = DiffusionModel()

        data = np.load(data_path)
        self.train = data[:int(len(data) * 0.9), :]
        self.test = data[int(len(data) * 0.9):, :]

        self.steps_per_epoch = len(self.train) // batch_size
        self.steps_per_test = len(self.test) // batch_size

        if los == 'mse':
            self.loss = MSE.mse_loss()
        if los == 'diffusion':
            self.loss = diffusion_loss()
        else:
            raise ValueError('Invalid loss function')

        if my_opt == 'adam':
            self.opt = adam_opt(self.model.parameters(), lr)
        else:
            raise ValueError('Invalid optimizer')

        if sampler_name == "ddpm":
            self.sampler = DDPMSampler(generator)
            self.time_steps = self.sampler.set_inference_timesteps(steps)
        else:
            raise ValueError('Invalid sampler')

        self.saved_path = saved_path
        self.batch_size = batch_size
        self.weight = [weight['signal'], weight['image']]
        self.shape = shape
        self.epoch = epoch

        self.classes = ['type N heartbeat', 'type S heartbeat', 'type V heartbeat', 'type F heartbeat', 'type Q heartbeat']

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('checkpoints/vae_trainer_{}'.format(self.timestamp))

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i in tqdm(range(self.steps_per_epoch)):
            start = i * self.batch_size
            Xbatch = self.train[start:start + self.batch_size, :128]
            class_label = self.train[start:start + self.batch_size, 128]
            prompt = [self.classes[int(label)] for label in class_label]

            tokens = self.tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            context = self.clip(tokens).to(self.device)

            context.to(self.device)

            inputs = np.empty((self.batch_size, *self.shape))
            for idx, signal in enumerate(Xbatch):
                Twxo, TF, *_ = ssq_stft(signal, n_fft=256)
                inputs[idx, 0, :, :] = TF.real[0:128, :]
                inputs[idx, 1, :, :] = TF.imag[0:128, :]
                v = inputs[idx, :, :, :]
                inputs[idx, :, :, :] = (v - v.min()) / (v.max() - v.min())

            input_image_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            labels = input_image_tensor
            latents = self.sampler.add_noise(input_image_tensor, self.time_steps[0])

            for idx, step in enumerate(self.time_steps):
                self.opt.zero_grad()

                time_embedding = get_time_embedding(step).to(self.device)

                input_image = latents
                model_input = self.encoder(input_image)

                z = self.model(model_input, context, time_embedding)

                model_output = self.decoder(z)

                latents = self.sampler.step(step, latents, model_output)

                if idx != 999:
                    next_input = self.sampler.add_noise(input_image_tensor, self.time_steps[idx+1])
                else:
                    next_input = input_image_tensor
                    loss = self.loss(latents, next_input)
                    running_loss += loss.item()

                loss = self.loss(latents, next_input)
                loss.backward()

                self.opt.step()

            if i % 500 == 499:
                last_loss = running_loss / self.steps_per_epoch
                print('\n')
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * self.steps_per_epoch + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def train_diffusion_model(self):
        epoch_number = 0

        for epoch in range(self.epoch):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, self.writer)

            print('\n')
            print('LOSS train: ', avg_loss)

            self.writer.add_scalars('loss',
                                    {'Training_loss': avg_loss},
                                    epoch_number + 1)
            self.writer.flush()

            torch.save(self.model.state_dict(), self.saved_path)
            epoch_number += 1

import json
from configs.inference_config import write_inference_config
from configs.vaetrain_config import write_vaetrain_config
from evaluate.inference import StableDiffusion
from train.train_autoencoder import TrainAutoencoder

state = 'train_vae'


def train_vae(config_path):
    write_vaetrain_config()
    with open(config_path, 'r') as openfile:
        my_configs = json.load(openfile)

    my_class = TrainAutoencoder(device=my_configs['device'],
                                seed=my_configs['seed'],
                                shape=my_configs['shape'],
                                data_path=my_configs['data_path'],
                                los=my_configs['los'],
                                my_opt=my_configs['my_opt'],
                                lr=my_configs['lr'],
                                batch_size=my_configs['batch_size'],
                                epoch=my_configs['epoch'],
                                saved_path=my_configs['saved_path'],
                                weight=my_configs['weight'])
    my_class.training()


def stable_diffusion_inference(prompt, config_path):
    write_inference_config()
    with open(config_path, 'r') as openfile:
        my_configs = json.load(openfile)

    my_class = StableDiffusion(ALLOW_CUDA=my_configs['ALLOW_CUDA'],
                               ALLOW_MPS=my_configs['ALLOW_MPS'],
                               tokenizier_path=my_configs['tokenizier_path'],
                               merg_path=my_configs['merg_path'],
                               model_file=my_configs['model_file'],
                               sampler=my_configs['sampler'],
                               num_inference_steps=my_configs['num_inference_steps'],
                               seed=my_configs['seed'],
                               uncond_prompt=my_configs['uncond_prompt'],
                               do_cfg=my_configs['do_cfg'],
                               cfg_scale=my_configs['cfg_scale'],
                               input_image=my_configs['input_image'],
                               image_path=my_configs['image_path'],
                               strength=my_configs['strength'])

    return my_class.generate_image(prompt)


# TODO: 1. Display training information
#       2. Complete VAE training with classification
#       3. Train diffusion model

if __name__ == "__main__":
    if state == 'inference':
        my_prompt = 'a beautiful dog'
        my_config_path = 'configs/inference_configs.json'
        image = stable_diffusion_inference(my_prompt, my_config_path)
        image.show()
    elif state == 'train_vae':
        my_config_path = 'configs/vaetrain_configs.json'
        train_vae(my_config_path)
    elif state == 'train_diff':
        pass
    else:
        raise ValueError('There are only two states of inference and train')

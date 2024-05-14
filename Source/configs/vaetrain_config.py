import json


def write_vaetrain_config():
    my_config = {'device': 'gpu',
                 'seed': 42,
                 'shape': (2, 128, 128),
                 'fc_shape': [256, 64],
                 'data_path': '../Dataset/Dataset.npy',
                 'los': 'customized',
                 'my_opt': 'adam',
                 'lr': 0.001,
                 'batch_size': 2,
                 'epoch': 50,
                 'saved_path': 'checkpoints/vae_2',
                 'weight': {'signal': 0.3, 'KLD': 0.3, 'class': 0.4}}

    with open("configs/vaetrain_configs.json", "w") as outfile:
        json.dump(my_config, outfile)

import json


def write_vaetrain_config():
    my_config = {'device': 'gpu',
                 'seed': 42,
                 'shape': (2, 128, 128),
                 'data_path': '../Dataset/Dataset.npy',
                 'my_loss': 'mse',
                 'my_opt': 'adam',
                 'lr': 0.001,
                 'batch_size': 2,
                 'epoch': 50,
                 'saved_path': 'checkpoints/vae'}

    with open("configs/vaetrain_configs.json", "w") as outfile:
        json.dump(my_config, outfile)

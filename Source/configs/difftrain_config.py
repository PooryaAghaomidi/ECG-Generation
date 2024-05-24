import json


def write_difftrain_config():
    my_config = {'device': 'gpu',
                 'seed': 42,
                 'shape': (2, 128, 128),
                 'data_path': '../Dataset/CleanDataset.npy',
                 'los': 'mse',
                 'my_opt': 'adam',
                 'lr': 0.001,
                 'batch_size': 2,
                 'epoch': 50,
                 'saved_path': 'checkpoints/diff',
                 'vae_path': 'checkpoints/vae',
                 'ckpt_path': 'checkpoints/v1-5-pruned.ckpt',
                 'weight': {'signal': 0.5, 'image': 0.5},
                 'steps': 1000,
                 'sampler_name': 'ddpm',
                 'tokenizier_path': "checkpoints/vocab.json",
                 'merg_path': "checkpoints/merges.txt"}

    with open("configs/difftrain_configs.json", "w") as outfile:
        json.dump(my_config, outfile)

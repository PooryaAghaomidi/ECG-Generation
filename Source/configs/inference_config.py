import json


def write_inference_config():
    my_config = {'ALLOW_CUDA': True,
                 'ALLOW_MPS': False,
                 'tokenizier_path': "checkpoints/vocab.json",
                 'merg_path': "checkpoints/merges.txt",
                 'model_file': "checkpoints/v1-5-pruned.ckpt",
                 'sampler': "ddpm",
                 'num_inference_steps': 50,
                 'seed': 42,
                 'uncond_prompt': "",
                 'do_cfg': True,
                 'cfg_scale': 8,
                 'input_image': None,
                 'image_path': "",
                 'strength': 0.9}

    with open("configs/inference_configs.json", "w") as outfile:
        json.dump(my_config, outfile)

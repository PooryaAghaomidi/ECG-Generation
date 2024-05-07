import os
import json
import random


def split(address='../../Dataset/'):
    files = os.listdir(address + 'Final')
    random.shuffle(files)

    with open(address + 'shuffled_data', "w") as fp:
        json.dump(files, fp)

    prompts = {'Heartbeats': {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'},
               'Noises': {0: 'No', 1: 'Low', 2: 'Medium', 3: 'High'},
               'Types': {128: 'Class', 129: 'Baseline wander', 130: 'Electrode motion artifact', 131: 'Muscle artifact', 132: 'Random'}}

    with open(address + 'prompts', "w") as fp:
        json.dump(prompts, fp)


split()

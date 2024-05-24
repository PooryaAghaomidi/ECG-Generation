import random
import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample

address = '../../Dataset/'
train_df = pd.read_csv(address + 'mitbih_train.csv', header=None)

df_1 = train_df[train_df[187] == 1]
df_2 = train_df[train_df[187] == 2]
df_3 = train_df[train_df[187] == 3]
df_4 = train_df[train_df[187] == 4]
df_0 = (train_df[train_df[187] == 0]).sample(n=20000, random_state=42)

df_1_upsample = resample(df_1, replace=True, n_samples=20000, random_state=123)
df_2_upsample = resample(df_2, replace=True, n_samples=20000, random_state=124)
df_3_upsample = resample(df_3, replace=True, n_samples=20000, random_state=125)
df_4_upsample = resample(df_4, replace=True, n_samples=20000, random_state=126)

train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

bwm = scipy.io.loadmat(address + 'bwm.mat')['val'][0, :]
emm = scipy.io.loadmat(address + 'emm.mat')['val'][0, :]
mam = scipy.io.loadmat(address + 'mam.mat')['val'][0, :]

bwm = 2 * ((bwm - bwm.min()) / (bwm.max() - bwm.min())) - 1
emm = 0.5 * (emm - emm.min()) / (emm.max() - emm.min())
mam = (mam - mam.min()) / (mam.max() - mam.min())
rdm = np.random.randint(100, size=(128))
rdm = 0.1 * ((rdm - rdm.min()) / (rdm.max() - rdm.min()))

counter = 0
# My_data = np.empty((200000, 129))
Clean_data = np.empty((100000, 129))

for i in tqdm(range(len(train_df))):
    row = train_df.iloc[i]
    My_signal = row[5:133]
    My_class = int(row[187])

    Clean_data[counter, :128] = np.array(My_signal)
    Clean_data[counter, 128] = My_class
    counter = counter + 1

    # start = random.randint(0, 640000)
    # noisy_sample = np.array(My_signal + bwm[start:start + 128] + emm[start:start + 128] + mam[start:start + 128] + rdm)

    # My_data[counter, :128] = noisy_sample
    # My_data[counter, 128] = My_class
    # counter = counter + 1

np.random.shuffle(Clean_data)
np.save(address + 'CleanDataset.npy', Clean_data)

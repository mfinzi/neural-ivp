import os
from scipy.io import arff
import numpy as np
import pandas as pd

targets = [b'1', b'2', b'3', b'4', b'5', b'6', b'7']
rf_num = int(5 * 1.e3)
# rf_num = int(1 * 1.e4)
seed = 21
np.random.seed(seed=seed)
home_path = os.environ["HOME"]
file_path = home_path + '/Downloads/shuttle.arff'
data, meta = arff.loadarff(file_path)
obs_n = data.shape[0]
# sample_size = 100
sample_size = int(obs_n * 0.75)
data = data[:sample_size]
df = pd.DataFrame(data)
aux = df['class'].to_numpy()
y = np.zeros(shape=(aux.shape[0], len(targets)))
for j in range(len(targets)):
    target = targets[j]
    for i in range(y.shape[0]):
        if aux[i] == target:
            y[i, j] = 1.
res = np.mean(y, axis=0)
text = 'Proportion of class: '
for i in range(res.shape[0]):
    text += f'{res[i]:1.3e} '
print(text)
x = df.drop('class', axis=1).to_numpy()

# offset = np.random.uniform(low=0., high=2. * np.pi, size=(rf_num,))
# coeff = np.random.normal(loc=0., scale=0.75, size=(x.shape[1], rf_num,))
# rff = (np.sqrt(2. / rf_num)) * np.cos(x @ coeff + offset)
offset = np.zeros(shape=(rf_num,))
coeff = np.random.normal(loc=0., scale=0.75, size=(x.shape[1], rf_num,))
z1 = np.cos(x @ coeff + offset)
z2 = np.sin(x @ coeff + offset)
rff = (1. / np.sqrt(rf_num)) * np.concatenate((z1, z2), axis=1)

print('\nGenerating matrices')
GTG = rff.T @ rff
vec = rff.T @ y
vec = vec.reshape(-1, order='F')

print('\nSaving files')
np.save(file='./data/shuttle_y.npy', arr=y)
np.save(file='./data/shuttle_x.npy', arr=rff)
np.save(file='./data/shuttle_gtg.npy', arr=GTG)
np.save(file='./data/shuttle_vec.npy', arr=vec)
print('\nSaved')

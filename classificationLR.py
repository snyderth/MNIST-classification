import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

training_data = loadmat('MNIST_train_image.mat')
training_labels = loadmat('MNIST_train_label.mat')
print(np.shape(training_data['trainX']))
sampler = np.random.choice(np.shape(training_data['trainX'])[1],10000)
print(sampler)
import pandas as pd
from pandas.core.algorithms import mode
from scipy.io import loadmat
from scipy import stats
from tqdm import tqdm
import numpy as np

def KNN(train_dataset: pd.DataFrame, input_dataset: pd.DataFrame, k, name):
    train_data = train_dataset.drop("labels", axis=1)
    input_data = input_dataset.drop("labels", axis=1)
    mse = pd.DataFrame(columns=["mse"],dtype=float)
    for index, row in tqdm(input_data.iterrows(),desc="Computing {} KNN".format(name),total=input_data.shape[0]):
        mse.loc[index] = ((train_data - row)**2).sum()
        knn = mse.nsmallest(k,"mse")
        train_nn = train_dataset.loc[knn.index, "labels"]
        cls = train_nn.mode
        input_dataset.loc["class",index] = cls
    return input_dataset

def KNN_np(train_data: np.array, train_labels:np.array, input_data: np.array, input_labels: np.array, k, name):
    classes = [0] * input_data.shape[0]
    for i in tqdm(range(input_data.shape[0]), desc="Compute {} KNN".format(name)):
        # print((train_data - input_data[i]).shape)
        mse = np.linalg.norm(train_data - input_data[i], axis=1) ** 2
        # print(mse)
        knn = np.argpartition(mse, k)
        classes[i] = stats.mode(train_labels[knn[0:k]])
    return np.array(classes) 



training_data = loadmat('MNIST_train_image.mat')
training_labels = loadmat('MNIST_train_label.mat')
train_df = pd.DataFrame(training_data['trainX']).T
train_lab = pd.DataFrame(training_labels['trainL'])

train_data = train_df.sample(n=50000).reset_index().drop('index',axis=1)
train_labels = train_lab.loc[train_data.index].reset_index().drop('index',axis=1)
# train_data['labels'] = train_labels[0]

validation_data = train_df.drop(train_data.index).reset_index().drop('index',axis=1)
validation_labels = train_lab.drop(train_data.index).reset_index().drop('index',axis=1)
# validation_data['labels'] = validation_labels[0]


testing_data = loadmat('MNIST_test_image.mat')
testing_labels = loadmat('MNIST_test_label.mat')
test_df = pd.DataFrame(testing_data['testX']).T
test_lab = pd.DataFrame(testing_labels['testL'])

# output = KNN(train_data, validation_data, 5, "test")
output = KNN_np(train_data.to_numpy(), train_labels.to_numpy(), validation_data.to_numpy(), validation_labels.to_numpy(), 5, "validate")
output_df = pd.DataFrame(output)
output_df.to_csv('KNN_output.csv')
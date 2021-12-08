import pandas as pd
from pandas.core.algorithms import mode
from scipy.io import loadmat
from scipy import stats
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

Knum = [1, 5, 10, 20, 50, 100, 200, 500]
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
        mse = cp.linalg.norm(cp.asarray(train_data) - cp.asarray(input_data[i]), axis=1) ** 2
        knn = cp.asnumpy(np.argpartition(mse, k))
        classes[i] = stats.mode(train_labels[knn[0:k]])[0][0][0]
        # print("\nKNN: {}, Actual: {}, Decision: {}".format(train_labels[knn[0:k]], classes[i], input_labels[i]))
    return np.array(classes) 



training_data = loadmat('MNIST_train_image.mat')
training_labels = loadmat('MNIST_train_label.mat')
train_df = pd.DataFrame(training_data['trainX']).T
train_lab = pd.DataFrame(training_labels['trainL'])

train_data = train_df.sample(n=59900).reset_index().drop('index',axis=1)
train_labels = train_lab.loc[train_data.index].reset_index().drop('index',axis=1)
# train_data['labels'] = train_labels[0]

validation_data = train_df.drop(train_data.index).reset_index().drop('index',axis=1)
validation_labels = train_lab.drop(train_data.index).reset_index().drop('index',axis=1)
# validation_data['labels'] = validation_labels[0]


testing_data = loadmat('MNIST_test_image.mat')
testing_labels = loadmat('MNIST_test_label.mat')
test_df = pd.DataFrame(testing_data['testX']).T
test_lab = pd.DataFrame(testing_labels['testL'])


success_df = pd.DataFrame()
for k in Knum:
# output = KNN(train_data, validation_data, 5, "test")
    output = KNN_np(train_data.to_numpy(), train_labels.to_numpy(), validation_data.to_numpy(), validation_labels.to_numpy(), k, "validate")
    output_df = pd.DataFrame(output, dtype=float)
    # print(output_df)
    # print(validation_labels)
    # print(((output_df - validation_labels.astype(float))**2).sum())
    num_good = (output_df == validation_labels.astype(float)).sum()
    success_df.loc[k, "success"] = num_good[0]
    
    output_df.to_csv('KNN_output.csv')

success_df.to_csv("Success.csv")
success_df.plot(x = success_df.index, y="success")
plt.plot()
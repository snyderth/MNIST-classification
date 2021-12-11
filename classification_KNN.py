import pandas as pd
from pandas.core.algorithms import mode
from scipy.io import loadmat
from scipy import stats
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import sys
# np.set_printoptions(threshold=sys.maxsize)
Knum = [1, 5, 10, 20, 50, 100, 200, 500]
def KNN(train_dataset: pd.DataFrame, input_dataset: pd.DataFrame, k, name):
    train_data = train_dataset.drop("labels", axis=1)
    input_data = input_dataset.drop("labels", axis=1)
    mse = pd.DataFrame(columns=["mse"],dtype=float)
    for index, row in tqdm(input_data.iterrows(),desc="Computing {} KNN".format(name),total=input_data.shape[0]):
        mse.loc[index] = ((train_data - row)**2).sum()
        train_nn = train_dataset.loc[knn.index, "labels"]
        cls = train_nn.mode
        input_dataset.loc["class",index] = cls
    return input_dataset

def KNN_np(train_data: np.array, train_labels:np.array, input_data: np.array, input_labels: np.array, k: list, name):
    classes = np.zeros((input_data.shape[0], len(k))) 
    error = np.zeros((input_data.shape[0], len(k))) 

    for i in tqdm(range(input_data.shape[0]), desc="Compute {} KNN".format(name)):
        mse = cp.sum(cp.square(cp.asarray(train_data) - cp.asarray(input_data[i])), axis=1)
        # print(mse[0:10])
        for kn in range(len(k)):
            knn = cp.asnumpy(np.argpartition(mse, k[kn]))
            
            # print(stats.mode(train_labels[knn[np.arange(k[kn])]]))
            # print(train_labels[knn[np.arange(k[kn])]])
            classes[i][kn] = stats.mode(train_labels[knn[np.arange(k[kn])]])[0][0][0]
            # print(classes[i][kn])
            if classes[i][kn] != input_labels[i]:
                error[i][kn] = 1
        # print("Guesses: {}, Actual: {}".format(classes[i], input_labels[i]))
    # print(cp.sum(error, axis=0)) 
    mean_error = cp.mean(error, axis=0)
    # print(mean_error)
        # print("\nKNN: {}, Actual: {}, Decision: {}".format(train_labels[knn[0:k]], classes[i], input_labels[i]))
    return np.array(classes), mean_error



training_data = loadmat('MNIST_train_image.mat')
training_labels = loadmat('MNIST_train_label.mat')
# print(training_data['trainX'].shape)
# training_data_std = np.std(np.std(training_data['trainX'], axis=0))
# training_data_mean = np.mean(np.mean(training_data['trainX'], axis=0))

train_df = pd.DataFrame(training_data['trainX']).T
train_lab = pd.DataFrame(training_labels['trainL'])

train_data = train_df.sample(n=50000).reset_index()
# print(train_data.index)
# print(train_data.loc[20,:])
# print(train_data.loc[20,:])
train_labels = train_lab.loc[train_data['index']].reset_index().drop('index',axis=1)
# train_data['labels'] = train_labels[0]
validation_data = train_df.drop(train_data['index']).reset_index().drop('index',axis=1)
validation_labels = train_lab.drop(train_data['index']).reset_index().drop('index',axis=1)
# validation_data['labels'] = validation_labels[0]
train_data = train_data.drop('index',axis=1)

testing_data = loadmat('MNIST_test_image.mat')
testing_labels = loadmat('MNIST_test_label.mat')
test_df = pd.DataFrame(testing_data['testX']).T
test_lab = pd.DataFrame(testing_labels['testL'])

# print(train_df)
# print(test_lab)
# success_df = pd.DataFrame()
# for k in Knum:
# output = KNN(train_data, validation_data, 5, "test")
output_train, error_perf_train = KNN_np(train_data.to_numpy(), train_labels.to_numpy(), train_data.to_numpy(), train_labels.to_numpy(), Knum, "train")
output_verif, error_perf_validation = KNN_np(train_data.to_numpy(), train_labels.to_numpy(), validation_data.to_numpy(), validation_labels.to_numpy(), Knum, "validate")
output_test, error_perf_test = KNN_np(train_data.to_numpy(), train_labels.to_numpy(), test_df.to_numpy(), test_lab.to_numpy(), Knum, "test")
# output, error_perf = KNN_np(train_data.to_numpy(), train_labels.to_numpy(), test_df.to_numpy(), test_lab.to_numpy(), Knum, "validate")
output_df = pd.DataFrame([error_perf_train, error_perf_validation, error_perf_test])
output_df.index = Knum
    # print(output_df)
    # print(validation_labels)
    # print(((output_df - validation_labels.astype(float))**2).sum())
# num_good = (output_df == validation_labels.astype(float)).sum()
# success_df.loc[k, "success"] = num_good[0]
    
output_df.to_csv('KNN_error.csv')

output_df.index = Knum
# success_df.to_csv("Success.csv")
# success_df.plot(x = success_df.index, y="success")
output_df.plot(x=output_df.index)
plt.show()
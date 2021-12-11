from os import device_encoding
import numpy as np
import cupy as cp
from numpy.lib.function_base import gradient
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
# np.set_printoptions(threshold=sys.maxsize)
import multiprocessing as mpt
import enum
import pandas as pd

class LogisticRegressionModelGD(object):
    def __init__(self, training_data: cp.ndarray, training_labels: cp.ndarray, beta_init: cp.ndarray, validation_data: cp.ndarray, validation_labels: cp.ndarray, learning_rate=0.1, lam=1.0):
        self.train_data = training_data
        self.train_labels = training_labels
        self.beta = beta_init
        self.lr = learning_rate
        self.lam = lam
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def __call__(self):
        self.cross_entropy_error_gradient_descent(self.train_data, self.train_labels, self.beta, self.validation_data, self.validation_labels, self.lr, self.lam)

    # @njit(parallel=True, nopython=True)
    def cross_entropy_error(self, x: cp.ndarray, training_labels: cp.ndarray, beta: cp.ndarray, indicator, reg_term):
        # print(cp.multiply(indicator, x@beta))
        # print(cp.multiply(indicator, cp.exp(cp.multiply(indicator, x@beta)))) 
        sub = cp.log(cp.exp((x@beta)).sum(axis=1))
        sub = cp.asarray([sub for i in range(beta.shape[1])])
        ln_post_prob = cp.multiply(indicator , (x @ beta) - sub.T)
        # ln_post_prob = ( (x @ beta) - sub.T)
        # print("CEE")
        # print(ln_post_prob)

        error = (-1/x.shape[0]) * ((ln_post_prob).sum() + reg_term * (cp.linalg.norm(beta, axis=1)**2).sum())
        return cp.asarray(error)
    # @njit(nopython=True)
    def posterior_prob(self,x: cp.ndarray, beta:cp.ndarray, indicator):
        numerator = cp.multiply(indicator, cp.exp(x @ beta))
        # numerator = cp.exp(x @ beta)
        # print(numerator.shape)
        denom = (cp.exp(x @ beta).sum(axis=1))
        arr = []
        for i in range(beta.shape[1]):
            arr.append(denom)
        denom = cp.asarray(arr)
        # print(denom.shape)
        # print(denom)
        post =cp.divide(numerator, denom.T)
        # print(post)
        return post
    
    def validate_map(self, valid_data: cp.ndarray, valid_labels: cp.ndarray, beta: cp.ndarray):
        result = (valid_data @ beta)
        # print(result.shape)
        max_result = (np.argpartition(cp.asnumpy(result), kth=result.shape[1]-1, axis=1))[:,-1]
        res = (np.zeros((len(max_result), 1)))
        for i in range(valid_labels.shape[0]):
            if valid_labels[i] != max_result[i]:
                res[i] = 1
        
        print(np.mean(res))
        



    def cross_entropy_error_gradient_descent(self,x:cp.ndarray, training_labels:cp.ndarray, beta:cp.ndarray, x_valid:cp.ndarray, valid_labels:cp.ndarray, eta, reg_term):
        indicator = cp.asarray(np.zeros((x.shape[0], beta.shape[1])))
        for i in range(x.shape[0]):
            indicator[i,training_labels[i]] = 1
        validation_indicator = cp.asarray(np.zeros((x_valid.shape[0], beta.shape[1])))
        for i in range(x_valid.shape[0]):
            validation_indicator[i,valid_labels[i]] = 1
        norm_error = []
        validation_norm_error = [] 
        # grad = -1/x.shape[0] * ((x.T @ (indicator - cp.multiply(indicator, self.posterior_prob(x, beta, indicator)).sum(axis = 1)))) #+ 2 * reg_term * beta
        for i in tqdm(range(3000), ascii=True, desc="Lambda: {}".format(reg_term)):

            # grad = -1/x.shape[0] * (x.T @ (indicator - self.posterior_prob(x, beta, indicator))) #+ 2 * reg_term * beta
            grad = -1/x.shape[0] * ((x.T @ (indicator - self.posterior_prob(x, beta, indicator)))) + 2 * reg_term * beta
            # print(grad)
            beta = beta - eta * grad

            err = cp.linalg.norm(self.cross_entropy_error(x, training_labels, beta, indicator, reg_term))**2
            validation_err = cp.linalg.norm(self.cross_entropy_error(x_valid, valid_labels, beta, validation_indicator, reg_term))**2
            self.validate_map(x_valid, valid_labels, beta)
            # print()
            # print("Valid Err: {}".format(validation_err))
            validation_norm_error.append(validation_err)
            norm_error.append(err)

        plt.figure(reg_term % 0.5)
        plt.subplot(211)
        plt.plot(norm_error)
        plt.title("Lambda {}, Step {}".format(reg_term,eta))
        plt.subplot(212)
        plt.plot(validation_norm_error)
        np.savetxt("./final_testing/beta_lam{}_{}.csv".format(reg_term, eta), cp.asnumpy(beta), delimiter=",")
        np.savetxt("./final_testing/norm_error{}_{}.csv".format(reg_term, eta), cp.asnumpy(norm_error), delimiter=",")
        np.savetxt("./final_testing/valid_norm_error{}_{}.csv".format(reg_term, eta), cp.asnumpy(validation_norm_error), delimiter=",")
        plt.savefig("./final_testing/beta_lam{}_{}.png".format(reg_term, eta))
        print("Saved!")
        # plt.show()
        # return beta # optimal beta

class Worker(mpt.Process):
    def __init__(self, work_queue):
        mpt.Process.__init__(self)
        self.q = work_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.q.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.q.task_done()
                break
            print('%s: %s' % (proc_name, next_task))
            answer = next_task()
            self.q.task_done()
            # self.result_queue.put(answer)
        return

# def worker(q):
#     job = q.get()
#     job.run()

def main():
    training_data = loadmat('MNIST_train_image.mat')
    training_labels = loadmat('MNIST_train_label.mat')

    sampler = np.random.choice(np.shape(training_data['trainX'])[1],10000, replace=False)
    # print(type(training_data['trainX'].T.get()))
    validation_data = (training_data['trainX'].T)[sampler]
    validation_labels = (training_labels['trainL'])[sampler]
    training_data = np.delete(training_data['trainX'].T, sampler, axis=0)
    training_labels = np.delete(training_labels['trainL'], sampler, axis=0)

    aug_train = [[1]] * training_data.shape[0]
    aug_valid = [[1]] * validation_data.shape[0]
    validation_data = np.append(validation_data, aug_valid, axis = 1)
    training_data = np.append(training_data, aug_train, axis = 1)
    # print(aug_train)
    beta_init = np.random.normal(0, size = (training_data.shape[1], 10))
    # beta_init = np.zeros((training_data.shape[1], 10)) #+ 0.000000001
    # print(validation_data.shape)
    # print(training_data.shape)
    # print(validation_labels.shape)
    # print(training_labels.shape)
    # model = LogisticRegressionModelGD(training_data, training_labels, beta_init, 0.1, 0.5)
    # model.run()
    # print(training_labels.dtype)
    # print(training_data.dtype)
    # print(beta_init.dtype)
    # lam = [6.5, 7, 7.5, 8, 8.5, 9,10, 10.5, 11, 11.5, 8, 8.5, 9]
    startTraining(training_data, training_labels, beta_init, validation_data, validation_labels)


def startTraining(training_data, training_labels, beta_init, validation_data, validation_labels):
    print(training_data.shape)
    print(validation_data.shape)
    lam = np.linspace(0.001, 116.0, 1)
    step = np.linspace(0.1,1, 10)
    # lam = np.linspace(1.0, 5.0, 1)
    # lam = np.linspace(14.5, 18, 8)
    work = mpt.JoinableQueue()
    num_workers = 1#len(lam)
    workers = [Worker(work) for i in range(num_workers)]

    for w in workers:
        w.start() 

    for l in (lam):
    # for s in step:
        print(l)
        work.put(LogisticRegressionModelGD(cp.asarray(training_data), cp.asarray(training_labels), cp.asarray(beta_init), cp.asarray(validation_data), cp.asarray(validation_labels), 0.2, l))
        # w = LogisticRegressionModelGD(training_data, training_labels, beta_init, 0.1, l/2)
        # w()
    for i in range(num_workers):
        work.put(None)
      
    work.join()
    # plt.show()

if __name__=="__main__":
    main()
    
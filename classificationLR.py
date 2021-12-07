import numpy as np
# import cupy as np
from numpy.lib.function_base import gradient
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
# np.set_printoptions(threshold=sys.maxsize)
import multiprocessing as mpt


class LogisticRegressionModelGD(object):
    def __init__(self, training_data, training_labels, beta_init, learning_rate=0.1, lam=1.0):
        self.train_data = training_data
        print(training_labels)
        self.train_labels = training_labels
        self.beta = beta_init
        self.lr = learning_rate
        self.lam = lam
        
    def __call__(self):
        self.cross_entropy_error_gradient_descent(self.train_data, self.train_labels, self.beta, self.lr, self.lam)

    # @njit(parallel=True, nopython=True)
    def cross_entropy_error(self, x: np.matrix, training_labels: np.matrix, beta: np.matrix, indicator: np.matrix, reg_term):
        # print(indicator.T @ posterior_prob(x, beta))
        # print(-1/x.shape[0])
        # print(posterior_prob(x,beta))
        error = (-1/x.shape[0]) * ((indicator.T @ np.log(self.posterior_prob(x, beta))) + reg_term * np.linalg.norm(beta)**2)
        return error
    # @njit(nopython=True)
    def posterior_prob(self,x: np.matrix, beta: np.matrix):
        denom = 0.0
    
        for i in range(beta.shape[1]):
            # print(beta[:,i])
            # print(np.exp(x[i].T @ beta[:,i]))
            # print("Mult: {} X {}".format(x[i].T.shape, beta.T[i].shape))
            denom = denom + np.exp(x[i].T @ beta[:,i])
            # print(denom)
        # print(x@beta < 0)
        return (x @ beta) / denom

    def cross_entropy_error_gradient_descent(self,x: np.matrix, training_labels: np.matrix, beta: np.matrix, eta, reg_term):
        indicator = np.matrix(np.zeros((x.shape[0], beta.shape[1])))
        for i in range(x.shape[0]):
            indicator[i,training_labels[i]] = 1
        norm_error = []
        
        grad = -1/x.shape[0] * ((x.T @ (indicator - self.posterior_prob(x, beta))) + 2 * reg_term * beta)
        # while np.norm(grad) > 1:
        print(grad.shape)
        # for i in tqdm(range(700), ascii=True, desc="Lambda: {}".format(reg_term)):
        for i in range(700):
            # print("Beta {}".format(beta))
            grad = -1/x.shape[0] * (x.T @ (indicator - self.posterior_prob(x, beta)) + 2 * reg_term * beta)
            # print(grad.shape)
            # print("Norm Grad: {}".format(np.linalg.norm(grad)))
            # print("Gradient: {}".format(grad >= 0.0))
            beta = beta + eta * grad
            err = np.linalg.norm(self.cross_entropy_error(x, training_labels, beta, indicator, reg_term)**2)
            # print("Norm error: {}".format(err))
            norm_error.append(err)

        plt.plot(norm_error)
        np.savetxt("beta_lam{}.csv".format(reg_term), beta, delimiter=",")
        np.savetxt("norm_error{}.csv".format(reg_term), norm_error, delimiter=",")
        plt.show()
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

    validation_data = (training_data['trainX'].T)[sampler]
    validation_labels = (training_labels['trainL'])[sampler]
    training_data = np.delete(training_data['trainX'].T, sampler, axis=0)
    training_labels = np.delete(training_labels['trainL'], sampler, axis=0)

    aug_train = [[1]] * training_data.shape[0]
    aug_valid = [[1]] * validation_data.shape[0]
    validation_data = np.append(validation_data, aug_valid, axis = 1)
    training_data = np.append(training_data, aug_train, axis = 1)
    # print(aug_train)
    # beta_init = np.random.normal(5, size = (training_data.shape[1], 10))
    beta_init = np.zeros((training_data.shape[1], 10)) + 5
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
    startTraining(training_data, training_labels, beta_init)


def startTraining(training_data: np.matrix, training_labels: np.matrix, beta_init: np.matrix):
    lam = np.linspace(6, 20, 28)
    work = mpt.JoinableQueue()
    num_workers = len(lam)
    workers = [Worker(work) for i in range(num_workers)]
    for w in workers:
        w.start()

    for l in (lam):
        work.put(LogisticRegressionModelGD(training_data, training_labels, beta_init, 0.1, l))
        # w = LogisticRegressionModelGD(training_data, training_labels, beta_init, 0.1, l/2)
        # w()
    for i in range(num_workers):
        work.put(None)
    
    work.join()
    plt.show()

if __name__=="__main__":
    main()
    
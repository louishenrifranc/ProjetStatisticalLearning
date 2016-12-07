import theano.tensor as T
import pandas as pd
import lasagne
from lasagne.layers import InputLayer, BatchNormLayer, DropoutLayer, DenseLayer, ReshapeLayer
import theano
import random
import numpy as np
import time


def read_file():
    data = pd.read_csv('kc_house_data.csv')
    for col in data:
        if col in ['id', 'date', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_renovated', 'zipcode', 'lat', 'long',
                   'sqft_living15', 'sqft_lot15']:
            data = data.drop(col, 1)
    data.to_pickle('data.csv')
    return data


class NN():
    def __init__(self, data):
        self.data = data
        self.N_EXAMPLES = len(data)
        self.BATCH_SIZE = 32
        self.N_INPUT = 9
        self.N_OUTPUT = 1

        X = T.fmatrix('X')
        Y = T.fvector('Y')
        model = {}
        model['l_in'] = InputLayer(input_var=X, shape=(self.BATCH_SIZE, self.N_INPUT))
        model['l_hid1'] = DenseLayer(model['l_in'],
                                     self.N_INPUT * 2)
        model['l_drop1'] = DropoutLayer(model['l_hid1'], 0.3)
        model['l_hid2'] = DenseLayer(model['l_drop1'],
                                     self.N_INPUT)
        model['l_out'] = DenseLayer(model['l_hid2'], 1)

        out = lasagne.layers.get_output(model['l_out'])

        loss = T.square(out - Y)
        cost = T.mean(loss)
        rmse = T.square(cost)

        all_params = lasagne.layers.get_all_params(model['l_out'])
        objectives = lasagne.updates.adadelta(cost, all_params)

        self.train_fn = theano.function([X, Y], cost, updates=objectives)
        self.test_fn = theano.function([X, Y], rmse)

        self.create_batches()

    def create_batches(self):
        self.p_train_examples = 0.7
        train_ind = random.sample(len(self.data), k=(int)(0.7 * len(self.data)))

        X_train = self.data[train_ind, self.data.columns != 'price'].as_matrix().astype(theano.config.floatX)
        Y_train = self.data[train_ind, 'price'].as_matrix().astype(theano.config.floatX)

        self.data = self.data.drop(self.data.index[train_ind])
        X_test = self.data[:, self.data.columns != 'price'].as_matrix().astype(theano.config.floatX)
        Y_test = self.data[:, 'price'].as_matrix().astype(theano.config.floatX)

        self.data = {
            'train': [X_train, Y_train],
            'test': [X_test, Y_test]
        }

    def get_next_batches(self, train_or_test):
        indices = np.arange(len(self.N_EXAMPLES))
        for start_idx in range(0, self.N_EXAMPLES - self.BATCH_SIZE + 1, self.BATCH_SIZE):
            excerpt = indices[start_idx:start_idx + self.BATCH_SIZE]
            yield self.data[train_or_test][0][excerpt, :], self.data[train_or_test][1][excerpt]

    def train(self, train_or_test):
        for epoch in range(100):
            train_err = 0
            n_train_batches = 0
            start_time = time.time()
            for X_batch_train in self.get_next_batches('train'):
                err_train = self.train_fn(theano.shared(X_batch_train))
                train_err += err_train
                n_train_batches += 1

            val_err = 0
            val_rec_err = 0
            n_val_batches = 0
            for X_batch_val in self.get_next_batches('train'):
                err = self.test_fn(theano.shared(X_batch_val))
                val_err += err[0]
                val_rec_err += err[1]
                n_val_batches += 1
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.NUM_EPOCH, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / n_train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / n_val_batches))


if __name__ == '__main__':
    data = read_file()
    neural_network = NN(data)

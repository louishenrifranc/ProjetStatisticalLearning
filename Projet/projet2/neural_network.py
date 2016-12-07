import theano.tensor as T
import pandas as pd
import lasagne
from lasagne.layers import InputLayer, BatchNormLayer, DropoutLayer, DenseLayer, ReshapeLayer
import theano
import random
import os
import cPickle
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def read_file():
    """
    Read the file and remove confusing informations
    :return:
    """
    data = pd.read_csv('kc_house_data.csv')
    data = data.drop(['id', 'date', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_renovated', 'zipcode', 'lat', 'long',
                      'sqft_living15', 'sqft_lot15'], 1)
    for col in data:
        print(data.ix[1:3, col], col)

    data.to_pickle('data.p')
    return data


plt.axis([0, 1000, 150000, 300000])
plt.ion()


def plot(train_loss, test_loss):
    """
    Plot a training loss and testing loss in realtime
    :param train_loss: vector of all previous value taken for the training loss
    :param test_loss: vector of all previous value taken for the testing loss
    :return: .
    """
    plt.scatter(len(train_loss), train_loss[-1], marker='o', c='r', s=0.01)
    plt.scatter(len(test_loss), test_loss[-1], marker='v', c='b', s=0.01)
    plt.plot(np.arange(len(train_loss)), train_loss, c='r', linewidth=1.2, label='train loss')
    plt.plot(np.arange(len(test_loss)), test_loss, c='b', linewidth=1.2, label='test loss')
    plt.pause(0.05)


class NN():
    def __init__(self,
                 data,
                 batch_size):

        self.data = data
        self.N_EXAMPLES = len(data)
        self.BATCH_SIZE = batch_size
        self.N_INPUT = 9
        self.N_OUTPUT = 1

        X = T.fmatrix('X')
        Y = T.fvector('Y')
        dropout_prob = T.scalar('p_drop', dtype=theano.config.floatX)
        rmsprop_ro = T.scalar('ro_rms', dtype=theano.config.floatX)

        # About batch_norm : This layer should be inserted between a linear transformation
        # (such as a DenseLayer, or Conv2DLayer) and its nonlinearity.
        # The convenience function batch_norm() modifies an existing layer to insert batch
        # normalization in front of its nonlinearity.

        # About the ordering of the batch norm and dropout layer:
        # -> MLP_Layer_FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> MLP_Layer_FC ->
        # https://arxiv.org/pdf/1502.03167.pdf
        self.model = {}
        self.model['l_in'] = InputLayer(input_var=X, shape=(self.BATCH_SIZE, self.N_INPUT))

        # Hidden 1
        self.model['l_hid1'] = DenseLayer(self.model['l_in'],
                                          self.N_INPUT * 10)
        self.model['l_hid1'] = lasagne.layers.batch_norm(self.model['l_hid1'])
        self.model['l_hid1'] = DropoutLayer(self.model['l_hid1'], p=dropout_prob)

        # Hidden 2
        self.model['l_hid2'] = DenseLayer(self.model['l_hid1'],
                                          self.N_INPUT * 5)
        self.model['l_hid2'] = lasagne.layers.batch_norm(self.model['l_hid2'])
        self.model['l_hid2'] = DropoutLayer(self.model['l_hid2'], p=dropout_prob)

        # Hidden 3
        self.model['l_hid3'] = DenseLayer(self.model['l_hid2'],
                                          self.N_INPUT * 2)
        self.model['l_hid3'] = lasagne.layers.batch_norm(self.model['l_hid3'])
        # self.model['l_hid3'] = DropoutLayer(self.model['l_hid3'], p=dropout_prob)

        # Hidden 4
        self.model['l_hid4'] = DenseLayer(self.model['l_hid3'],
                                          self.N_INPUT)

        self.model['l_out'] = DenseLayer(self.model['l_hid4'], 1)

        # Loss expression for training
        out = lasagne.layers.get_output(self.model['l_out'])
        out = T.reshape(out, (out.shape[0],))
        loss = T.mean(T.sqr(T.sub(out, Y)), axis=0)
        l2_reg = lasagne.regularization.regularize_network_params({self.model['l_out']: 0.01},
                                                                  penalty=lasagne.regularization.l2)
        loss += l2_reg
        all_params = lasagne.layers.get_all_params(self.model['l_out'], trainable=True)
        updates = lasagne.updates.rmsprop(loss, all_params, rho=rmsprop_ro)

        # Loss expression for testing
        out_test = lasagne.layers.get_output(self.model['l_out'], deterministic=True)
        out_test = T.reshape(out_test, (out_test.shape[0],))
        loss_test = T.mean(T.sqr(T.sub(out_test, Y)), axis=0)
        rmse_test = T.sqrt(loss_test)

        self.train_fn = theano.function([X, Y, dropout_prob, rmsprop_ro], [loss], updates=updates)
        self.test_fn = theano.function([X, Y], rmse_test)

        self.create_batches()

    def create_batches(self):
        self.p_train_examples = 0.8
        train_ind = random.sample(xrange(len(self.data)), k=(int)(self.p_train_examples * len(self.data)))
        X_train = self.data.ix[train_ind, self.data.columns != 'price'].as_matrix().astype(theano.config.floatX)

        Y_train = self.data.ix[train_ind, self.data.columns == 'price'].as_matrix().astype(theano.config.floatX)
        Y_train = np.reshape(Y_train, (Y_train.shape[0],))

        self.data = self.data.drop(self.data.index[train_ind])
        X_test = self.data.ix[:, self.data.columns != 'price'].as_matrix().astype(theano.config.floatX)
        Y_test = self.data.ix[:, self.data.columns == 'price'].as_matrix().astype(theano.config.floatX)
        Y_test = np.reshape(Y_test, (Y_test.shape[0],))

        self.data = {
            'train': [X_train, Y_train],
            'test': [X_test, Y_test]
        }

    def get_next_batches(self, train_or_test):
        nb_examples = np.shape(self.data[train_or_test][0])[0]
        indices = np.arange(nb_examples)
        np.random.shuffle(indices)
        for start_idx in range(0, nb_examples - self.BATCH_SIZE + 1, self.BATCH_SIZE):
            excerpt = indices[start_idx:start_idx + self.BATCH_SIZE]
            yield self.data[train_or_test][0][excerpt], self.data[train_or_test][1][excerpt]

    def save_model(self, output_layer, filename='model1.p'):
        """Pickels the parameters within a Lasagne model."""
        data = lasagne.layers.get_all_param_values(output_layer)
        filename = os.path.join('./', filename)
        filename = '%s.%s' % (filename, 'params')
        with open(filename, 'w') as f:
            cPickle.dump(data, f)

    def load_model(self, output_layer, filename='model.p'):
        """Unpickles and loads parameters into a Lasagne model."""
        filename = os.path.join('./', '%s.%s' % (filename, 'params'))
        with open(filename, 'r') as f:
            data = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, data)

    def train(self):
        self.NUM_EPOCH = 1000
        train_loss = []
        test_loss = []
        p_drop = 0.2
        ro_rms = 0.9

        time_patience = 10
        is_improving = False
        max_time_patience = 30
        max_time_improving = 5
        max_improvement_patience = 0.3

        for epoch in range(self.NUM_EPOCH):
            train_err = 0
            n_train_batches = 0
            start_time = time.time()
            for batch_train in self.get_next_batches('train'):
                X_batch_train, y_batch_train = batch_train
                #  print(X_batch_train.shape, y_batch_train.shape)

                err_train = self.train_fn(X_batch_train, y_batch_train, p_drop, ro_rms)
                # if random.random() < 0.05:
                #    print(err_train[1], err_train[2], err_train[3])
                train_err += err_train[0]
                n_train_batches += 1

            val_rmse = 0
            val_rec_err = 0
            n_val_batches = 0
            for batch_val in self.get_next_batches('train'):
                X_batch_val, y_batch_val = batch_val
                err = self.test_fn(X_batch_val, y_batch_val)
                val_rmse += err
                n_val_batches += 1

            # Patience and then improve
            new_train_error = np.sqrt(train_err / n_train_batches)
            if len(train_loss) > 5:
                if min(train_loss) < new_train_error and not is_improving:
                    time_patience -= 1
                    #print('Decrease patience %d' % time_patience)
                    if time_patience == 0:
                        time_patience = max_time_improving
                        is_improving = True
                        ro_rms -= max_improvement_patience
                        #print('[Update] Increasing ro %f' % ro_rms)
                elif min(train_loss) > new_train_error and not is_improving and time_patience != 10:
                    time_patience = 10
                    #print('Found new minimum, reset patience')
                if is_improving:
                    time_patience -= 1
                    ro_rms += (max_improvement_patience / (max_time_improving))
                    if time_patience == 0:
                        time_patience = max_time_patience
                        is_improving = False
                        #print('End patience %f' % ro_rms)
                    #print('Still ro_rms = %f' % ro_rms)

            train_loss.append(new_train_error)
            test_loss.append(val_rmse / n_val_batches)
            # plot(train_loss, test_loss)
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, self.NUM_EPOCH, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(new_train_error))
            print("  validation loss:\t\t{:.6f}\n".format(val_rmse / n_val_batches))

            if test_loss[-1] == min(test_loss):
                self.save_model(self.model['l_out'])
                print('\t\tModel saved')

                # Increasing dropout over time
                # Zp_drop += (0.4 / self.NUM_EPOCH)

        cPickle.dump(train_loss, open('train_loss1', 'wb'))
        cPickle.dump(test_loss, open('test_loss1', 'wb'))
        plt.savefig('foo.png')


if __name__ == '__main__':
    # read_file()
    data = pd.read_pickle('data.p')
    neural_network = NN(data, batch_size=32)
    neural_network.train()

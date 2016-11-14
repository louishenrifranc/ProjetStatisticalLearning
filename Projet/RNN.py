from __future__ import print_function
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
from lasagne.layers import InputLayer, LSTMLayer, DenseLayer, ReshapeLayer, DimshuffleLayer


class RNN(object):
    def __init__(self):

        # Input data
        self.sentences = cPickle.load(open('data/numberPickle2.py', 'rb'))
        self.embeddings = cPickle.load(open('data/index_to_embedding1.p', 'rb'))

        # Size of batch,
        # Must be a multiple of 25 because there are 25 headlines per day
        self.SIZE_BATCH = 25

        # Number of sentences per words
        self.SIZE_EXAMPLE = 25
        # Dimension space of the latent variable
        self.N_EMBEDDING = len(self.embeddings[1])
        self.N_EXAMPLE_PER_BATCH = self.SIZE_BATCH / self.SIZE_EXAMPLE
        assert self.N_EXAMPLE_PER_BATCH % 1 == 0, "wrong batch size"
        # Length of the longest sentence
        self.MAX_LENGTH = self.get_max_lenth()

        # Number of epoch
        self.N_EPOCH = 30

        # Size of the hidden space of the RNN
        self.N_HIDDEN = self.N_EMBEDDING

        self.target = T.vector('y')
        # We build the network starting at the input layer
        l_in = InputLayer(shape=(self.SIZE_BATCH, self.MAX_LENGTH, self.N_EMBEDDING))

        # The network also needs a way to provide a mask for each sequence.  We'll
        # use a separate input layer for that.  Since the mask only determines
        # which indices are part of the sequence for each batch entry, they are
        # supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
        l_mask = InputLayer(shape=(self.SIZE_BATCH, self.MAX_LENGTH))

        l_forward = LSTMLayer(l_in, self.N_HIDDEN, mask_input=l_mask,
                              only_return_final=True, grad_clipping=100)

        l_backward = LSTMLayer(l_in, self.N_HIDDEN, mask_input=l_mask,
                               only_return_final=True,
                               backwards=True,
                               grad_clipping=100)

        l_shp = lasagne.layers.ConcatLayer([l_forward, l_backward])

        l_mlp1 = lasagne.layers.DenseLayer(l_shp, num_units=50,
                                           nonlinearity=lasagne.nonlinearities.rectify)
        l_drop = lasagne.layers.DropoutLayer(l_mlp1, p=0.2)

        l_out = lasagne.layers.DenseLayer(l_drop, num_units=1,
                                          nonlinearity=lasagne.nonlinearities.linear)

        network_output = lasagne.layers.get_output(l_out)

        self.all_params = lasagne.layers.get_all_params(l_out)
        self.cost = T.mean(T.square(network_output - self.target))
        self.updates = lasagne.updates.adadelta(self.cost, self.all_params)

        self.train_fn = theano.function([l_in.input_var, self.target, l_mask.input_var], [self.cost, network_output],
                                        updates=self.updates, on_unused_input='ignore')

        self.test_fn = theano.function([l_in.input_var, self.target, l_mask.input_var],
                                       [self.cost, network_output], on_unused_input='ignore')

    def train(self):
        print('Start training')

        for epoch in range(self.N_EPOCH):
            X_val, y_val, mask_val = self.gen_data(newEpoch=True)
            start_time = time.time()
            nb_iter_train = 0
            while True:
                X_train, y_train, mask_train = self.gen_data()
                if X_train is None:
                    break
                nb_iter_train += 1

                cost, pred = self.train_fn(X_train, y_train, mask_train)
                # print('Iter ', nb_iter_train, ' : ', cost, np.transpose(pred), y_train)
                if nb_iter_train % 50 == 0:
                    cost, predict = self.test_fn(X_val, y_val, mask_val)
                    mean = [np.mean(predict[i * self.SIZE_EXAMPLE:(i + 1) * self.SIZE_EXAMPLE]) for i in
                            range(self.N_EXAMPLE_PER_BATCH)]
                    output = [y_val[i * self.SIZE_EXAMPLE] for i in range(self.N_EXAMPLE_PER_BATCH)]

                    print('Epoch', epoch, 'Iter %d: ', nb_iter_train, 'cost', cost, 'predicted', mean,
                          '\n\t\ttrue label', output)
                    start_time = time.time()

            test_cost = 0
            nb_iter_test = 0
            while True:
                X_test, y_test, mask_test = self.gen_data(train_phase=False)
                if X_test is None:
                    test_cost /= nb_iter_test
                    print('Epoch', epoch, 'Test mean error', test_cost)

                    break
                test_cost += self.test_fn(X_test, y_test, mask_test)[0]
                nb_iter_test += 1

    def get_max_lenth(self):
        """
        Get the maximum length from all sentences in the corpus
        :return: the maximum length
        """
        max_length = 0
        for column in self.sentences:
            if column != 'Date' and column != 'Label' and column != 'Combined':
                for index, _ in self.sentences.iterrows():
                    max_length = max(max_length, len(self.sentences[column][index]))
        return max_length

    def gen_data(self,
                 newEpoch=False,
                 test_set_proportion=0.1,
                 train_phase=True
                 ):
        if newEpoch == True:
            self.index_shuf = range(len(self.sentences))
            np.random.shuffle(self.index_shuf)
            self.current_index = 0

        if train_phase and self.current_index > (1 - test_set_proportion) * len(self.sentences):
            return None, None, None
        if not train_phase and self.current_index >= len(self.sentences):
            return None, None, None
        else:
            X_value = np.zeros((self.SIZE_BATCH, self.MAX_LENGTH, self.N_EMBEDDING), dtype="float32")
            mask = np.zeros((self.SIZE_BATCH, self.MAX_LENGTH), dtype="float32")
            i = 0
            L = int(self.SIZE_BATCH / self.SIZE_EXAMPLE)
            y_value = np.zeros((self.SIZE_BATCH), dtype="float32")
            for b in range(L):
                embedding = self.sentences.ix[self.current_index, 2:(self.SIZE_EXAMPLE + 2)]
                for e in embedding:
                    for j in range(len(e)):
                        X_value[i, j] = self.embeddings[e[j]]
                    mask[i, :len(e)] = 1
                    y_value[i] = self.sentences['Label'][self.current_index]
                    i += 1
                self.current_index += 1
            return X_value, y_value, mask


if __name__ == '__main__':
    rnn = RNN()
    rnn.train()

from lasagne.layers import Layer, InputLayer, DropoutLayer, DenseLayer, NonlinearityLayer, ElemwiseMergeLayer, SliceLayer, ConcatLayer, get_output, get_all_params
from lasagne.objectives import categorical_crossentropy, squared_error, aggregate
from lasagne.nonlinearities import softmax, rectify, sigmoid
from lasagne.updates import sgd, adagrad, adam, adadelta, apply_nesterov_momentum
from lasagne.init import Constant
import theano.tensor as T
import theano
import numpy as np

class NeuralForestLayer(Layer):
    def __init__(self, incoming, depth, n_estimators, n_outputs, pi_iters, **kwargs):
        self._incoming = incoming
        self._depth = depth
        self._n_estimators = n_estimators
        self._n_outputs = n_outputs
        self._pi_iters = pi_iters
        super(NeuralForestLayer, self).__init__(incoming, **kwargs)

        pi_init = Constant(val=1.0 / n_outputs)(((1 << (depth - 1)) * n_estimators, n_outputs))
        pi_name = "%s.%s" % (self.name, 'pi') if self.name is not None else 'pi'
        self.pi = theano.shared(pi_init, name=pi_name)

        # what we want to do here is pi / pi.sum(axis=1)
        # to be safe, if certain rows only contain zeroes (for some pi all y's became 0),
        #     replace such row with 1/n_outputs
        sum_pi_over_y = self.pi.sum(axis=1).dimshuffle(0, 'x')
        all_0_y = T.eq(sum_pi_over_y, 0)
        norm_pi_body = (self.pi + all_0_y * (1.0 / n_outputs)) / (sum_pi_over_y + all_0_y)
        self.normalize_pi = theano.function([], [], updates=[(self.pi, norm_pi_body)])
        self.update_pi_one_iter = self.get_update_pi_one_iter_func()

        self.normalize_pi()

        t_input = T.matrix('t_input')
        self.f_leaf_proba = theano.function([t_input], self.get_probabilities_for(get_output(incoming, t_input)))

    def update_pi(self, X, y):
        leaf_proba = self.f_leaf_proba(X)

        for i in range(self._pi_iters):
            self.update_pi_one_iter(leaf_proba, y)
            self.normalize_pi()

    def get_update_pi_one_iter_func(self):
        proba = T.matrix('proba')
        y = T.matrix('y')

        # reshape and dimshuffle everything to (samples, leaves, estimators, outputs)
        n_leaves = 1 << (self._depth - 1)
        pi_shaped = self.pi.reshape((n_leaves, self._n_estimators, self._n_outputs)).dimshuffle('x', 0, 1, 2)
        proba_shaped = proba.reshape((-1, n_leaves, self._n_estimators)).dimshuffle(0, 1, 2, 'x')
        y_shaped = y.dimshuffle(0, 'x', 'x', 1)

        # compute the nominator and denominator for the iteration
        common = pi_shaped * proba_shaped
        nominator = common * y_shaped
        denominator = common.sum(axis=1).dimshuffle((0, 'x', 1, 2)) # sum over leaves, and broadcast the sum
        denominator = denominator + T.eq(denominator, 0)
        result = (nominator / denominator).sum(0).reshape((n_leaves * self._n_estimators, self._n_outputs))

        return theano.function([proba, y], [], updates=[(self.pi, result)])

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self._n_outputs)

    def get_params(self, trainable=False):
        return [] # intentionally not returning pi

    def get_bias_params(self, trainable=False):
        return []

    def get_probabilities_for(self, input):
        lastOffset = 0
        nextOffset = self._n_estimators
        lastTensor = input[:,0:self._n_estimators]
        for i in range(self._depth - 1):
            lastWidth = (1 << i) * self._n_estimators
            lastOffset, midOffset, nextOffset = nextOffset, nextOffset + lastWidth, nextOffset + lastWidth * 2
            leftTensor = input[:,lastOffset:midOffset]
            rightTensor = input[:,midOffset:nextOffset]

            leftProduct = lastTensor * leftTensor
            rightProduct = (1 - lastTensor) * rightTensor

            lastTensor = T.concatenate([leftProduct, rightProduct], axis=1)
            
        return lastTensor

    def get_output_for(self, input, *args, **kwargs):
        p = self.get_probabilities_for(input)
        return p.dot(self.pi)



from lasagne.layers import Layer, InputLayer, DropoutLayer, DenseLayer, NonlinearityLayer, ElemwiseMergeLayer, SliceLayer, ConcatLayer, get_output, get_all_params
from lasagne.objectives import categorical_crossentropy, squared_error, aggregate
from lasagne.nonlinearities import softmax, rectify, sigmoid
from lasagne.updates import sgd, adagrad, adam, adadelta, apply_nesterov_momentum
from lasagne.init import Constant
import theano.tensor as T
import theano
import numpy as np

from utils import BatchIterator
from neuralforestlayer import NeuralForestLayer

__DEBUG_NO_FOREST__ = False

class ShallowNeuralForest:
    def __init__(self, n_inputs, n_outputs, regression, multiclass=False, depth=5, n_estimators=20, n_hidden=128, learning_rate=0.01, num_epochs=500, pi_iters=20, sgd_iters=10, batch_size=1000, momentum=0.0, dropout=0.0, loss=None, update=adagrad):
        """
        Parameters
        ----------
        n_inputs : number of input features
        n_outputs : number of classes to predict (1 for regression)
            for 2 class classification n_outputs should be 2, not 1
        regression : True for regression, False for classification
        multiclass : not used
        depth : depth of each tree in the ensemble
        n_estimators : number of trees in the ensemble
        n_hidden : number of neurons in the hidden layer
        pi_iters : number of iterations for the iterative algorithm that updates pi
        sgd_iters : number of full iterations of sgd between two consequtive updates of pi
        loss : theano loss function. If None, squared error will be used for regression and
            cross entropy will be used for classification
        update : theano update function
        """
        self._depth = depth
        self._n_estimators = n_estimators
        self._n_hidden = n_hidden
        self._n_outputs = n_outputs
        self._loss = loss
        self._regression = regression
        self._multiclass = multiclass
        self._learning_rate = learning_rate
        self._num_epochs = num_epochs
        self._pi_iters = pi_iters
        self._sgd_iters = sgd_iters
        self._batch_size = batch_size
        self._momentum = momentum
        self._update = update

        self.t_input = T.matrix('input')
        self.t_label = T.matrix('output')

        self._cached_trainable_params = None
        self._cached_params = None

        self._n_net_out = n_estimators * ((1 << depth) - 1)

        self.l_input = InputLayer((None, n_inputs))
        self.l_dense1 = DenseLayer(self.l_input, self._n_hidden, nonlinearity=rectify)
        if dropout != 0:
            self.l_dense1 = DropoutLayer(self.l_dense1, p=dropout)
        if not __DEBUG_NO_FOREST__:
            self.l_dense2 = DenseLayer(self.l_dense1, self._n_net_out, nonlinearity=sigmoid)
            self.l_forest = NeuralForestLayer(self.l_dense2, self._depth, self._n_estimators, self._n_outputs, self._pi_iters)
        else:
            self.l_forest = DenseLayer(self.l_dense1, self._n_outputs, nonlinearity=softmax)

    def _create_functions(self):
        self._update_func = self._update(self._get_loss_function(), self._get_all_trainable_params(), self._learning_rate)
        if self._momentum != 0:
            self._update_func = apply_nesterov_momentum(self._update_func, self._get_all_trainable_params(), self._momentum)
        self._loss_func = self._get_loss_function()
        self._train_function = theano.function([self.t_input, self.t_label], self._get_loss_function(), updates=self._update_func)

    def fit(self, X, y, X_val = None, y_val = None, on_epoch = None, verbose = False):
        """ Train the model

        Parameters
        ----------
        X : input vector for the training set
        y : output vector for the training set. Onehot is required for classification
        X_val : if not None, input vector for the validation set
        y_val : it not None, input vector for the validation set
        on_epoch : a callback that is called after each epoch
            if X_val is None, the signature is (epoch, training_error, accuracy)
            if X_val is not None, the signature is (epoch, training_error, validation_error, accuracy)
            on iterations that update pi the training error is reported for the previous iteration
        verbose : if True, spams current step on each epoch
        """
        self._create_functions()

        X = X.astype(np.float32)
        y = y.astype(np.float32)
        self._x_mean = np.mean(X, axis=0)
        self._x_std = np.std(X, axis=0)
        self._x_std[self._x_std == 0] = 1
        X = (X - self._x_mean) / self._x_std
        if y_val is not None:
            assert X_val is not None
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.float32)
            X_val = (X_val - self._x_mean) / self._x_std

        if X_val is not None:
            assert y_val is not None

            predictions = self._predict_internal(self._get_output())
            accuracy = T.mean(T.eq(predictions, self._predict_internal(self.t_label)))

            test_function = theano.function([self.t_input, self.t_label], [self._get_loss_function(), accuracy])

        iterator = BatchIterator(self._batch_size)

        loss = 0
        for epoch in range(self._num_epochs):

            # update the values of pi
            if not __DEBUG_NO_FOREST__ and epoch % self._sgd_iters == 0:
                if verbose: print "updating pi"
                self.l_forest.update_pi(X, y)
                if verbose: print "recreating update funcs"
                self._create_functions()

            else:
                if verbose: print "updating theta"
                loss = 0
                deno = 0
                # update the network parameters
                for Xb, yb in iterator(X, y):
                    loss += self._train_function(Xb, yb)
                    deno += 1

                loss /= deno

            if X_val is not None:
                tloss = 0
                accur = 0
                deno = 0
                iterator = BatchIterator(self._batch_size)
                for Xb, yb in iterator(X_val, y_val):
                    tl, ac = test_function(Xb, yb)
                    tloss += tl
                    accur += ac
                    deno += 1
                tloss /= deno
                accur /= deno

            if on_epoch is not None:
                if X_val is None:
                    on_epoch(epoch, loss)
                else:
                    on_epoch(epoch, loss, tloss, accur)

        return self

    def _predict_internal(self, y):
        if not self._regression and not self._multiclass:
            return y.argmax(axis=1)
        else:
            return y >= 0.5

    def predict(self, X):
        ret = self.predict_proba(X)
        return self._predict_internal(ret)

    def predict_proba(self, X):
        X = X.astype(np.float32)
        X = (X - self._x_mean) / self._x_std
        predict_function = theano.function([self.t_input], self._get_output())
        return predict_function(X)

    def _get_loss_function(self):
        # TODO: remove `or True`
        if self._loss is None:
            if self._regression:
                self._loss = squared_error
            else:
                self._loss = categorical_crossentropy
        return aggregate(self._loss(self._get_output(), self.t_label), mode='mean')

    def _get_output(self):
        return get_output(self.l_forest, self.t_input)

    def _get_all_trainable_params(self):
        if self._cached_trainable_params is None:
            self._cached_trainable_params = get_all_params(self.l_forest, trainable=True)
        return self._cached_trainable_params
    
    def _get_all_params(self):
        if self._cached_params is None:
            self._cached_params = get_all_params(self.l_forest)
        return self._cached_params

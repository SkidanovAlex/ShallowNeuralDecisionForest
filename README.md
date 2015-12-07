# ShallowNeuralDecisionForest
Implementation of Deep Neural Decision Forest based on MSR paper:

http://research.microsoft.com/pubs/255952/ICCV15_DeepNDF_main.pdf

It is built using Theano and Lasagne.

Layer
-----
neuralforestlayer.py implements a lasagne layer for the deep neural decision forest.

One important note (that took me a while to figure out) is that if the `update` function maintains some state with each parameters (for instance, the momentum, or learning rate), those states don't necessarily make sense after the PI is updated. To go around it one can either not use an update step that maintains such state (sgd without momentum), or recreate the update functions (and therefore reset their state) every time PI is updated.

It is the responsibility of the model to call `update_pi` or a regular basis.

Model
-----
neuralforest.py implements the entire shallow neural decision forest model with one hidden layer. It uses `adagrad` as an update function by default, and therefore has to recreate the update function each time it recomputes the PI.

Optimizations
-------------
The probabilities are returned from the network in an order that allows to evaluate the ensemble on the GPU. If the tree looks like this:

```
        1
     /     \
    2       3
   / \     / \
  4   5   6   7
```

then the layer returns probabilities in order: `1 2 3 4 6 5 7`. Note that 4 and 6 (left children of the layer) both go before 5 and 7 (right children on the same layer). Moreover, all the probabilities for a given node in all the trees are grouped together. So, if we have three trees, `A`, `B` and `C`, the probabilities vector will contain probablities `A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4 A6 B6 C6 A5 B5 C5 A7 B7 C7`. This way we can compute probabilties on each layer for all the trees at once in two multiplications:

    accumulated_layer_prob[layer][:mid] = layer_prob[layer][:mid] * accumulated_layer_prob[layer-1]
    accumulated_layer_prob[layer][mid:] = layer_prob[layer][mid:] * (1 - accumulated_layer_prob[layer-1])

Results
-------
For most of the tasks I tried this model on it performs on par with a shallow neural network, however it always requires different hyper-parameters (size of hidden layer, momentum, dropout) than the neural network. With `sgd` the convergence always takes significantly longer than for a shallow NN, but with more sophisticated update functions, such as `adagrad`, convergence happens quickly (but still slower than for a NN).

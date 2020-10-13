class Node (object):
    '''represents a node that outputs a function applied to the sum of the
    inputs multiplied by different weights with a bias applied'''
    def __init__(self, weights, bias, function):
        '''initializes Node with weights and bias and sets output function'''
        self.w = weights
        self.b = bias
        self.f = function
        self.input = 0

    def output(self, inputs):
        '''returns node output based on inputs and stores input'''
        #add checks for inputs matching weights
        self.input = 0
        for i in range(len(inputs)):
            self.input += inputs[i] * self.w[i]
        self.input += self.b
        return self.f.at(self.input)

    def grad_w(self, inputs, d_node, step):
        '''calculates and applies partial derivatives of Error with respect to
        weights based off last inputs and partial derivative of Error with
        respect to this node's output (d_node). Must be done after an output'''
        #add checks for inputs matching weights
        for i, x in enumerate(inputs):
            self.w[i] -= step * x * self.f.derivative(self.input) * d_node

    def grad_b(self, d_node, step):
        '''calculates and applies partial derivatives of Error with respect to
        bias based off last inputs and partial derivative of Error with
        respect to this node's output (d_node). Must be done after an output'''
        self.b -= step * self.f.derivative(self.input) * d_node

    def back_prop(self, d_node):
        '''outputs a list of partial derivatives of Error wrt every input.
        Should be done BEFORE updating gradients'''
        back_prop_d = []
        for w in self.w:
            back_prop_d.append(w*self.f.derivative(self.input))
        return back_prop_d

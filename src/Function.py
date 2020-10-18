import math
class Function (object):
    '''super class for node functions'''
    def __init__():
        pass
    
    def at(x):
        '''evaluates function at x'''
        pass

    def derivative(x):
        '''evaluates the derivative of function at x'''
        pass

class Linear (Function):
    '''y = x'''
    def at(x):
        return x

    def derivative(x):
        return 1

class ReLu (Function):
    '''linear if x > 0'''
    def at(x):
        return max(x,0)

    def derivative(x):
        return int(x > 0)

class Sigmoid (Function):
    '''normalized sigmoid function'''
    def at(x):
        return (1/(1+math.exp(-x)))*2 - 1

    def derivative(x):
        return 2*math.exp(-x)/math.pow(1+math.exp(-x),2)

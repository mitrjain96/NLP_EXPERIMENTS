import theano
from theano import tensor as T
from theano import printing as printf
from theano import function
import lasagne

def softmax(x):
    e_x = T.exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out

def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])


def constant_param(value=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Constant(value).sample(shape), borrow=True)
    
def normal_param(std=0.1, mean=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Normal(std, mean).sample(shape), borrow=True)

def cosine_similarity(A,B):
	return T.dot(A,T.transpose(B))/(T.dot(A,T.transpose(A))*T.dot(B,T.transpose(B)))

def cosine_proximity_loss(A,B):
	return (1 - cosine_similarity(A,B))

def print_shape(A):
	print_op = printf.Print('vector',attrs=['shape'])
	printed = print_op(A)
	f = function([A],printed)
	return f(A)

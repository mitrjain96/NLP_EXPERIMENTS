import theano
import utils
import nn_utils
import lasagne
import numpy as np
from theano import tensor as T

class RNN(object):
    def __init__(self,hidden_nodes,vocab_size):
        self.hid_dim=hidden_nodes
        self.vocab_size=vocab_size

        self.W=nn_utils.normal_param(std=0.1, shape=(self.hid_dim, self.hid_dim))
        self.U=nn_utils.normal_param(std=0.1, shape=(self.hid_dim,self.vocab_size))
        self.V=nn_utils.normal_param(std=0.1, shape=(self.vocab_size,self.hid_dim))
        self.dummy=nn_utils.constant_param(value=0.0, shape=(self.hid_dim,))
        self.params=[self.U,self.V,self.W]
        self.wordInd=[]

        sent1=T.wvector('X') # X=[xt,xt+1......]
        expected_vals=T.wvector('Y') # Y=[yt,yt+1,.....]
        st_s,_=theano.scan(fn=self.computation,sequences=[sent1],outputs_info=[T.zeros_like(self.dummy)])
        ot_s,_=theano.scan(fn=self.predict,sequences=[st_s])
        losses,_=theano.scan(fn=self.loss,sequences=[ot_s,expected_vals])
        self.avg_loss=T.sum(losses)/expected_vals.shape[0]
        updates=lasagne.updates.adadelta(self.avg_loss,self.params)
        self.train=theano.function([sent1,expected_vals],[],updates=updates, on_unused_input='warn')

    def computation(self,ind,prev_hid_state):
        U_xt=self.U[:,ind]
        st=T.tanh(U_xt + T.dot(self.W,prev_hid_state))
        return st

    def loss(self,val1,val2):
        return T.sqrt(abs(T.square(val1)-T.square(val2)))

    def predict(self,st):
        ot=nn_utils.softmax(T.dot(self.V,st))
        return T.argmax(ot)

    def trainX(self,train_x,train_y):
        for num in range(len(train_x)):
            print(np.array(train_x[num]).shape)
            print(train_y[num])
            self.train(train_x[num],train_y[num])

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 19:13:18 2018

@author: igeh
"""

from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def show_x(x):
    img = Image.fromarray(x.numpy().reshape(img_shape))
    img.show()

def classify(i, p):
    return p



img_shape = (16, 16)

mat = scipy.io.loadmat('usps_all.mat')
raw_data = mat['data']

Nseg = 5
Nset = 400
Nclasses = 10

dtype = torch.float64
test_data = np.zeros((Nseg, 256, Nset//Nseg, Nclasses), dtype=np.double)
train_data = np.zeros((Nseg, 256, Nset - Nset//Nseg, Nclasses), dtype=np.double)
class_data = np.zeros((Nseg, Nclasses, Nset - Nset//Nseg, Nclasses), dtype=np.double)
bin_data = raw_data.copy()

class_data_test = np.zeros((Nset, 10, 10), dtype=np.int)
for i in range(Nset):
    for c in range(Nclasses):
        y_hat = np.zeros(10, np.int)
        y_hat[c] = 1
        class_data_test[i,c] = y_hat

train_data = torch.from_numpy( bin_data[:, 0:Nset, :]).permute((1,2,0)).flatten(0,1).type(dtype)

class_data_test = torch.from_numpy( class_data_test).type(dtype).permute((0,1,2)).flatten(0,1)

#print(train_data.shape)
#print(class_data.shape)
#raise Exception

dtype = torch.float64
N, D_in, H, D_out = Nclasses*Nset, 256, 100, 10

x = Variable(train_data.type(dtype), requires_grad=False)
y = Variable(class_data_test.type(dtype), requires_grad=False)

b = Variable(0.01*torch.randn(D_in).type(dtype), requires_grad=True)
w1 = Variable(0.1*torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(0.4*torch.randn(H, D_out).type(dtype), requires_grad=True)


learning_rate = 0.07
for t in range(N):
    x_t = Variable(x[t].data, requires_grad=False)
    y_pred = x_t.add(b).matmul(w1).sigmoid().matmul(w2).sigmoid()

    loss = (y_pred - y[t]).pow(2).mean()
    
    print(t, loss.data)

    loss.backward()

    b.data  -= learning_rate * b.grad.data
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    #print(w1.grad.norm())
    b.grad.data.zero_()
    w1.grad.data.zero_()
    w2.grad.data.zero_()


#y_pred = x.add(b).matmul(w1).sigmoid().mm(w2).sigmoid()
#ans = torch.argmax(y, dim=1)
#ans_p = torch.argmax(y_pred, dim=1)
#
#valid = ((ans_p - ans) == 0).sum()
#print(valid)

#y_test = test_data.mm(w1).tanh().mm(w2).tanh()
#(M, _) = y_test.shape
#for j in range(M):
#    
#y_pred = x.mm(w1).clamp(min=0).mm(w2).tanh()


#show_x(test_data.permute((2,0,3,1))[1,2,0].numpy())
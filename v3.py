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


def data(dataset, i, j):
    """ возвращает i-тый вектор j-того символа. """
    return dataset[:,i,j].reshape((16,16)).transpose()

def show_img(dataset, i, j):
    """ рисует i-тое изображение j-того символа. """
    img = Image.fromarray(data(dataset, i, j), 'L')
    img.show()

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
#test_data = bin_data[:, Nset:500, :]

train_data = torch.from_numpy( bin_data[:, 0:Nset, :]).permute((1,2,0)).flatten(0,1)

class_data_test = torch.from_numpy( class_data_test).type(dtype).permute((0,1,2)).flatten(0,1)

#test_data = torch.from_numpy(test_data).permute((0,3,2,1)).flatten(start_dim=0,end_dim=2)

#train_data = torch.from_numpy(train_data).permute((0,3,2,1)).flatten(start_dim=0,end_dim=2)

#class_data = torch.from_numpy(class_data).permute((0,3,2,1)).flatten(start_dim=0,end_dim=2)



#print(train_data.shape)
#print(class_data.shape)
#raise Exception

dtype = torch.float64
N, D_in, H, D_out = Nclasses*Nset, 256, 20, 10

x = Variable(train_data.type(dtype), requires_grad=False)
y = Variable(class_data_test.type(dtype), requires_grad=False)

#b = Variable(torch.randn(D_in).type(dtype), requires_grad=True)
#w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
#w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)



learning_rate = 0.005
for t in range(10000):
    y_pred = x.add(b).mm(w1).tanh().mm(w2).sigmoid()

    #ans = torch.argmax(y, dim=1)
    #ans_p = torch.argmax(y_pred, dim=1)
    
    #nValid = ((ans_p - ans) == 0).sum()

    loss = (y_pred - y).pow(2).mean()
    print(t, loss.data)

    loss.backward()
    

    b.data -= learning_rate * b.grad.data
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    #print(b.grad.norm())
    b.grad.data.zero_()
    w1.grad.data.zero_()
    w2.grad.data.zero_()


y_pred = x.add(b).mm(w1).tanh().mm(w2).sigmoid()
ans = torch.argmax(y, dim=1)
ans_p = torch.argmax(y_pred, dim=1)

valid = ((ans_p - ans) == 0).sum()        
print(valid)

#y_test = test_data.mm(w1).tanh().mm(w2).tanh()
#(M, _) = y_test.shape
#for j in range(M):
#    
#y_pred = x.mm(w1).clamp(min=0).mm(w2).tanh()


#show_x(test_data.permute((2,0,3,1))[1,2,0].numpy())
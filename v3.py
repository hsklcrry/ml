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


img_shape = (16, 16)

mat = scipy.io.loadmat('usps_all.mat')
raw_data = mat['data']

Nseg = 5
Nset = 400
Mset = 100
Nclasses = 10

bin_data = raw_data.copy()

train_class = np.zeros((Nset, 10, 10), dtype=np.int)
test_class = np.zeros((Mset, 10, 10), dtype=np.int)
for i in range(Nset):
    for c in range(Nclasses):
        y_hat = np.zeros(10, np.int)
        y_hat[c] = 1
        train_class[i,c] = y_hat

for i in range(Mset):
    for c in range(Nclasses):
        y_hat = np.zeros(10, np.int)
        y_hat[c] = 1
        test_class[i,c] = y_hat


dtype = torch.float64
train_data = torch.from_numpy( bin_data[:, 0:Nset, :]).permute((1,2,0)).flatten(0,1).type(dtype)
test_data = torch.from_numpy( bin_data[:, Nset:(Nset + Mset), :]).permute((1,2,0)).flatten(0,1).type(dtype)

test_class = torch.from_numpy( test_class).type(dtype).permute((0,1,2)).flatten(0,1)
train_class = torch.from_numpy( train_class).type(dtype).permute((0,1,2)).flatten(0,1)


N, D_in, H1, H2, D_out = Nclasses*Nset, 256, 500, 100, 10

x = Variable(train_data.type(dtype), requires_grad=False)
y = Variable(train_class.type(dtype), requires_grad=False)

b = Variable(0.09*torch.randn(H1).type(dtype), requires_grad=True)
w1 = Variable(0.02*torch.randn(D_in, H1).type(dtype), requires_grad=True)
w2 = Variable(0.08*torch.randn(H1, H2).type(dtype), requires_grad=True)
w3 = Variable(0.2*torch.randn(H2, D_out).type(dtype), requires_grad=True)

def forward(x, b):
    return x.matmul(w1).add(b).tanh().matmul(w2).tanh().matmul(w3).sigmoid()

learning_rate = 0.01
k = 0
epochs = 10
for epoch in range(epochs):
    for (x_t, y_t) in zip(x, y):
        x1 = Variable(x_t.data, requires_grad=False)
        y_pred = forward(x1, b)

        loss = (y_pred - y_t).pow(2).mean()
        print(loss.data, k / (len(x)*epochs))

        loss.backward()

        b.data  -= learning_rate * b.grad.data
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data
        w3.data -= learning_rate * w3.grad.data

        b.grad.data.zero_()
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        w3.grad.data.zero_()
        
        k += 1


valid = 0
for t in range(Nclasses*Mset):
    x_t = test_data[t]
    y_pred = activ(x_t, b)
    
    ans_p = torch.argmax(y_pred)
    ans = torch.argmax(test_class[t])
    
    if ans == ans_p:
        valid += 1

print('valid = {valid}%'.format(valid=100*valid/(Nclasses * Mset)))

print('Dispersion(b) = {d}'.format(d = torch.mean(b*b) - (torch.mean(b))**2))
print('Dispersion(w{n}) = {d}'.format(n=1, d = torch.mean(w1*w1) - (torch.mean(w1))**2))
print('Dispersion(w{n}) = {d}'.format(n=2, d = torch.mean(w2*w2) - (torch.mean(w2))**2))
print('Dispersion(w{n}) = {d}'.format(n=3, d = torch.mean(w3*w3) - (torch.mean(w3))**2))
#    
#    y_pred = x.add(b).matmul(w1).sigmoid().mm(w2).sigmoid()
#    ans = torch.argmax(y, dim=1)
#    ans_p = torch.argmax(y_pred, dim=1)
#    
#    valid = ((ans_p - ans) == 0).sum()
#    print(valid)
#    
#    y_test = test_data.mm(w1).tanh().mm(w2).tanh()
#    (M, _) = y_test.shape
#    for j in range(M):
#        
#    y_pred = x.mm(w1).clamp(min=0).mm(w2).tanh()
#    
#    
#    show_x(test_data.permute((2,0,3,1))[1,2,0].numpy())
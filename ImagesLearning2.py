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
Nset = 20
Nclasses = 10

bin_data = []
seg = []

test_data = np.zeros((Nseg, 256, Nset//Nseg, Nclasses), dtype=np.double)
train_data = np.zeros((Nseg, 256, Nset - Nset//Nseg, Nclasses), dtype=np.double)
class_data = np.zeros((Nseg, Nclasses, Nset - Nset//Nseg, Nclasses), dtype=np.double)
bin_data = raw_data.copy()


seg = []
for j in range(0,Nseg):
    ## делаем 5 сегментов
    seg.append(bin_data[:, j:Nset:Nseg, :])
e = []
for j in range(0, Nseg):
# цикл по сегментам
    e.append(seg.copy())
    # вытаскиваем один сегмент и помещаем в test_data
    test_data[j] = np.array(e[j].pop(j))
    # объединяем четыре оставшихся сегмента в train_data
    train_data[j] = np.concatenate(e[j], axis = 1)

e = []

for i in range(Nseg):
    for j in range(Nset - Nset//Nseg):
        for k in range(Nclasses):
            y_hat = np.zeros(10, np.float64)
            y_hat[k] = 1
            class_data[i,:,j,k] = y_hat

print(train_data.shape)
print(class_data.shape)
    
def MyNetworkForward(weights, bias, x):
    h1 = weights @ x + bias
    a1 = torch.tanh(h1)
    return a1



test_data = torch.from_numpy(test_data).permute((0,3,2,1)).flatten(start_dim=0,end_dim=2)

train_data = torch.from_numpy(train_data).permute((0,3,2,1)).flatten(start_dim=0,end_dim=2)

class_data = torch.from_numpy(class_data).permute((0,3,2,1)).flatten(start_dim=0,end_dim=2)

print(train_data.shape)
print(class_data.shape)
#raise Exception

dtype = torch.float64
N, D_in, H, D_out = 800, 256, 3, 10

x = Variable(train_data.type(dtype), requires_grad=False)
y = Variable(class_data.type(dtype), requires_grad=False)
b = Variable(torch.randn(H).type(dtype), requires_grad=True)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 0.005
for t in range(1000):
    y_pred = x.mm(w1).sigmoid().mm(w2).sigmoid()

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data)

    loss.backward()

   # b.data -= learning_rate * b.grad.data
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    
    #print(w1.grad.norm())
    #b.grad.data.zero_()
    w1.grad.data.zero_()
    w2.grad.data.zero_()
#
#y_test = test_data.mm(w1).tanh().mm(w2).tanh()
#(M, _) = y_test.shape
#for j in range(M):
#y_pred = x.mm(w1).clamp(min=0).mm(w2).tanh()


#show_x(test_data.permute((2,0,3,1))[1,2,0].numpy())
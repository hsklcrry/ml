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
    img = Image.fromarray(x.reshape(img_shape))
    img.show()
    
def classify(i, p):
    return p


img_shape = (16, 16)

mat = scipy.io.loadmat('usps_all.mat')
raw_data = mat['data']
##

## todo 
Nseg = 5
Nset = 20
Nclasses = 10

bin_data = []
seg = []

test_data = np.zeros((Nseg, 256, Nset//Nseg, Nclasses), dtype=np.double)
train_data = np.zeros((Nseg, 256, Nset - Nset//Nseg, Nclasses), dtype=np.double)
class_data = np.zeros((Nseg, Nset - Nset//Nseg, Nclasses, Nclasses), dtype=np.double)
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



for i in range(Nseg):
    for j in range(Nset - Nset//Nseg):
        for k in range(Nclasses):
            y_hat = np.zeros(10, np.float64)
            y_hat[k] = 1
            class_data[i,j,k] = y_hat

print(train_data.shape)
print(class_data.shape)
    
def MyNetworkForward(weights, bias, x):
    h1 = weights @ x + bias
    a1 = torch.tanh(h1)
    return a1


ls = []

test_data = torch.from_numpy(test_data).permute((0,2,3,1)).flatten(0,2)

train_data = torch.from_numpy(train_data).permute((0,2,3,1)).flatten(0,2)

class_data = torch.from_numpy(class_data).permute((0,1,3,2)).flatten(0,2)

print(train_data.shape)
print(class_data.shape)
#raise Exception

b = (2 * np.random.rand( 10) - 1) / 10
b = torch.from_numpy(b)
offset = Variable(b)


dtype = torch.float64
N, D_in, H, D_out = 800, 256, 20, 10

x = Variable(train_data, requires_grad=False)
y = Variable(class_data, requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(20):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()

#show_x(test_data.permute((2,0,3,1))[1,2,0].numpy())
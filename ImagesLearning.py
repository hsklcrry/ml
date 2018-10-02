from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import numpy as np


def discretize(h):
    return np.vectorize(
    lambda x: 255 if x/255.0 > h else 0 )

def data(dataset, i, j):
    """ возвращает i-тый вектор j-того символа. """
    return dataset[:,i,j].reshape((16,16)).transpose()

def show_img(dataset, i, j):
    """ рисует i-тое изображение j-того символа. """
    img = Image.fromarray(data(dataset, i, j), 'L')
    img.show()
    
def classify(i, p):
    return p

img_shape = (16, 16)


mat = scipy.io.loadmat('usps_all.mat')
raw_data = mat['data']
##

## todo 
Nseg = 5

bin_data = []
seg = []
res = np.zeros((5, 10, 120), dtype=np.int8)
# res.shape = (кол-во h, кол-во сегментов, кол-во классов, мощность выборки)

test_data = 0
train_data = 0
bin_data = raw_data.copy()
seg = []
for j in range(0,Nseg):
    ## делаем 5 сегментов
    seg.append(bin_data[:, j:600:Nseg, :])
e = []
for j in range(0, Nseg):
# цикл по сегментам
    e.append(seg.copy())
    # вытаскиваем один сегмент и помещаем в test_data
    test_data = np.array(e[j].pop(j))
    # объединяем четыре оставшихся сегмента в train_data
    train_data = np.concatenate(e[j], axis = 1)
    # обучение
    #theta = NBtrain(train_data, classify)
    #for p in range(0,10):
       # for k in range(0,120):
            #res[j, p, k] = NBclassify(test_data[:, k, p], theta)
           # res[i, j, p, k] = NBclassify(test_data[:, k, p], theta)

#for i in range(0,len(test_images)):
#    test_images[i] = np.array(test_images[i])/255
#
#for i in range(0,len(tr_images)):
#    tr_images[i] = np.array(tr_images[i])/255

def activ(x):
    return np.maximum(x,0)
    #return (np.exp(2*x)-1)/(1+np.exp(2*x))
    #return 1/(1+math.exp(-x))
    #return math.tanh(x)

def nn_calculate(img):
    resp = list(range(0,10))
    for i in range(0,10):
        r = w[:,i]*img
        r = activ(np.sum(r))+b[i]
        resp[i] = r
    return np.argmax(resp)

w = (2 * np.random.rand(10, 256) - 1) / 10
b = (2 * np.random.rand(10) - 1) / 10

(_, Nsamples, Nclasses) = train_data.shape
resp = np.zeros(10, dtype=np.float32)
epoch_max=20
for epoch in range(epoch_max):
    for c in range(Nclasses):
        for n in range (Nsamples):
            img = train_data[:,n, c]
            for i in range(0, 10):
                r = w[i] * img
                r = activ(np.sum(r) + b[i])
                resp[i] = r
            resp_cls = np.argmax(resp)
            resp = np.zeros(10, dtype=np.float32)
            resp[resp_cls] = 1.0
            true_resp = np.zeros(10, dtype=np.float32)
            true_resp[c] = 1.0
            error = resp - true_resp
            delta = error * ((resp >= 0) * np.ones(10))
            for i in range(0, 10):
                w[i] -= np.dot(img, delta[i])
                b[i] -= delta[i]



(_, Ntest_samples, Nclasses) = test_data.shape

def nn_calculate(img):
    resp = list(range(0,10))
    for i in range(0,10):
        r = w[i] * img
        r = activ(np.sum(r)+b[i])
        resp[i] = r
    return np.argmax(resp)

total = Ntest_samples
valid = 0
invalid = []

for c in range(Nclasses):
    for n in range(Ntest_samples):
        img = test_data[:, n, c]
        predicted = nn_calculate(img)
        if predicted == c:
            valid = valid + 1
        #else:
            #invalid.append("image": img, "predicted": predicted, "true": true)

    print("accuracy {}".format(valid/total)+" epoch",epoch)
    
raise Exception
f = open("H:\Data\end.jpg", "rb")
numb1 = np.array(Image.open(f)) / 255
numb = np.zeros(shape=(784), dtype=np.int16)
i=0
j=0
for k in range (784):
    if j==28:
        j=0
        i+=1
    numb[k]=numb1[i][j]
    j+=1
resp=list(range(0,10))
for i in range(0,10):
    r = w[i] * numb
    r = activ(np.sum(r) + b[i])
    resp[i] = r
    s = str(i)+" "+str(resp[i])
    print(s)
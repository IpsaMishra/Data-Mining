import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import torchvision
import torchvision.transforms as transforms

def data_subset(raw_data_list, class_list):
    subset_data_list = []
    for i in range(len(raw_data_list)):
        for j in range(len(class_list)):
            if raw_data_list[i][1] == class_list[j]:
                each_data = []
                each_data.append(raw_data_list[i][0])
                each_data.append(j)
                subset_data_list.append(each_data)
            else:
                continue
    print("subset_data_list len: ", len(subset_data_list))
    return subset_data_list

def generate_data(mu, sigma, n_samples):
    x = np.random.multivariate_normal(mu, sigma, n_samples)
    return x

def pred_calc(w,X):
    z = np.matmul(X,w)
    y = 1 / (1 + np.exp(-z))
    return y

def loss_calc(y,t):
    l = np.sum(- t * np.log(y) - (1 - t) * np.log(1 - y))
    return l / np.size(t)

def grad_calc(X,y,t):
    d = y - t
    if np.size(t) == 1:
        g = d * X
    else:
        g = np.matmul(np.transpose(X),d)
    return g / np.size(t)

def LogReg(Xtrain,ytrain):
    loss = []
    grad = []
    n_weight = np.shape(Xtrain)[1]
    w = np.random.random(n_weight)
    if train == 'batch':
        y = pred_calc(w,Xtrain)
        l = loss_calc(y,ytrain)
        loss.append(l)
        g = grad_calc(Xtrain,y,ytrain)
        grad.append(np.mean(np.abs(g)))
        for i in range(maxiter):
            w = w - lr * g
            y = pred_calc(w,Xtrain)
            le = loss_calc(y,ytrain)
            loss.append(le)
            g = grad_calc(Xtrain,y,ytrain)
            grad.append(np.mean(np.abs(g)))
            if np.abs(le - l) <= L1diff:
                print('Number of epochs: ',i+1)
                break
            l = le
    
    elif train == 'online':
        indx = np.random.permutation(len(Xtrain))
        y = pred_calc(w,Xtrain[indx[0]])
        l = loss_calc(y,ytrain[indx[0]])
        loss.append(l)
        g = grad_calc(Xtrain[indx[0]],np.array([y]),np.array([ytrain[indx[0]]]))
        grad.append(np.mean(np.abs(g)))
        for i in range(maxiter):
            k = (i + 1) % len(Xtrain)
            w = w - lr * g
            y = pred_calc(w,Xtrain[indx[k]])
            le = loss_calc(y,ytrain[indx[k]])
            loss.append(le)
            g = grad_calc(Xtrain[indx[k]],np.array([y]),np.array([ytrain[indx[k]]]))
            grad.append(np.mean(np.abs(g)))
            if np.abs(le - l) <= L1diff:
                print('Number of epochs: ',i+1)
                break
            l = le
    
    loss = np.array(loss)
    grad = np.array(grad)
    plt.figure()
    plt.plot(loss)
    plt.figure()
    plt.plot(grad)
    return w, loss, grad

lr = 0.001
maxiter = 100000
L1diff = 0.0001
train = 'online'         # You can have 'batch' or 'online' training option
seedval = 0
dataset = 'generated'   # Data can be 'generated' or 'fashion_mnist'
mu1 = np.array([1,0])
mu2 = np.array([0,1.5])
sigma1 = np.array([[1,0.75],[0.75,1]])
sigma2 = np.array([[1,0.75],[0.75,1]])
n_train = 500
n_test = 250


if dataset == 'generated':
    np.random.seed(seedval)
    Xtrain1 = generate_data(mu1,sigma1,n_train)
    Xtrain2 = generate_data(mu2,sigma2,n_train)
    Xtrain = np.concatenate((Xtrain1,Xtrain2))
    del Xtrain1, Xtrain2
    ytrain = np.concatenate((np.ones(n_train),np.zeros(n_train)))
    Xtrain = np.append(Xtrain,np.ones((np.size(ytrain),1)),axis=1)
    
    Xtest1 = generate_data(mu1,sigma1,n_test)
    Xtest2 = generate_data(mu2,sigma2,n_test)
    Xtest = np.concatenate((Xtest1,Xtest2))
    del Xtest1, Xtest2
    ytest = np.concatenate((np.ones(n_test),np.zeros(n_test)))
    Xtest = np.append(Xtest,np.ones((np.size(ytest),1)),axis=1)
    
    w, loss, grad = LogReg(Xtrain,ytrain)
    y = 0.0 + (pred_calc(w,Xtest) > 0.5)
    
    plt.figure()
    plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
    x1 = -3 + 6 * np.arange(10000) / 10000
    x2 = -(w[0] * x1 + w[2]) / w[1]
    plt.plot(x1,x2)
    
elif dataset == 'fashion_mnist':
    print('fashion_mnist')
    data_path = './'
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
    train_data_raw = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True,
                                                       transform=transform)
    test_data_raw = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True,
                                                      transform=transform)
    train_data = data_subset(train_data_raw, fashion_mnist_class_list)
    test_data = data_subset(test_data_raw, fashion_mnist_class_list)
else:
    print('Select dataset: "generated" or "fashion_mnist" and run again')


y = 0.0 + (pred_calc(w,Xtest) > 0.5)
print('Accuracy: ', np.sum(1-np.abs(y-ytest))/len(Xtest))
## Do NOT modify the code in this file
from __future__ import print_function
import torch
import numpy as np
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from parameter_setting import parameter_setting
from data_loader import data_loader
from mlp_model import Net
from train_test import train, test

## parameters
args = parameter_setting()

## data loaders for training and testing;
train_loader, test_loader, train_data, test_data = data_loader(args)

## load the multi-layered perceptron (MLP) model
mlp_model = Net(args)

## optimizer used to implement weight update using gradient descent
optimizer = torch.optim.SGD(mlp_model.parameters(), lr=args.lr)

## train the model
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch_idx in range(args.epoch_number):
    train_loss_ret, train_acc_ret = train(args, mlp_model, train_loader, optimizer, epoch_idx)
    train_loss_list.append(train_loss_ret)
    train_acc_list.append(train_acc_ret)

    test_loss_ret, test_acc_ret = test(args, mlp_model, test_loader, epoch_idx)
    test_loss_list.append(test_loss_ret)
    test_acc_list.append(test_acc_ret)

## plot loss
x_axis = np.arange(args.epoch_number)
loss_fig = plt.figure()
plt.plot(x_axis, train_loss_list, 'r')
plt.plot(x_axis, test_loss_list, 'b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(("train", "test"))
os.makedirs(args.output_folder, exist_ok=True)
loss_fig.savefig(os.path.join(args.output_folder, args.dataset_name + "_loss.pdf"))
plt.close(loss_fig)

## plot accuracy
acc_fig = plt.figure()
plt.plot(x_axis, train_acc_list, 'r')
plt.plot(x_axis, test_acc_list, 'b')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(("train", "test"))
os.makedirs(args.output_folder, exist_ok=True)
acc_fig.savefig(os.path.join(args.output_folder, args.dataset_name + "_accuracy.pdf"))
plt.close(acc_fig)

## save loss and accuracy to the file
loss_acc_np = np.arange(args.epoch_number).reshape((-1, 1))
loss_acc_np = np.append(loss_acc_np, np.array(train_loss_list).reshape((-1, 1)), axis=1)
loss_acc_np = np.append(loss_acc_np, np.array(test_loss_list).reshape((-1, 1)), axis=1)
loss_acc_np = np.append(loss_acc_np, np.array(train_acc_list).reshape((-1, 1)), axis=1)
loss_acc_np = np.append(loss_acc_np, np.array(test_acc_list).reshape((-1, 1)), axis=1)
os.makedirs(args.output_folder, exist_ok=True)
np.savetxt(os.path.join(args.output_folder,args.dataset_name + "_loss_acc.txt"), loss_acc_np)

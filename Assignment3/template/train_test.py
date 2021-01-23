## Do NOT modify the code in this file
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np

from utils import save_fig


## train the model
def train(args, model, train_loader, optimizer, epoch_index):
    model.train()

    correct = 0
    epoch_train_loss = 0.0

    ## use a weight matrix to store the weight of the network which will be used for visualization
    weight_matrix = np.zeros((args.image_fashion_mnist_width, args.image_fashion_mnist_height))

    for batch_idx, (data, target) in enumerate(train_loader):

        ## preparing the data fed into the neural network
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        ## obtain the output of the network
        if args.dataset_name == "cifar10":
            output = model(data)
        elif args.dataset_name == "fashion_mnist":

            # if viusalize the weights of neural network
            if args.visual_flag:
                output, weight_ret = model(data)
                weight_matrix = weight_ret.detach().numpy()
            # not viusalize the weights of neural network
            else:
                output = model(data)

        # calculate the predication
        train_pred = torch.argmax(F.softmax(output, dim=1), dim=1).view(-1, )

        # count the correct prediction
        correct += train_pred.eq(target.data).sum()

        # The cross entropy loss is calculated, softmax function is embedded in F.cross_entropy() function
        loss = F.cross_entropy(output, target)
        epoch_train_loss += loss.item()

        # loss is used for back-propagation (BP) in MLP
        loss.backward()
        optimizer.step()

    epoch_train_accuracy = 100. * correct / len(train_loader.dataset)
    epoch_loss_mean = epoch_train_loss / len(train_loader)

    if epoch_index % args.log_interval == 0:
        print('Train Epoch: {}\tLoss: {:.6f} Accuracy: {:.2f}%'.format(epoch_index, epoch_loss_mean,
                                                                       epoch_train_accuracy))

    # visualize the weights of neural network for Fashion MNIST dataset
    if args.dataset_name == "fashion_mnist" and args.visual_flag and epoch_index % args.save_weight_interval == 0:
        os.makedirs(args.output_folder, exist_ok=True)
        save_fig(args, weight_matrix, os.path.join(args.output_folder, "network_weights_" + str(epoch_index) + ".pdf"))

    return epoch_loss_mean, epoch_train_accuracy


## test the model
def test(args, model, test_loader, epoch_index):
    model.eval()
    epoch_test_loss, correct = 0, 0

    # testing does not need the gradient
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)

            if args.dataset_name == "cifar10":
                output = model(data)
            elif args.dataset_name == "fashion_mnist":
                if args.visual_flag:
                    output, _ = model(data)
                else:
                    output = model(data)

            # the process is the same with training part
            epoch_test_loss += F.cross_entropy(output, target).item()
            test_pred = torch.argmax(F.softmax(output, dim=1), dim=1).view(-1, )
            correct += test_pred.eq(target.data).cpu().sum()

        epoch_test_loss_mean = epoch_test_loss / len(test_loader)

    epoch_test_accuracy = 100. * correct / len(test_loader.dataset)

    if epoch_index % args.log_interval == 0:
        print('\n      Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_test_loss_mean, correct, len(test_loader.dataset), epoch_test_accuracy))

    return epoch_test_loss_mean, epoch_test_accuracy

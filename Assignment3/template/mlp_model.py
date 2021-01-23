import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.dropout = args.dropout


        ###### Modify/Add your code here ######
        self.iop_cifar10 = torch.nn.Linear(args.image_cifar10_width * args.image_cifar10_height,args.class_number_cifar10)
        self.inp_cifar10 = torch.nn.Linear(args.image_cifar10_width * args.image_cifar10_height,args.n_nodes)
        self.hid_cifar10 = torch.nn.Linear(args.n_nodes,args.n_nodes)
        self.out_cifar10 = torch.nn.Linear(args.n_nodes,args.class_number_cifar10)

        self.iop_fashion_mnist = torch.nn.Linear(args.image_fashion_mnist_width * args.image_fashion_mnist_height,args.class_number_fashion_mnist)
        self.inp_fashion_mnist = torch.nn.Linear(args.image_fashion_mnist_width * args.image_fashion_mnist_height,args.n_nodes)
        self.hid_fashion_mnist = torch.nn.Linear(args.n_nodes,args.n_nodes)
        self.out_fashion_mnist = torch.nn.Linear(args.n_nodes,args.class_number_fashion_mnist)

        self.dropout_mod1 = torch.nn.Dropout(p = self.dropout)
        self.dropout_mod2 = torch.nn.Dropout(p = self.dropout)

    def forward(self, x):

        if self.args.dataset_name == "cifar10":
            x = x.view(-1, self.args.image_cifar10_width * self.args.image_cifar10_height)

            ###### Mofify/Add your code here ######
            if self.args.n_hidden == 0:
                x = self.iop_cifar10(x)
            elif self.args.n_hidden == 1:
                x = F.relu(self.inp_cifar10(x))
                x = self.dropout_mod1(x)
                x = self.out_cifar10(x)
            else:
                x = F.relu(self.inp_cifar10(x))
                x = self.dropout_mod1(x)
                x = F.relu(self.hid_cifar10(x))
                x = self.dropout_mod2(x)
                x = self.out_cifar10(x)

            return x

        elif self.args.dataset_name == "fashion_mnist":
            x = x.view(-1, self.args.image_fashion_mnist_width * self.args.image_fashion_mnist_height)


            if self.args.visual_flag:

                ###### Mofify/Add your code here ######
                if self.args.n_hidden == 0:
                    x = self.iop_fashion_mnist(x)
                    network_weight = self.iop_fashion_mnist.weight
                elif self.args.n_hidden == 1:
                    x = F.relu(self.inp_fashion_mnist(x))
                    x = self.dropout_mod1(x)
                    x = self.out_fashion_mnist(x)
                    network_weight = self.inp_fashion_mnist.weight
                else:
                    x = F.relu(self.inp_fashion_mnist(x))
                    x = self.dropout_mod1(x)
                    x = F.relu(self.hid_fashion_mnist(x))
                    x = self.dropout_mod2(x)
                    x = self.out_fashion_mnist(x)
                    network_weight = self.inp_fashion_mnist.weight

                return x, network_weight

            else:

                ###### Mofify/Add your code here ######
                if self.args.n_hidden == 0:
                    x = self.iop_fashion_mnist(x)
                elif self.args.n_hidden == 1:
                    x = F.relu(self.inp_fashion_mnist(x))
                    x = self.dropout_mod1(x)
                    x = self.out_fashion_mnist(x)
                else:
                    x = F.relu(self.inp_fashion_mnist(x))
                    x = self.dropout_mod1(x)
                    x = F.relu(self.hid_fashion_mnist(x))
                    x = self.dropout_mod2(x)
                    x = self.out_fashion_mnist(x)

                return x


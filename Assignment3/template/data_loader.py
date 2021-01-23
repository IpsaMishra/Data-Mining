import torch
import torchvision
import torchvision.transforms as transforms


def data_subset(raw_data_list, class_list):
    '''
    This function is used to select part of the dataset. The selected classes are listed in the input argument --- "class_list".
    '''
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


def data_loader(args):
    '''
    CIFAR10:       [0 airplane,    1 automobile, 2 bird,     3 cat,   4 deer, 5 dog,    6 frog,  7 horse,   8 ship, 9 truck]
    Fashion MNIST: [0 T-shirt/top, 1 Trouser,    2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag,  9 Ankle boot]

    train_data, test_data: type -- list; You need to transform them into numpy if you want to use them for logistic regression.
    '''

    ###### Modify/Add your code here to implement "transform" used below for data transform/normalization ######
    ## transform = ...
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])




    ## The index of the classes selected, for example: [0, 1]. If none indices are selected, the whole dataset will be used.
    cifar10_class_list = []        ## You may need to change the index number if you want to select different classes.
    fashion_mnist_class_list = []  ## You may need to change the index number if you want to select different classes.


    ###### Do NOT modify the code below
    ## generate data loader for CIFAR10 and Fashion MNIST
    if args.dataset_name == "cifar10":
        train_data_raw = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True,
                                                      transform=transform)
        test_data_raw = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True,
                                                     transform=transform)

        if len(cifar10_class_list) > 0:
            train_data = data_subset(train_data_raw, cifar10_class_list)
            test_data = data_subset(test_data_raw, cifar10_class_list)
        else:
            train_data = train_data_raw
            test_data = test_data_raw
    elif args.dataset_name == "fashion_mnist":
        train_data_raw = torchvision.datasets.FashionMNIST(root=args.data_path, train=True, download=True,
                                                       transform=transform)
        test_data_raw = torchvision.datasets.FashionMNIST(root=args.data_path, train=False, download=True,
                                                      transform=transform)
        if len(fashion_mnist_class_list) > 0:
            train_data = data_subset(train_data_raw, fashion_mnist_class_list)
            test_data = data_subset(test_data_raw, fashion_mnist_class_list)
        else:
            train_data = train_data_raw
            test_data = test_data_raw

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    print("Working on {} dataset ... ".format(args.dataset_name))
    print("len of train_loader dataset: ", len(train_loader.dataset))
    print("len of test_loader dataset: ", len(test_loader.dataset))

    return train_loader, test_loader, train_data, test_data

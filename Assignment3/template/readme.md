## Image Classifier using Multi-layered Perceptron (MLP)

- data_loader.py: This file is used to prepare training or testing data for our neural network.
- main.py: This is the main program of the MLP classifier. By running this file, you can train your network and test its performance.
- mlp_model.py: This file includes the pipeline of our MLP image classifier. You need to complete it to make it runnable.
- parameter_setting.py: This file includes all the parameters used in this project. By adding/changing the parameters in this file, we can change the setting of our experiment.
- train_test.py: This file implements both training and testing processes of our image classifier. During the training process, our model will update the parameters/weights of the MLP classifier using back propagation algorithm. While in the testing process, our model will utilize the learned parameters/weights in the training process to do inference to evaluate the performance our model. 
- utils.py: This file includes some useful functions used in this project. Here, a function to plot and save the visualization results of the neural network weights is included.

Please complete this image classifier according to the comments in each file. If there is a comment "## Do NOT modify the code in this file" at the head of that file,
you do not need to change any code of that file.

In the project, the output files including neural network weights visualization figure and loss/accuracy figure/txt file are all saved in the folder named "output" which will be generated automatically. In the "**_loss_acc.txt" file, there are five columns indicating epoch index, train loss, test loss, train accuracy and test accuracy, respectively. You can optionally choose to use those saved results to do analysis.
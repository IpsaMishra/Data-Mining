## Do NOT modify the code in this file
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# save the visualization of weights into a file
def save_fig(args, inner_matrix, inner_file_name):
    inner_fig = plt.figure()
    for i in range(inner_matrix.shape[0]):
        ax = inner_fig.add_subplot(1, inner_matrix.shape[0], i + 1)
        ax.imshow(inner_matrix[i].reshape(args.image_fashion_mnist_width, args.image_fashion_mnist_height))
        plt.xticks([])
        plt.yticks([])
    inner_fig.savefig(inner_file_name)
    plt.close(inner_fig)

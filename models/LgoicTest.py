from tensorflow.examples.tutorials.mnist import input_data
mnist_data_folder = "MNIST_data/"
data = input_data.read_data_sets(mnist_data_folder, one_hot=False)
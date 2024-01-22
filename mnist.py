## MNIST dataset
import logging
import struct
import numpy
import matplotlib.pyplot as plt

def read_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = []
        for i in range(num):
            label = struct.unpack('B', f.read(1))[0]
            labels.append(label)
    print(magic,"  ",num)
    labels = numpy.array(labels)
    return labels

def read_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = []
        for i in range(num):
            image = []
            for j in range(rows * cols):
                pixel = struct.unpack('B', f.read(1))[0]
                image.append(pixel)
            images.append(image)
    print(magic,"  ",num, "  ",rows,"  ",cols)
    images = numpy.array(images).reshape(-1,1,28,28)
    return images

def pre_load():

    train_labels = read_labels('./MNIST/train-labels-idx1-ubyte')
    train_images = read_images('./MNIST/train-images-idx3-ubyte')
    test_labels = read_labels('./MNIST/t10k-labels-idx1-ubyte')
    test_images = read_images('./MNIST/t10k-images-idx3-ubyte')
    print(train_images.shape,train_labels.shape)
    print(test_images.shape,test_labels.shape)
    numpy.save("./mnist_py/train_images.npy",train_images)
    numpy.save("./mnist_py/train_labels.npy",train_labels)
    numpy.save("./mnist_py/test_images.npy",test_images)
    numpy.save("./mnist_py/test_labels.npy",test_labels)

def load_mnist():
    train_images = numpy.load("./mnist_py/train_images.npy").astype('float32')
    train_labels = numpy.load("./mnist_py/train_labels.npy").astype('int32')
    test_images = numpy.load("./mnist_py/test_images.npy").astype('float32')
    test_labels = numpy.load("./mnist_py/test_labels.npy").astype('int32')
    return train_images,train_labels,test_images,test_labels

def show():
    train_images,train_labels,test_images,test_labels = load_mnist()
    print(train_images.shape,train_labels.shape)
    print(test_images.shape,test_labels.shape)
    num_show = 10
    print(train_labels[:num_show],train_labels[:num_show].shape,train_labels[:num_show].dtype)
    # print(train_images[:num_show])
    # plt.imshow(train_images[0],cmap='gray')
    for i in range(num_show):
        plt.subplot(2,5,i+1)
        plt.imshow(train_images[i,0],cmap='gray')
    plt.savefig("./mnist_py/0.png")


def resample(images,labels,times):
    ## N= labels.shape[0], times : resample times
    ## 1. get index of each class
    ## 2. every class, resample then to get new index
    ## 3. get new images and labels
    new_indexs = []
    for i in range(10):
        index = numpy.where(labels == i)[0]
        # print(index.shape)
        new_index = numpy.random.choice(index,int(times[i]*len(index)))
        new_indexs.append(new_index)
    new_indexs = numpy.concatenate(new_indexs,axis=0)
    new_images = images[new_indexs]
    new_labels = labels[new_indexs]
    return new_images,new_labels
    pass

if __name__ == '__main__':
    # pre_load()
    show()


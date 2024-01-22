
""" CIFAR10 dataset
data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
"""

import pickle
import numpy as np
import itertools


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data_array(file):
    # return: data: [batch,channel, row, col ]
    dict = unpickle(file)
    data = dict[b'data']
    labels = np.array(dict[b'labels']) 
    data = data.reshape(10000, 3, 32, 32)
    return data,labels

def get_small():
    # Get a small dataset for running
    # return: train data : every class of label has 500 samples
    #         test data : every class of label has 100 samples
    train_data, train_labels, test_data, test_labels = _load_cifar10()
    train_data = train_data.reshape(50000, 3, 32, 32)
    train_labels = np.array(train_labels).reshape(50000)
    test_data = test_data.reshape(10000, 3, 32, 32)
    test_labels = np.array(test_labels).reshape(10000)

    small_train_data = []
    small_train_labels = []
    small_test_data = []
    small_test_labels = []
    for i in range(10):
        small_train_data.append(train_data[train_labels==i,:,:,:][:500])
        small_train_labels.append(train_labels[train_labels==i][:500])
        small_test_data.append(test_data[test_labels==i,:,:,:][:100])
        small_test_labels.append(test_labels[test_labels==i][:100])
    small_train_data = np.concatenate(small_train_data)
    small_train_labels = np.concatenate(small_train_labels)
    small_test_data = np.concatenate(small_test_data)
    small_test_labels = np.concatenate(small_test_labels)

    ## Shuffle
    idx = np.arange(len(small_train_data))
    np.random.shuffle(idx)  
    small_train_data = small_train_data[idx]
    small_train_labels = small_train_labels[idx]
    idx = np.arange(len(small_test_data))
    np.random.shuffle(idx)
    small_test_data = small_test_data[idx]
    small_test_labels = small_test_labels[idx]
    return small_train_data, small_train_labels, small_test_data, small_test_labels

def save_small():
    small_train_data, small_train_labels, small_test_data, small_test_labels = get_small()
    print(small_train_data.shape)
    print(small_train_labels.shape)
    print(small_test_data.shape)
    print(small_test_labels.shape)
    np.save('./cifar10_small/small_train_data.npy',small_train_data)
    np.save('./cifar10_small/small_train_labels.npy',small_train_labels)
    np.save('./cifar10_small/small_test_data.npy',small_test_data)
    np.save('./cifar10_small/small_test_labels.npy',small_test_labels)

def read_small():
    small_train_data = np.load('./cifar10_small/small_train_data.npy')
    small_train_labels = np.load('./cifar10_small/small_train_labels.npy')
    small_test_data = np.load('./cifar10_small/small_test_data.npy')
    small_test_labels = np.load('./cifar10_small/small_test_labels.npy')
    return small_train_data, small_train_labels, small_test_data, small_test_labels

def _load_cifar10():
    # return: train_data, train_labels , test_data, test_labels
    data_dir = './cifar-10-batches-py'
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for i in range(1,6):
        file = data_dir + '/data_batch_' + str(i)
        data,labels = get_data_array(file)
        train_data.append(data)
        train_labels.append(labels)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    file = data_dir + '/test_batch'
    test_data,test_labels = get_data_array(file)

    return train_data, train_labels, test_data, test_labels

def load_cifar10(small=True):
    if small:
        return get_small()
    else:
        return _load_cifar10()

if __name__=="__main__":
    save_small()

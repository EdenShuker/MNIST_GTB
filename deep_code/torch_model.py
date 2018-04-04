from time import time
import mnist
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch as tr
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


def create_net(dims_ls, act_f_ls):
    """
    :param dims_ls: array-like of integers, each is dimension.
    :param act_f_ls: array-like of activation functions, each is nn.Module, like nn.ReLU().
    :return: sequential model of linear layers with activation functions after each layer.
    """
    layers = []
    for i, act_f in enumerate(act_f_ls):
        layers.append(nn.Linear(dims_ls[i], dims_ls[i + 1]))
        layers.append(nn.Dropout(p=0.1))
        layers.append(act_f)

    net = nn.Sequential(*layers)
    return net


def train_on(model, data, labels, epochs=50, model_name=None, batch_size=128):
    """
    :param model: nn.Module, like nn.Sequential.
    :param data: array-like of data samples.
    :param labels: array-like of labels, connected to data.
    :param epochs: number of epochs.
    :param model_name: None for not saving the model after training,
                       otherwise, string that represents the name of the file to be save the model in.
    :param batch_size: batch size.
    """
    criterion = nn.BCELoss()
    opt = tr.optim.Adam(model.parameters())
    data_set = TensorDataset(tr.Tensor(data), tr.Tensor(labels))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch_data in data_loader:
            opt.zero_grad()

            # x, y = Variable(tr.FloatTensor(x)), Variable(tr.FloatTensor(y))
            x = Variable(batch_data[0])
            y = Variable(batch_data[1])
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

    if model_name:
        tr.save(model, model_name)


def train_mode(train, dims_ls, act_ls_f, act_ls_b, batch_size=128, epochs=50):
    """
    :param train: array-like of images.
    :param dims_ls: array-like of integers of size (d+1), each represents a dimension.
    :param act_ls_f: array-like of activation-functions, each is nn.Module, like nn.ReLU().
    :param act_ls_b: array-like of activation-functions, each is nn.Module, like nn.ReLU().
    :param batch_size: batch size.
    :param epochs: num epochs.
    """
    encoder = create_net(dims_ls, act_ls_f)
    dims_ls.reverse()
    decoder = create_net(dims_ls, act_ls_b)

    auto_encoder = nn.Sequential(encoder, decoder)
    train_on(auto_encoder, train, train, model_name='first', batch_size=batch_size, epochs=epochs)


def check_acc(model, train, y_train, test, y_test):
    """
    :param model: nn.Module, the encoder.
    :param train: array-like of images.
    :param y_train: array-like of labels connected to train.
    :param test: array-like of images.
    :param y_test: array-like of labels connected to test.
    """
    y_preds = []
    for x, y in zip(test, y_test):
        # init for current sample
        x_rep = model(Variable(tr.FloatTensor(x))).data.numpy()
        most_similar_val, most_similar_label = 1000000, -1

        # find most similar vector and take its label
        for other_x, other_y in zip(train, y_train):
            dist = tr.np.linalg.norm(x_rep - other_x)
            if dist < most_similar_val:  # update values
                most_similar_val = dist
                most_similar_label = other_y

        y_preds.append(most_similar_label)

    print confusion_matrix(y_test, y_preds)
    acc = accuracy_score(y_test, y_preds)
    print 'accuracy %0.2f%%' % (acc * 100.0)


def represent_all(model, train, y_train, num_samples=100):
    """
    :param model: nn.Module, the encoder.
    :param train: array-like of images.
    :param y_train: array-like of labels connected to train.
    :param num_samples: number of samples to take from train.
    :return: tuple of - array of numpy-vectors after representing each by the encoder,
                        list of labels connected to the vectors.
    """
    train_vecs_presentors = []
    y_presentors = []
    data_and_labels = zip(train, y_train)
    tr.np.random.shuffle(data_and_labels)
    for i in range(num_samples):
        x, y = data_and_labels[i]
        train_vecs_presentors.append(model(Variable(tr.FloatTensor(x))).data.numpy())
        y_presentors.append(y)

    return tr.np.array(train_vecs_presentors), y_presentors


def test_mode(mndata, train, y_train):
    """
    :param mndata: mnist-object.
    :param train: array-like of images.
    :param y_train: array-like of labels connected to train.
    """
    auto_encoder = tr.load('first')
    encoder = auto_encoder._modules['0']
    encoder.eval()

    train, y_train = represent_all(encoder, train, y_train)

    test, y_test = mndata.load_testing()
    test = tr.np.array(test).astype('float32') / 255
    y_test = tr.np.array(y_test)

    check_acc(encoder, train, y_train, test, y_test)


def main():
    mode = 'train'
    mndata = mnist.MNIST('../data')
    train, y_train = mndata.load_training()
    train = tr.np.array(train).astype('float32') / 255

    t = time()
    if mode == 'train':
        dims_ls = [784, 128, 64, 32]
        act_ls_f = [nn.ReLU()] * 3
        act_ls_b = [nn.ReLU(), nn.ReLU(), nn.Sigmoid()]
        train_mode(train, dims_ls, act_ls_f, act_ls_b, batch_size=256, epochs=50)
        print 'time to train:', time() - t

    if mode == 'test':
        test_mode(mndata, train, y_train)
        print 'time to check accuracy:', time() - t


if __name__ == '__main__':
    t0 = time()
    main()
    print 'time to run:', time() - t0

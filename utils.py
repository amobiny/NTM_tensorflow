import numpy as np
import h5py


def generate_random_strings(batch_size, seq_length, vector_dim):
    return np.random.randint(0, 2, size=[batch_size, seq_length, vector_dim]).astype(np.float32)


def one_hot_encode(x, dim):
    """
    Returns a one-hot-vector array
    :param x: array of indeces of shape [batch_size, #images, #labels for each image (5 for 5-hot-encoding)]
    :param dim: A scalar defining the depth of the one hot dimension.
    :return: one-hot-encoded array of shape [batch_size, #images, #labels for each image, dim]
    """
    res = np.zeros(np.shape(x) + (dim,), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


def five_hot_decode(x):
    x = np.reshape(x, newshape=np.shape(x)[:-1] + (5, 5))

    def f(a):
        return sum([a[i] * 5 ** i for i in range(5)])

    return np.apply_along_axis(f, -1, np.argmax(x, axis=-1))


def baseN(num, b):
    """
    converts a base 10 number to a base b number.
    for example, if b=5 (which is used for 5-hot-encoding):
                num = 34 = (4x5^0)+(1x5^1)+(1x5^2) --> returns 114
                num = 237 = (2x5^0)+(2X5^1)+(4x5^2)+(1x5^3) --> returns 1422
    :param num: number in base 10
    :param b: destination base
    :return: number in base b
    """
    return ((num == 0) and "0") or (baseN(num // b, b).lstrip("0") + "0123456789abcdefghijklmnopqrstuvwxyz"[num % b])


def compute_accuracy(args, y, output):
    """
    Computes the accuracy list for all instances
    :param args: arguments
    :param y: true labels (i.e. ground truth)  [batch_size, seq_length]
    :param output: predicted labels  [batch_size, seq_length]
    :return: list of accuracy values for instances  [seq_length/n_classes]
    """
    correct = [0] * args.seq_length
    total = [0] * args.seq_length
    if args.label_type == 'one_hot':
        y_decode = one_hot_decode(y)
        output_decode = one_hot_decode(output)
    elif args.label_type == 'five_hot':
        y_decode = five_hot_decode(y)
        output_decode = five_hot_decode(output)
    for i in range(np.shape(y)[0]):
        y_i = y_decode[i]
        output_i = output_decode[i]
        # print(y_i)
        # print(output_i)
        class_count = {}
        for j in range(args.seq_length):
            if y_i[j] not in class_count:
                class_count[y_i[j]] = 0
            class_count[y_i[j]] += 1
            total[class_count[y_i[j]]] += 1
            if y_i[j] == output_i[j]:
                correct[class_count[y_i[j]]] += 1
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, (args.seq_length/args.n_classes)+1)]


def display_and_save(args, all_acc, acc, all_loss, loss, episode, mode='train'):
    """
    Display and save results in HDF5 file
    :param args: arguments
    :param all_acc: all accuracy values so far  [(episode/disp_freq)-1, seq_length/n_classes]
    :param acc: accuracies of the current episode  [seq_length/n_classes]
    :param all_loss: all loss values so far  [(episode/disp_freq)-1]
    :param loss: loss value of the current episode  [1,]
    :param episode: value of current episode
    :param mode: train or test
    :return: all accuracy and loss values so far
    """
    print('---------------------------------------------------------------------------------------------')
    if mode == 'test':
        print('**************************************** Test Result ****************************************')
    for accu in acc:
        print '{:.02%}\t'.format(accu),
    print '{0}\t{1:0.03}'.format(episode, loss)
    if args.save_hdf5:
        # save results in HDF5 file
        all_acc = np.concatenate((all_acc, np.array(acc).reshape([-1, args.seq_length/args.n_classes])))
        all_loss = np.append(all_loss, loss)
        h5f = h5py.File(args.save_dir + '/' + args.model + '_' + args.label_type + '/' + mode + '_results.h5', 'w')
        h5f.create_dataset('all_acc', data=all_acc)
        h5f.create_dataset('all_loss', data=all_loss)
        h5f.close()
    return all_acc, all_loss


def load_results(args, last_episode, mode='train'):
    """
    loads the accuracy and loss results so far while loading a model to continue training
    :param args: arguments
    :param last_episode: Last train/test episode
    :param mode: train or test
    :return: all accuracy and loss values so far
    """
    h5f = h5py.File(args.save_dir + '/' + args.model + '_' + args.label_type + '/' + mode + '_results.h5', 'r')
    if mode == 'train':
        all_acc = h5f['all_acc'][:last_episode / args.disp_freq]
        all_loss = h5f['all_loss'][:last_episode / args.disp_freq]
    elif mode == 'test':
        all_acc = h5f['all_acc'][:last_episode / args.test_freq]
        all_loss = h5f['all_loss'][:last_episode / args.test_freq]
    h5f.close()
    return all_acc, all_loss

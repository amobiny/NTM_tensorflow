from DataLoader import OmniglotDataLoader
from utils import compute_accuracy, display_and_save, load_results
from config import args
import tensorflow as tf
import numpy as np
from model import NTMOneShotLearningModel
import os


def main():
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


def train(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.save_dir + '/' + args.model + '_' + args.label_type):
        os.makedirs(args.save_dir + '/' + args.model + '_' + args.label_type)

    with tf.Session() as sess:
        if args.restore_training:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model + '_' + args.label_type)
            saver.restore(sess, ckpt.model_checkpoint_path)
            last_episode = int(str(ckpt.model_checkpoint_path).split('-')[-1])
            all_acc_train, all_loss_train = load_results(args, last_episode, mode='train')
            all_acc_test, all_loss_test = load_results(args, last_episode, mode='test')
        else:
            saver = tf.train.Saver(tf.global_variables())
            tf.global_variables_initializer().run()
            all_acc_train = all_acc_test = np.zeros((0, args.seq_length / args.n_classes))
            all_loss_train = all_loss_test = np.array([])

        train_writer = tf.summary.FileWriter(args.tensorboard_dir + args.model + '_' + args.label_type +
                                             '/train/', sess.graph)
        test_writer = tf.summary.FileWriter(args.tensorboard_dir + args.model + '_' + args.label_type + '/test/')
        print('---------------------------------------------------------------------------------------------')
        print(args)
        print('---------------------------------------------------------------------------------------------')
        print("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tepisode\tloss")
        for episode in range(args.num_episodes):

            # Train
            x_image, x_label, y = data_loader.fetch_batch(args,
                                                          mode='train',
                                                          augment=args.augment,
                                                          sample_strategy=args.sample_strategy)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            sess.run(model.train_op, feed_dict=feed_dict)
            if episode % args.disp_freq == 0 and episode > 0:
                output, train_loss = sess.run([model.o, model.loss], feed_dict=feed_dict)
                summary_train = sess.run(model.loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(summary_train, episode)
                train_acc = compute_accuracy(args, y, output)
                all_acc_train, all_loss_train = display_and_save(args, all_acc_train, train_acc,
                                                                 all_loss_train, train_loss, episode, mode='train')

            # Test
            if episode % args.test_freq == 0 and episode > 0:
                x_image, x_label, y = data_loader.fetch_batch(args,
                                                              mode='test',
                                                              augment=args.augment,
                                                              sample_strategy=args.sample_strategy)
                feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
                output, test_loss = sess.run([model.o, model.loss], feed_dict=feed_dict)
                summary_test = sess.run(model.loss_summary, feed_dict=feed_dict)
                test_writer.add_summary(summary_test, episode)
                test_acc = compute_accuracy(args, y, output)
                all_acc_test, all_loss_test = display_and_save(args, all_acc_test, test_acc,
                                                               all_loss_test, test_loss, episode, mode='test')

            # Save model
            if episode % args.save_freq == 0 and episode > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '_' + args.label_type + '/model.tfmodel',
                           global_step=episode)


def test(args):
    model = NTMOneShotLearningModel(args)
    data_loader = OmniglotDataLoader(args)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model + '_' + args.label_type)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tloss")
        y_list = []
        output_list = []
        loss_list = []
        for episode in range(args.test_batch_num):
            x_image, x_label, y = data_loader.fetch_batch(args,
                                                          mode='test',
                                                          augment=args.augment,
                                                          sample_strategy=args.sample_strategy)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            output, learning_loss = sess.run([model.o, model.loss], feed_dict=feed_dict)
            y_list.append(y)
            output_list.append(output)
            loss_list.append(learning_loss)
        accuracy = compute_accuracy(args, np.concatenate(y_list, axis=0), np.concatenate(output_list, axis=0))
        for accu in accuracy:
            print('%.4f' % accu)
        print(np.mean(loss_list))


if __name__ == '__main__':
    main()

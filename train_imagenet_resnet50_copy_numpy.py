#!/usr/bin/env python

from __future__ import print_function
import argparse
import multiprocessing
import os
import os.path
import random
import sys

import numpy as np

import chainer
import chainer.cuda
from chainer import training
from chainer.training import extensions

import chainermn
import models.alex as alex
import models.googlenet as googlenet
import models.googlenetbn as googlenetbn
import models.nin as nin
import models.resnet50 as resnet50
# import models.vgg16 as vgg16

from chainer import configuration
chainer.disable_experimental_feature_warning = True

#import matplotlib
#matplotlib.use('Agg')

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


# chainermn.create_multi_node_evaluator can be also used with user customized
# evaluator classes that inherit chainer.training.extensions.Evaluator.
class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        with configuration.using_config('train', False):
          ret = super(TestModeEvaluator, self).evaluate()
        with configuration.using_config('train', True):
          return ret


def main():
    # Check if GPU is available
    # (ImageNet example does not support CPU execution)
    # if not chainer.cuda.available:
    #     raise RuntimeError("ImageNet requires GPU support.")

    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
        # 'vgg16': vgg16.VGG16,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='resnet50',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=128,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=32,
                        help='Number of epochs to train')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    # --root is not used
    #parser.add_argument('--root', '-R', default='.',
    #                    help='Root directory path of image files')
    parser.add_argument('--train_root', '-TR', default='.',
                        help='Root directory path of train image files')
    parser.add_argument('--val_root', '-VR', default='.',
                        help='Root directory path of val image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--communicator', default='naive')
    parser.add_argument('--polyshift', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=True)
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # Prepare ChainerMN communicator.
    comm = chainermn.create_communicator(args.communicator)
    use_gpu = not args.cpu



    if use_gpu:
        device = comm.intra_rank
    else:
        device = -1

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Num epoch: {}'.format(args.epoch))
        print('Minibatch-size: {}'.format(args.batchsize))
        print('Microbatch-size: {}'.format(args.batchsize // comm.size))
        print('GPU : {}'.format(use_gpu))
        print('==========================================')

    for r in range(comm.size):
        if r == comm.rank:
            print("Rank {} [{}]: comm.intra_rank = {}".format(comm.rank,
                                                              os.uname()[1],
                                                              comm.intra_rank))
            sys.stdout.flush()
        comm.mpi_comm.barrier()

    sys.stdout.flush()
    comm.mpi_comm.barrier()


    model = archs[args.arch](comm)
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    if use_gpu:
        chainer.cuda.get_device(device).use()  # Make the GPU current
        model.to_gpu()
    model.to_intel64()

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    if os.path.exists(args.mean):
        mean = np.load(args.mean)
    else:
        print('Warning: {} does not exist'.format(args.mean))
        mean = None

    if comm.rank == 0:
        train = PreprocessedDataset(args.train, args.train_root, mean, model.insize)
        val = PreprocessedDataset(
            args.val, args.val_root, mean, model.insize, False)
    else:
        train = None
        val = None
    train = chainermn.scatter_dataset(train, comm)
    val = chainermn.scatter_dataset(val, comm)

    # We need to change the start method of multiprocessing module if we are
    # using InfiniBand and MultiprocessIterator. This is because processes
    # often crash when calling fork if they are using Infiniband.
    # (c.f., https://www.open-mpi.org/faq/?category=tuning#fork-warning )
    # multiprocessing.set_start_method('forkserver')

    train_iter = chainer.iterators.MultithreadIterator(
        train, args.batchsize // comm.size, n_threads=args.loaderjob)
    val_iter = chainer.iterators.MultithreadIterator(
        val, args.val_batchsize // comm.size, repeat=False, n_threads=args.loaderjob)

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(lr=0.1, momentum=0.9), comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)
        

    if args.polyshift:
        print("poly")
        trainer.extend(extensions.polynomial_shift.PolynomialShift('lr', (1, 320000)), trigger=(1, 'iteration'))

    #val_interval = (10 if args.test else 10), 'iteration'
    #log_interval = (10 if args.test else 10), 'iteration'
    val_interval = 1, 'epoch'
    log_interval = 1, 'epoch'

    # Create a multi node evaluator from an evaluator.
    evaluator = TestModeEvaluator(val_iter, model, device=device)
    
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'lr'
        ]), trigger=log_interval)
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                             'epoch', file_name='loss.png'))
        trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()

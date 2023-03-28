import os
from Logger import Logger
from Datasets import CIFAR10, CIFAR100, ImageNet

from models.resnet import resnet18
from models.resnet import resnet34 as resnet34_imagenet
from models.resnet import resnet50 as resnet50_imagenet
from models.resnet import resnet101 as resnet101_imagenet
from models.googlenet import googlenet as googlenet_imagenet
from models.inception import inception_v3 as inception_imagenet
from models.densenet import densenet121 as densenet_imagenet

basedir, _ = os.path.split(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'data')

RESULTS_DIR = os.path.join(basedir, 'results')

DEBUG = False
USER_CMD = None
SEED = 123
BATCH_SIZE = 128
VERBOSITY = 0
INCEPTION = False

MODELS = {'resnet18_imagenet': resnet18,
          'resnet34_imagenet': resnet34_imagenet,
          'resnet50_imagenet': resnet50_imagenet,
          'resnet101_imagenet': resnet101_imagenet,
          'googlenet_imagenet': googlenet_imagenet,
          'inception_imagenet': inception_imagenet,
          'densenet_imagenet': densenet_imagenet,
          'resnet18_cifar100': resnet18}

DATASETS = {'cifar10':
                {'ptr': CIFAR10,  'dir': os.path.join(basedir, 'datasets')},
            'cifar100':
                {'ptr': CIFAR100, 'dir': os.path.join(basedir, 'datasets')},
            'imagenet':
                {'ptr': ImageNet, 'dir': '/mnt/ilsvrc2012'}}
LOG = Logger()

X_FP =  {'sign': 0, 'exponent': 0, 'mantissa': 2}
W_FP =  {'sign': 0, 'exponent': 1, 'mantissa': 4}
SUM_FP =  {'sign': 0, 'exponent': 1, 'mantissa': 4}

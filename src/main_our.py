import argparse
import os
import numpy as np
import torch
import sys
import Config as cfg
from NeuralNet import NeuralNet
from Datasets import Datasets
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Gil Shomron, gilsho@campus.technion.ac.il',
                                 formatter_class=argparse.RawTextHelpFormatter)

model_names = list(cfg.MODELS.keys())

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, required=True,
                    help='model architectures and datasets:\n' + ' | '.join(model_names))
parser.add_argument('--action', choices=['QUANTIZE', 'INFERENCE'], required=True,
                    help='QUANTIZE: symmetric min-max uniform quantization\n'
                         'INFERENCE: either regular inference or hardware simulated inference')
parser.add_argument('--desc',
                    help='additional string to the test')
parser.add_argument('--chkp', default=None, metavar='PATH',
                    help='model checkpoint')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--x_bits', default=None, type=int, metavar='N',
                    help='activations quantization bits')
parser.add_argument('--w_bits', default=None, type=int, metavar='N',
                    help='weights quantization bits')
parser.add_argument('--skip_bn_recal', action='store_true',
                    help='skip BatchNorm recalibration (relevant only to the INFERENCE action)')
parser.add_argument('--round_mode', choices=['ROUND', 'RAW'], default='ROUND',
                    help='rounding (i.e., nearbyint, default) or raw')
parser.add_argument('--unfold', action='store_true',
                    help='enable tensor unfolding')
parser.add_argument('--gpu', nargs='+', default=None,
                    help='GPU to run on (default: 0)')
parser.add_argument('-v', '--verbosity', default=0, type=int,
                    help='verbosity level (0,1,2) (default: 0)')
parser.add_argument('--boris', default=0, type=int,
                    help='when boris is 1 we run full expirement (0,1) (default: 0)')

def quantize_network(arch, dataset, train_gen, test_gen, model_chkp=None,
                     x_bits=8, w_bits=8, desc=None, layer_to_quant=None):
    # Initialize log file
    name_str = '{}-{}_quantize_x-{}_w-{}'.format(arch, dataset, x_bits, w_bits)
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    name_str = name_str + '_seed-{}'.format(cfg.SEED)

    cfg.LOG.start_new_log(name=name_str)
    cfg.LOG.write('desc={}, x_bits={}, w_bits={}'.format(desc, x_bits, w_bits))

    # Initialize model
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp) #flag per layer, support list for activations and weights

    # Set configurations
    nn.model.set_quantize(True)
    nn.model.set_quantization_bits(x_bits, w_bits)
    nn.model.set_min_max_update(True)
    #quant by layer
    # if layer_to_quant:
    #     nn.model.set_layer_quant_flag(True)
    #     nn.model.set_layer_to_quant(layer_to_quant)

    nn.best_top1_acc = 0
    nn.next_train_epoch = 0

    # Learning rate is set to zero
    nn.train(train_gen, test_gen, epochs=1, lr=0, iterations=2048 / cfg.BATCH_SIZE)

    cfg.LOG.close_log()
    return nn.best_top1_acc


def inference(arch, dataset, train_gen, test_gen, model_chkp, x_bits=None, w_bits=None,
              is_round=None, skip_bn_recal=False, is_unfold=False, desc=None):

    # Get test string ready
    name_str = '{}-{}_inference'.format(arch, dataset)
    name_str = name_str + '_x-{}_w-{}'.format(x_bits, w_bits) if x_bits is not None and w_bits is not None else name_str
    name_str = name_str + '_round' if sparq_x or sparq_w and is_round else name_str + '_raw'
    name_str = name_str + '_noBN' if skip_bn_recal else name_str
    name_str = name_str + '_unfold' if is_unfold else name_str
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    name_str = name_str + '_seed-{}'.format(cfg.SEED)

    # Start log
    cfg.LOG.start_new_log(name=name_str)

    # Init neural net
    nn = NeuralNet(arch, dataset, model_chkp=model_chkp)

    # Configuration
    nn.model.set_quantize(x_bits is not None and w_bits is not None)
    nn.model.set_quantization_bits(x_bits, w_bits)
    nn.model.set_sparq(sparq_x, sparq_w)
    nn.model.set_round(is_round)
    nn.model.set_unfold(is_unfold)
    nn.model.set_shift_opt(shift_opt_x, shift_opt_w)
    nn.model.set_group_sz(group_sz_x, group_sz_w)
    nn.model.set_min_max_update(False)

    # Print configuration
    cfg.LOG.write_title('Configurations')
    nn.model.print_config()

    # Start test!
    cfg.LOG.write_title('Start Test')

    if skip_bn_recal is None or skip_bn_recal is False:
        cfg.LOG.write('Conducting BN recalibration')
        nn.next_train_epoch = 0
        nn.train(train_gen, test_gen, epochs=1, lr=0, iterations=2048 / cfg.BATCH_SIZE)
    else:
        cfg.LOG.write('Skipping BN recalibration')
        nn.test(test_gen)

    return

def plot_accuracies(w_bits_list,x_bits_list,top1_accuracy_mat):

    fig = plt.figure(figsize=(16,8))
    for w_bits in w_bits_list:
        plt.plot(x_bits_list,top1_accuracy_mat[w_bits,:],label = 'W bits = {}'.format(w_bits))
    plt.grid(True)
    plt.title('top1 acc as function of x quant for differnet w quant')
    plt.xlabel('X quantization bits')
    plt.ylabel('top 1 accuracy')
    plt.legend()
    fig.savefig(os.path.join("/home/borisneliza/results/",'top1_x_quant_w_quant.png'))
    plt.close(fig)

def main():
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    arch = args.arch.split('_')[0]
    dataset = args.arch.split('_')[1]

    cfg.BATCH_SIZE = args.batch_size
    cfg.VERBOSITY = args.verbosity
    cfg.USER_CMD = ' '.join(sys.argv)
    cfg.INCEPTION = (arch == 'inception')

    dataset_ = Datasets.get(dataset)

    # Deterministic random numbers
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    test_gen, _ = dataset_.testset(batch_size=args.batch_size)
    train_gen, _ = dataset_.trainset(batch_size=args.batch_size)

    model_chkp = None if args.chkp is None else cfg.RESULTS_DIR + '/' + args.chkp

    if args.action == 'QUANTIZE':
        if args.boris == 1:
            w_bits_list = [5,6,7,8]
            x_bits_list = [5,6,7,8]
            top1_accuracy_mat = torch.empty((max(w_bits_list)+1,len(x_bits_list)))
            for w_bits in w_bits_list:
                x_index = 0
                for x_bits in x_bits_list:
                    acc = quantize_network(arch, dataset, train_gen, test_gen,
                                                                model_chkp=model_chkp,
                                                                x_bits=x_bits, w_bits=w_bits, desc=args.desc)
                    top1_accuracy_mat[w_bits,x_index] = acc
                    x_index = x_index + 1
                    print("mat:" , top1_accuracy_mat)                        
        else:
            quantize_network(arch, dataset, train_gen, test_gen,
                            model_chkp=model_chkp,
                            x_bits=args.x_bits, w_bits=args.w_bits, desc=args.desc, layer_to_quant=3)
    elif args.action == 'INFERENCE':
        inference(arch, dataset, train_gen, test_gen,
                  model_chkp=model_chkp,
                  x_bits=args.x_bits, w_bits=args.w_bits,
                  is_round=(args.round_mode == 'ROUND'),
                  skip_bn_recal=args.skip_bn_recal, is_unfold=args.unfold, desc=args.desc)
    
    if args.boris ==1 :
        plot_accuracies(w_bits_list,x_bits_list,top1_accuracy_mat)

    return


if __name__ == '__main__':
    main()

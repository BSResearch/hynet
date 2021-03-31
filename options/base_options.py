import argparse
import os
from util import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # data params
        self.parser.add_argument('--dataroot', required=True,
                                 help='path to graphs (should have sub-folders train, test)')
        self.parser.add_argument('--dataset_mode', default='segmentation')
        self.parser.add_argument('--ninput_edges', type=int, default=2250,
                                 help='# of input edges (will include dummy edges)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples per epoch')

        # network params
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--nef_hidden_size', type=int, default=[32, 32, 32, 64, 64, 128],
                                 help='input hidden size for series of NEF embedding layers')
        self.parser.add_argument('--num_head', type=int, default=[2, 2, 2, 2, 2, 2], help='number of attention head for'
                                                                                          'NEF embedding layer')
        self.parser.add_argument('--fcn_hidden_size', type=int, default=[128, 128, 64, 32],
                                 help='input hidden size for'
                                      'fully connected layer')
        self.parser.add_argument('--embedding_hidden_size', type=int, default=[32],
                                 help='input hidden size for'
                                      'initial encoding layer fully connected layer.'
                                      'Note: The list consist of the hidden layers sizes from first to the second last'
                                      'The last layer has the size which is equal to the fist layer of '
                                      'HyNet_hidden_size')
        self.parser.add_argument('--gat_dropout', type=float, default=0.5, help='GAT layer drop out')
        self.parser.add_argument('--fcn_dropout', type=float, default=0.50, help='FCN layer drop out')
        self.parser.add_argument('--classification_element', type=str,
                                 help='select the type classification: node, edge, face')
        # general params
        self.parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='debug',
                                 help='name of the experiment.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        self.parser.add_argument('--save_prediction_for_test_files', default=False, help='True if segmentation result '
                                                                                         'is required')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt

import torch
from .base_dataset import BaseDataset
import os
from util.util import is_graph_file
import numpy as np
import dgl


class SegmentationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = self.make_dataset(self.dir)
        self.classes = np.loadtxt(os.path.join(self.root, 'classes.txt'))
        self.offset = self.classes[0]
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        self.canonical_etypes = self.get_canoncial_etypes()
        opt.nclasses = self.nclasses

    def __getitem__(self, index):
        path = self.paths[index]
        graph_list, label_dict = dgl.load_graphs(path)
        graph = graph_list[0]
        assert torch.sum(torch.isnan(graph.ndata['geometric_feat']['edge'])) == 0, print(path)
        graph.apply_nodes(lambda nodes:
                          {'geometric_feat': (nodes.data['geometric_feat'] - self.mean_node_feat)/self.std_node_feat},
                          ntype='node')
        graph.apply_nodes(lambda nodes:
                          {'geometric_feat': (nodes.data['geometric_feat'] - self.mean_edge_feat)/self.std_edge_feat},
                          ntype='edge')
        graph.apply_nodes(lambda nodes:
                          {'geometric_feat': (nodes.data['geometric_feat'] - self.mean_face_feat)/self.std_face_feat},
                          ntype='face')
        label = graph.ndata['label']
        if self.opt.save_segmentation_for_test_files:
            return path, graph, label
        return graph, label

    def __len__(self):
        return self.size

    @staticmethod
    def make_dataset(path):
        graphs = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_graph_file(fname):
                    path = os.path.join(root, fname)
                    graphs.append(path)
        return graphs

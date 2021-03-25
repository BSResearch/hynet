import torch.utils.data as data
import numpy as np
import pickle
import os
import dgl
import torch


class BaseDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.mean = 0
        self.std = 1
        super(BaseDataset, self).__init__()

    def get_canoncial_etypes(self):
        for _, _, file_names in sorted(os.walk(self.dir)):
            for i in range(1):
                g_sample_name = file_names[i]
        g_sample_path = os.path.join(self.dir, g_sample_name)
        g_sample, _ = dgl.load_graphs(g_sample_path)
        g_sample = g_sample[0]
        return g_sample.canonical_etypes

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        """

        mean_std_cache = os.path.join(self.root, 'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')

            # read the first graph to get the dimensions
            # _, _, g_samples = sorted(os.walk(self.dir))
            for _, _, file_names in sorted(os.walk(self.dir)):
                for i in range(1):
                    g_sample_name = file_names[i]
            g_sample_path = os.path.join(self.dir, g_sample_name)
            g_sample, _ = dgl.load_graphs(g_sample_path)
            g_sample = g_sample[0]
            node_feat_len = g_sample.ndata['geometric_feat']['node'].shape[1]
            edge_feat_len = g_sample.ndata['geometric_feat']['edge'].shape[1]
            face_feat_len = g_sample.ndata['geometric_feat']['face'].shape[1]
            mean_node_feat, std_node_feat = torch.zeros((node_feat_len)), torch.zeros((node_feat_len))
            mean_edge_feat, std_edge_feat = torch.zeros((edge_feat_len)), torch.zeros((edge_feat_len))
            mean_face_feat, std_face_feat = torch.zeros((face_feat_len)), torch.zeros((face_feat_len))
            count = 0
            for root, _, fnames in sorted(os.walk(self.dir)):
                for fname in fnames:
                    # if int(fname.split('_')[-3]) == 0:
                    fpath = os.path.join(self.dir, fname)
                    g_list, label_dict = dgl.load_graphs(fpath)
                    g = g_list[0]

                    # node features
                    node_feat_mean_single_graph = g.ndata['geometric_feat']['node'].mean(axis=0)
                    node_feat_std_single_graph = g.ndata['geometric_feat']['node'].std(axis=0)
                    mean_node_feat = mean_node_feat + node_feat_mean_single_graph
                    std_node_feat = std_node_feat + node_feat_std_single_graph

                    # edge features
                    edge_feat_mean_single_graph = g.ndata['geometric_feat']['edge'].mean(axis=0)
                    edge_feat_std_single_graph = g.ndata['geometric_feat']['edge'].std(axis=0)
                    mean_edge_feat = mean_edge_feat + edge_feat_mean_single_graph
                    std_edge_feat = std_edge_feat + edge_feat_std_single_graph

                    # face_features
                    face_feat_mean_single_graph = g.ndata['geometric_feat']['face'].mean(axis=0)
                    face_feat_std_single_graph = g.ndata['geometric_feat']['face'].std(axis=0)
                    mean_face_feat = mean_face_feat + face_feat_mean_single_graph
                    std_face_feat = std_face_feat + face_feat_std_single_graph
                    count = count + 1

            mean_node_feat = mean_node_feat / count
            std_node_feat = std_node_feat / count
            mean_edge_feat = mean_edge_feat / count
            std_edge_feat = std_edge_feat / count
            mean_face_feat = mean_face_feat / count
            std_face_feat = std_face_feat / count
            transform_dict = {
                'mean_node_feat': mean_node_feat,
                'std_node_feat': std_node_feat,
                'node_feat_len': node_feat_len,
                'mean_edge_feat': mean_edge_feat,
                'std_edge_feat': std_edge_feat,
                'edge_feat_len': edge_feat_len,
                'mean_face_feat': mean_face_feat,
                'std_face_feat': std_face_feat,
                'face_feat_len': face_feat_len
            }
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)

        # open mean/std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean_node_feat = transform_dict['mean_node_feat']
            self.std_node_feat = transform_dict['std_node_feat']
            self.node_feat_len = transform_dict['node_feat_len']
            self.mean_edge_feat = transform_dict['mean_edge_feat']
            self.std_edge_feat = transform_dict['std_edge_feat']
            self.edge_feat_len = transform_dict['edge_feat_len']
            self.mean_face_feat = transform_dict['mean_face_feat']
            self.std_face_feat = transform_dict['std_face_feat']
            self.face_feat_len = transform_dict['face_feat_len']


def collate_fn(samples):
    # paths, graphs, labels = map(list, zip(*samples))
    graphs, labels = map(list, zip(*samples))
    meta_labels = {}
    keys = labels[0].keys()
    batched_graph = dgl.batch(graphs)
    for key in keys:
        meta_labels.update({key: torch.cat([label[key] for label in labels])})
    return batched_graph, meta_labels


def collate_fn_2(samples):
    paths, graphs, labels = map(list, zip(*samples))
    meta_labels = {}
    keys = labels[0].keys()
    batched_graph = dgl.batch(graphs)
    for key in keys:
        meta_labels.update({key: torch.cat([label[key] for label in labels])})
    return paths, batched_graph, meta_labels
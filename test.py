from nn.hynet import HyNet
import torch
from options.test_options import TestOptions
from data import DataLoader
import warnings
from util.accuracy_calculation import soft_accuracy_meshCNN, soft_accuracy_count, soft_accuracy_edge_length
import os
import dgl

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opt = TestOptions().parse()
    opt.serial_batches = True
    dataset = DataLoader(opt)
    device = dataset.dataset.device
    dataset_size = len(dataset)
    classification_element = opt.classification_element

    node_type_feature_len = {'node': dataset.dataset.node_feat_len,
                             'edge': dataset.dataset.edge_feat_len,
                             'face': dataset.dataset.face_feat_len}
    canonical_etypes = dataset.dataset.canonical_etypes
    model = HyNet(canonical_etypes=canonical_etypes,
                  in_size=node_type_feature_len,
                  embedding_hidden_size=opt.embedding_hidden_size,
                  hidden_size=opt.nef_hidden_size,
                  num_heads=opt.num_head,
                  fcn_hidden_size=opt.fcn_hidden_size,
                  out_size=opt.nclasses,
                  gat_dropout=opt.gat_dropout,
                  fcn_dropout=opt.fcn_dropout,
                  classification_element=classification_element).double().to(device)
    print(opt.model_file)

    model.load_state_dict(torch.load(opt.model_file))
    total_correct = {'meshCNN_based': 0, 'count_based': 0, 'edge_length_based': 0, 'hard_edge_label': 0}
    total_sample = {'meshCNN_based': 0, 'count_based': 0, 'edge_length_based': 0}
    model.eval()
    if opt.save_prediction_for_test_files:
        for i, (path, graph, label) in enumerate(dataset.dataloader):
            with torch.no_grad():
                graph = graph.to(device)
                label_test = label[classification_element].long().to(device)
                pred = model(graph)
                pred_class = pred[classification_element].data.max(1)[1]
                graph.nodes['edge'].data['prediction_class'] = pred_class
                graph_name = path[0].split('/')[-1]
                filename_to_save = os.path.join(opt.results_dir, graph_name)
                dgl.data.utils.save_graphs(filename_to_save, [graph])

    else:
        for i, (graph, label) in enumerate(dataset.dataloader):
            with torch.no_grad():
                graph = graph.to(device)
                label_test = label[classification_element].long().to(device)
                pred = model(graph)
                pred_class = pred[classification_element].data.max(1)[1]
                graph.nodes['edge'].data['prediction_class'] = pred_class
                edges_length = graph.ndata['edge_length']['edge'].to(device)
                edges_area = graph.ndata['edge_area']['edge'].to(device)
                soft_labels = graph.ndata['edge_soft_label']['edge'].to(device)
                correct_mesh_CNN, total_mesh_CNN = soft_accuracy_meshCNN(pred_class, soft_labels, edges_area)
                total_correct['meshCNN_based'] += correct_mesh_CNN
                total_sample['meshCNN_based'] += total_mesh_CNN
                correct_edge_length, total_edge_length = soft_accuracy_edge_length(pred_class, soft_labels,
                                                                                   edges_length, device)
                total_correct['edge_length_based'] += correct_edge_length
                total_sample['edge_length_based'] += total_edge_length
                correct_edge_number, total_edge_number = soft_accuracy_count(pred_class, soft_labels, device)
                total_correct['count_based'] += correct_edge_number
                total_sample['count_based'] += total_edge_number
                total_correct['hard_edge_label'] += pred_class.eq(label_test).sum().item()

        print('accuracy based on edge length: ',
              (total_correct['edge_length_based'] / total_sample['edge_length_based']) * 100)
        print('accuracy based on edge area: ',
              (total_correct['meshCNN_based'] / total_sample['meshCNN_based']) * 100)
        print('accuracy based on number of correct soft edge label: ',
              (total_correct['count_based'] / total_sample['count_based']) * 100)
        print('accuracy based on number of correct hard edge: ',
              (total_correct['hard_edge_label'] / total_sample['count_based']) * 100)

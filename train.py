from nn.hynet import HyNet
import torch
from options.train_options import TrainOptions
from data import DataLoader
import warnings
import torch.optim as optim
import torch.nn as nn
import os
from util import util
from util.accuracy_calculation import soft_accuracy_count
from options.test_options import TestOptions
from nn.networks import get_scheduler
from nn.networks import update_learning_rate
import pickle

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    # prepare train dataset
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)
    classification_element = opt.classification_element

    # prepare test dataset
    test_opt = TestOptions().parse()
    test_opt.serial_batches = True
    test_opt.batch_size = 4
    test_dataset = DataLoader(test_opt)

    device = dataset.dataset.device

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

    if opt.pretrained_model_file is not None:
        model.load_state_dict(torch.load(opt.pretrained_model_file))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters: ', pytorch_total_params)

    total_steps = 0
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = get_scheduler(optimizer, opt)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_metric_dir = os.path.join(save_dir, 'metric')
    util.mkdir(save_metric_dir)

    epoch_losses = []
    train_accuracy = []
    test_accuracy = []
    test_accuracy_soft_edge = []

    for epoch in range(100):
        model.train()

        epoch_loss = 0
        train_corrects_total = {classification_element: 0}
        train_samples_total = {classification_element: 0}

        for i, (graph, label) in enumerate(dataset.dataloader):

            batch_loss = 0
            graph = graph.to(device)
            label[classification_element] = label[classification_element].long().to(device)
            optimizer.zero_grad()
            train_prediction = model(graph)
            loss = loss_func(train_prediction[classification_element], label[classification_element])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # find accuracy:
            train_pred_class = {}
            train_pred_class[classification_element] = train_prediction[classification_element].data.max(1)[1]
            train_corrects_total[classification_element] += train_pred_class[classification_element].\
                eq(label[classification_element]).sum().item()
            for key in train_samples_total.keys():
                train_samples_total[key] += len(label[key])

        # find accuracy on the test set
        test_corrects_total = {classification_element: 0}
        test_samples_total = {classification_element: 0}
        test_soft_corrects_total = 0
        test_sample_total_edge = 0

        for t, (test_graph, test_label) in enumerate(test_dataset.dataloader):
            with torch.no_grad():
                model.eval()
                test_graph = test_graph.to(device)
                test_label[classification_element] = test_label[classification_element].long().to(device)
                test_prediction = model(test_graph)
                test_pred_class = {}
                test_pred_class[classification_element] = test_prediction[classification_element].data.max(1)[1]
                test_corrects_total[classification_element] += test_pred_class[classification_element]. \
                    eq(test_label[classification_element]).sum().item()
                test_samples_total[classification_element] += len(test_label[classification_element])
                if classification_element == 'edge':
                    soft_labels = test_graph.ndata['edge_soft_label']['edge'].to(device)
                    # note the accuracy based on the soft segmentation labels is based on the number of correct labels
                    # as oppose to the the soft accuracy reported in the paper which consider the edge length
                    correct_test_edge, total_test_edge = soft_accuracy_count(test_prediction['edge'], soft_labels, device)
                    test_soft_corrects_total += correct_test_edge
                    test_sample_total_edge += total_test_edge

        test_epoch_acc = {}
        test_correct_total_report = 0
        test_samples_total_report = 0
        for key in test_corrects_total.keys():
            test_epoch_acc[key] = 100 * (test_corrects_total[key] / test_samples_total[key])
            test_correct_total_report += test_corrects_total[key]
            test_samples_total_report += test_samples_total[key]
        test_epoch_acc['total'] = 100 * (test_correct_total_report / test_samples_total_report)

        test_accuracy.append(test_epoch_acc)
        if classification_element == 'edge':
            test_epoch_soft_edge_acc = 100 * (test_soft_corrects_total / test_sample_total_edge)
            test_accuracy_soft_edge.append(test_epoch_soft_edge_acc)

        epoch_loss /= i + 1

        epoch_loss_dict = {classification_element: epoch_loss}
        epoch_losses.append(epoch_loss_dict)

        # compute train accuracy
        train_correct_total_report = 0
        train_samples_total_report = 0
        train_epoch_acc = {}
        for key in train_corrects_total.keys():
            train_epoch_acc[key] = 100 * (train_corrects_total[key] / train_samples_total[key])
            train_correct_total_report += train_corrects_total[key]
            train_samples_total_report += train_samples_total[key]

        # train_epoch_acc['total'] = 100 * (train_correct_total_report / train_samples_total_report)

        # epoch_acc = 100 * (correct_prediction / total_prediction)
        train_accuracy.append(train_epoch_acc)
        for key in train_epoch_acc.keys():
            if key != 'edge':
                print('Epoch {}, loss {} {:.4f}, train {} accuracy {:.4f}, test {} accuracy {:.4f}'.
                      format(epoch, key, epoch_loss_dict[key], key, train_epoch_acc[key], key, test_epoch_acc[key]))
            if key == 'edge':
                print('Epoch {}, loss {} {:.4f}, train {} accuracy {:.4f}, test {} accuracy {:.4f}, '
                      'test {} accuracy based on smooth edge segments (number) {:.4f} '.
                      format(epoch, key, epoch_loss_dict[key], key, train_epoch_acc[key], key, test_epoch_acc[key], key,
                             test_epoch_soft_edge_acc))

        if epoch % opt.save_epoch_freq == 0:
            save_filename = '%s_net.pth' % epoch
            save_path = os.path.join(save_dir, save_filename)
            torch.save(model.state_dict(), save_path)

            # save metric every save_epoch_freq epoch
            epoch_loss_file = 'epoch_losses_%s.npy' % epoch
            epoch_loss_file_name = os.path.join(save_metric_dir, epoch_loss_file)
            with open(epoch_loss_file_name, 'wb') as f1:
                pickle.dump(epoch_losses, f1)
            f1.close()

            epoch_acc_file = 'epoch_train_accuracy_%s.npy' % epoch
            epoch_acc_file_name = os.path.join(save_metric_dir, epoch_acc_file)
            with open(epoch_acc_file_name, 'wb') as f2:
                pickle.dump(train_accuracy, f2)
            f2.close()

            epoch_test_acc_file = 'epoch_test_accuracy_%s.npy' % epoch
            epoch_test_acc_file_name = os.path.join(save_metric_dir, epoch_test_acc_file)
            with open(epoch_test_acc_file_name, 'wb') as f3:
                pickle.dump(test_accuracy, f3)
            f3.close()

            if classification_element == 'edge':
                epoch_test_acc_file = 'epoch_soft_test_accuracy_%s.npy' % epoch
                epoch_test_acc_file_name = os.path.join(save_metric_dir, epoch_test_acc_file)
                with open(epoch_test_acc_file_name, 'wb') as f4:
                    pickle.dump(test_accuracy_soft_edge, f4)
                f4.close()

        update_learning_rate(scheduler, optimizer, opt, test_epoch_acc[classification_element])

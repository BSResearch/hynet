import torch


def soft_accuracy_count(pred, soft_labels, device):
    """
    This function works based on the soft segmentation label. It counts the number of edges that is labeled correctly
    considering the ground truth for soft segmentation edge lables
    :param pred: output array of hynet model
    :param soft_labels: soft labels for edge segmentation
    :param device: GPU or CPU
    :return:number of correct predicted edges and total number of edges
    """
    one_hot_tensor = torch.zeros(soft_labels.size()[0], soft_labels.size()[1]).to(device)
    one_hot_tensor[torch.tensor(range(soft_labels.size()[0])), pred] = 1
    correct = (soft_labels * one_hot_tensor).sum()
    total = soft_labels.size()[0]
    return correct, total


def soft_accuracy_meshCNN(pred, soft_labels, edge_area):
    """
    This accuracy calculation is same as soft accuracy calculation of
    MeshCNN: A Network with an Edge: https://github.com/ranahanocka/MeshCNN/ and
    Primal-Dual Mesh Convolutional Neural Networks: https://github.com/MIT-SPARK/PD-MeshNet
    :param pred: output array of hynet model
    :param soft_labels: soft labels for edge segmentation
    :param edge_area: check MeshCNN github for detail calculation of edge area
    :return: correct weighted by edge_area and total edge area
    """
    correct = 0
    correct_mat = soft_labels.gather(1, pred.unsqueeze(dim=1))
    correct = (correct_mat.float()[:, 0] * edge_area).sum()
    total = edge_area.sum()
    return correct, total


def soft_accuracy_edge_length(pred, soft_labels, edges_length, device):
    """
    This accuracy calculation is based on consider the edge length i.e. the correct predicted edge is weighted by its
    length
    :param pred: output array of hynet model
    :param soft_labels: soft labels for edge segmentation
    :param edges_length: length of edges of mesh
    :param device:  GPU/CPU
    :return: correct wdge weighted by edge length and total edge length
    """
    one_hot_tensor = torch.zeros(soft_labels.size()[0], soft_labels.size()[1]).to(device)
    one_hot_tensor[torch.tensor(range(soft_labels.size()[0])), pred] = 1
    edges_length_sum = edges_length.sum()
    edges_length = edges_length
    correct = soft_labels * one_hot_tensor
    correct_length = correct * edges_length.reshape((len(edges_length), 1))
    return correct_length.sum(), edges_length_sum

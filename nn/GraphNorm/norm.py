# This script from the official github of GraphNorm paper https://github.com/lsj2408/GraphNorm
# has been modified to keep moving average and std for evaluation mode
import torch
import torch.nn as nn

class Norm(nn.Module):

    def __init__(self, norm_type, hidden_dim=64, eps=1e-6, momentum=0.1, print_info=None):
        super(Norm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

            self.register_buffer("moving_avg", torch.zeros(1, hidden_dim))
            self.register_buffer("moving_var", torch.ones(1, hidden_dim))
            self.register_buffer("eps", torch.tensor(eps))
            self.register_buffer("momentum", torch.tensor(momentum))

    def forward(self, graph, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        if self.training:
            batch_list = graph.batch_num_nodes()
            batch_size = len(batch_list)
            # batch_list = torch.Tensor(batch_list).long().to(tensor.device)
            batch_list = torch.as_tensor(batch_list).long().to(tensor.device)
            batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
            batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
            mean = torch.zeros(batch_size, *tensor.shape[1:], dtype=torch.float64).to(tensor.device)
            mean = mean.scatter_add_(0, batch_index, tensor)
            mean = (mean.T / batch_list).T
            sub = tensor - mean * self.mean_scale
            std = torch.zeros(batch_size, *tensor.shape[1:], dtype=torch.float64).to(tensor.device)
            std = std.scatter_add_(0, batch_index, sub.pow(2))
            std = ((std.T / batch_list).T + 1e-6).sqrt()

            with torch.no_grad():
                self.moving_avg.mul_(1 - self.momentum).add_(mean * self.momentum)
                self.moving_var.mul_(1 - self.momentum).add_(std * self.momentum)
        else:
            mean = self.moving_avg
            sub = tensor - mean * self.mean_scale
            std = self.moving_var

        return self.weight * sub / std + self.bias
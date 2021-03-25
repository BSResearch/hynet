import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from .GraphNorm.norm import Norm


class AdditiveAttention(nn.Module):
    """
    Second-level attention layer
    Method: Additive attention
    Implementation of Additive Attention by https://arxiv.org/pdf/1409.0473.pdf
    The table of result in the paper is based on this class.
    """

    def __init__(self, in_size, hidden_size=128):
        super(AdditiveAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        y = self.project(z)
        beta = torch.softmax(y, dim=1)
        s = (beta * z).sum(1)
        return s


class MultiHeadAttention(nn.Module):
    """
    Second-level attention layer
    Method: scaled dot-product by https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, embed_size, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads=num_head, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, query, key, value):
        result, _ = self.self_attention(query, key, value)
        # result = query + self.dropout(result)
        result = self.dropout(result)
        out = self.layer_norm(result)
        return out


class NEFLayer(nn.Module):
    """
    NEF embedding layer:
    """

    def __init__(self, canonical_etypes, in_size, out_size, layer_num_heads, dropout, norm_type):
        super(NEFLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.node_query_adaptor = nn.ModuleList()
        self.node_query_adaptor.append(nn.Linear(in_size['node'], out_size * layer_num_heads))
        self.node_query_adaptor.append(nn.ReLU())
        self.node_query_adaptor.append(nn.Dropout(0.2))
        # self.node_query_adaptor.append(nn.BatchNorm1d(out_size* layer_num_heads))
        self.node_query_adaptor_func = nn.Sequential(*self.node_query_adaptor)

        self.edge_query_adaptor = nn.ModuleList()
        self.edge_query_adaptor.append(nn.Linear(in_size['edge'], out_size * layer_num_heads))
        self.edge_query_adaptor.append(nn.ReLU())
        self.edge_query_adaptor.append(nn.Dropout(0.2))
        # self.node_query_adaptor.append(nn.BatchNorm1d(out_size* layer_num_heads))
        self.edge_query_adaptor_func = nn.Sequential(*self.edge_query_adaptor)

        self.face_query_adaptor = nn.ModuleList()
        self.face_query_adaptor.append(nn.Linear(in_size['face'], out_size * layer_num_heads))
        self.face_query_adaptor.append(nn.ReLU())
        self.face_query_adaptor.append(nn.Dropout(0.2))
        # self.node_query_adaptor.append(nn.BatchNorm1d(out_size* layer_num_heads))
        self.face_query_adaptor_func = nn.Sequential(*self.face_query_adaptor)

        for i in range(len(canonical_etypes)):
            src_in_size = in_size[canonical_etypes[i][0]]
            dst_in_size = in_size[canonical_etypes[i][2]]
            if canonical_etypes[i][0] == canonical_etypes[i][2]:
                self.gat_layers.append(GATConv(src_in_size, out_size, layer_num_heads, feat_drop=0,
                                               attn_drop=dropout, negative_slope=0.2, residual=False, activation=F.elu,
                                               allow_zero_in_degree=True))
            else:
                self.gat_layers.append(GATConv((src_in_size, dst_in_size), out_size, layer_num_heads, feat_drop=0,
                                               attn_drop=dropout, negative_slope=0.2, residual=False, activation=F.elu,
                                               allow_zero_in_degree=True))

        self.additive_self_attention = True
        self.scaled_dot_product_attention = not self.additive_self_attention
        if self.additive_self_attention:
            self.semantic_attention_node = AdditiveAttention(in_size=out_size * layer_num_heads)
            self.semantic_attention_edge = AdditiveAttention(in_size=out_size * layer_num_heads)
            self.semantic_attention_face = AdditiveAttention(in_size=out_size * layer_num_heads)

        if self.scaled_dot_product_attention:
            # If self.additive_self_attention == False we use scaled dot-product self attention
            # This section is for experiment for scaled dot-product self attention.
            # The architecture and table reported in the paper is not based on this section
            self.multihead_attention_node = MultiHeadAttention(embed_size=out_size * layer_num_heads,
                                                               num_head=8, dropout=0.35)
            self.multihead_attention_edge = MultiHeadAttention(embed_size=out_size * layer_num_heads,
                                                               num_head=8, dropout=0.35)
            self.multihead_attention_face = MultiHeadAttention(embed_size=out_size * layer_num_heads,
                                                               num_head=8, dropout=0.35)

        # GraphNorm layer
        self.norm_node = Norm(norm_type, out_size * layer_num_heads)
        self.norm_edge = Norm(norm_type, out_size * layer_num_heads)
        self.norm_face = Norm(norm_type, out_size * layer_num_heads)

        self.canonical_etypes = canonical_etypes

    def forward(self, g, sub_g_edge=None):
        g_new_list = []
        gat_embeddings = {'node': [], 'edge': [], 'face': []}

        # The first member of the gat_embeddings list for each key in dict is computed by applying an MLP on the
        # hybrid graph vertices features without considering its connection to the neighboring node
        gat_embeddings['node'].append(self.node_query_adaptor_func(g.nodes['node'].data['geometric_feat']))
        gat_embeddings['edge'].append(self.edge_query_adaptor_func(g.nodes['edge'].data['geometric_feat']))
        gat_embeddings['face'].append(self.face_query_adaptor_func(g.nodes['face'].data['geometric_feat']))
        for i in range(len(self.canonical_etypes)):
            g_new = g.edge_type_subgraph([self.canonical_etypes[i][1]])
            g_new_list.append(g_new)

        for i in range(len(self.canonical_etypes)):
            g_new = g_new_list[i]
            if self.canonical_etypes[i][0] == self.canonical_etypes[i][2]:
                gat_embeddings[self.canonical_etypes[i][2]].append(
                    self.gat_layers[i](g_new, g_new.ndata['geometric_feat']).flatten(1))
            else:
                gat_embeddings[self.canonical_etypes[i][2]].append(
                    self.gat_layers[i](g_new, (g_new.ndata['geometric_feat'][self.canonical_etypes[i][0]],
                                               g_new.ndata['geometric_feat'][self.canonical_etypes[i][2]])).flatten(1))

        nef_out = {}
        if self.additive_self_attention:
            gat_embeddings['node'] = torch.stack(gat_embeddings['node'], dim=1)
            gat_embeddings['edge'] = torch.stack(gat_embeddings['edge'], dim=1)
            gat_embeddings['face'] = torch.stack(gat_embeddings['face'], dim=1)
            nef_out['node'] = self.semantic_attention_node(gat_embeddings['node'])
            nef_out['edge'] = self.semantic_attention_edge(gat_embeddings['edge'])
            nef_out['face'] = self.semantic_attention_face(gat_embeddings['face'])

            g.nodes['node'].data['geometric_feat'] = nef_out['node']
            g.nodes['edge'].data['geometric_feat'] = nef_out['edge']
            g.nodes['face'].data['geometric_feat'] = nef_out['face']

        if self.scaled_dot_product_attention:
            node_query = gat_embeddings['node'][0]
            node_query = node_query.unsqueeze_(-1)
            node_query = node_query.permute(2, 0, 1)
            # # print(node_query.size())
            nef_out['node'] = self.multihead_attention_node(node_query,
                                                            gat_embeddings['node'][1:],
                                                            gat_embeddings['node'][1:])

            edge_query = gat_embeddings['edge'][0]
            edge_query = edge_query.unsqueeze_(-1)
            edge_query = edge_query.permute(2, 0, 1)
            nef_out['edge'] = self.multihead_attention_edge(edge_query,
                                                            gat_embeddings['edge'][1:],
                                                            gat_embeddings['edge'][1:])

            face_query = gat_embeddings['face'][0]
            face_query = face_query.unsqueeze_(-1)
            face_query = face_query.permute(2, 0, 1)

            nef_out['face'] = self.multihead_attention_face(face_query,
                                                            gat_embeddings['face'][1:],
                                                            gat_embeddings['face'][1:])
        # # print(torch.squeeze(semantic_out['node']).size())
            g.nodes['node'].data['geometric_feat'] = torch.squeeze(nef_out['node'])
            g.nodes['edge'].data['geometric_feat'] = torch.squeeze(nef_out['edge'])
            g.nodes['face'].data['geometric_feat'] = torch.squeeze(nef_out['face'])

        # normalization
        sub_g_node = g.node_type_subgraph(['node'])
        node_geometric_feat = self.norm_node(sub_g_node, sub_g_node.nodes['node'].data['geometric_feat'])
        sub_g_edge = g.node_type_subgraph(['edge'])
        edge_geometric_feat = self.norm_edge(sub_g_edge, sub_g_edge.nodes['edge'].data['geometric_feat'])
        sub_g_face = g.node_type_subgraph(['face'])
        face_geometric_feat = self.norm_face(sub_g_face, sub_g_face.nodes['face'].data['geometric_feat'])

        g.nodes['node'].data['geometric_feat'] = node_geometric_feat
        g.nodes['edge'].data['geometric_feat'] = edge_geometric_feat
        g.nodes['face'].data['geometric_feat'] = face_geometric_feat
        return g

class HyNet(nn.Module):
    """
    HyNet network
    """

    def __init__(self, canonical_etypes, in_size, embedding_hidden_size, hidden_size, num_heads, fcn_hidden_size,
                 out_size, gat_dropout, fcn_dropout, classification_element, norm_type='gn'):
        super(HyNet, self).__init__()

        self.classification_element = classification_element

        # A feature transform sub-network  to embed different features in the same feature space given that the feature
        # vectors of the nodes in H (Hybrid Graph) have distinct features.
        # node feature transform
        self.node_embedding_layer = nn.ModuleList()
        self.node_embedding_layer.append(nn.Linear(in_size['node'], embedding_hidden_size[0]))
        self.node_embedding_layer.append(nn.ReLU())
        self.node_embedding_layer.append(nn.Dropout(0.2))
        self.node_embedding_layer.append(nn.BatchNorm1d(embedding_hidden_size[0]))
        for j in range(1, len(embedding_hidden_size)):
            self.node_embedding_layer.append(nn.Linear(embedding_hidden_size[j - 1], embedding_hidden_size[j]))
            self.node_embedding_layer.append(nn.ReLU())
            self.node_embedding_layer.append(nn.Dropout(0.2))
            self.node_embedding_layer.append(nn.BatchNorm1d(embedding_hidden_size[j]))

        self.node_embedding_layer.append(nn.Linear(embedding_hidden_size[-1], hidden_size[0]))
        self.node_embedding_layer.append(nn.ReLU())
        self.node_embedding_layer.append(nn.Dropout(0.2))
        self.node_embedding_layer.append(nn.BatchNorm1d(hidden_size[0]))

        self.node_embedding = nn.Sequential(*self.node_embedding_layer)

        # # edge feature transform
        self.edge_embedding_layer = nn.ModuleList()
        self.edge_embedding_layer.append(nn.Linear(in_size['edge'], embedding_hidden_size[0]))
        self.edge_embedding_layer.append(nn.ReLU())
        self.edge_embedding_layer.append(nn.Dropout(0.2))
        self.edge_embedding_layer.append(nn.BatchNorm1d(embedding_hidden_size[0]))
        for j in range(1, len(embedding_hidden_size)):
            self.edge_embedding_layer.append(nn.Linear(embedding_hidden_size[j - 1], embedding_hidden_size[j]))
            self.edge_embedding_layer.append(nn.ReLU())
            self.edge_embedding_layer.append(nn.Dropout(0.2))
            self.edge_embedding_layer.append(nn.BatchNorm1d(embedding_hidden_size[j]))

        self.edge_embedding_layer.append(nn.Linear(embedding_hidden_size[-1], hidden_size[0]))
        self.edge_embedding_layer.append(nn.ReLU())
        self.edge_embedding_layer.append(nn.Dropout(0.2))
        self.edge_embedding_layer.append(nn.BatchNorm1d(hidden_size[0]))

        self.edge_embedding = nn.Sequential(*self.edge_embedding_layer)

        # # face feature transform
        self.face_embedding_layer = nn.ModuleList()
        self.face_embedding_layer.append(nn.Linear(in_size['face'], embedding_hidden_size[0]))
        self.face_embedding_layer.append(nn.ReLU())
        self.face_embedding_layer.append(nn.Dropout(0.2))
        self.face_embedding_layer.append(nn.BatchNorm1d(embedding_hidden_size[0]))
        for j in range(1, len(embedding_hidden_size)):
            self.face_embedding_layer.append(nn.Linear(embedding_hidden_size[j - 1], embedding_hidden_size[j]))
            self.face_embedding_layer.append(nn.ReLU())
            self.face_embedding_layer.append(nn.Dropout(0.2))
            self.face_embedding_layer.append(nn.BatchNorm1d(embedding_hidden_size[j]))
        self.face_embedding_layer.append(nn.Linear(embedding_hidden_size[-1], hidden_size[0]))
        self.face_embedding_layer.append(nn.ReLU())
        self.face_embedding_layer.append(nn.Dropout(0.2))
        self.face_embedding_layer.append(nn.BatchNorm1d(hidden_size[0]))
        self.face_embedding = nn.Sequential(*self.face_embedding_layer)

        in_size = {'node': hidden_size[0],
                   'edge': hidden_size[0],
                   'face': hidden_size[0]}

        self.layers = nn.ModuleList()
        self.layers.append(NEFLayer(canonical_etypes, in_size, hidden_size[0], num_heads[0], gat_dropout, norm_type))
        for l in range(1, len(num_heads)):
            in_size = {'node': hidden_size[l - 1] * num_heads[l - 1],
                       'edge': hidden_size[l - 1] * num_heads[l - 1],
                       'face': hidden_size[l - 1] * num_heads[l - 1]}
            self.layers.append(NEFLayer(canonical_etypes, in_size,
                                        hidden_size[l], num_heads[l], gat_dropout, norm_type))

        self.predict = nn.ModuleList()
        self.predict.append(nn.Linear(hidden_size[-1] * num_heads[-1], fcn_hidden_size[0]))
        self.predict.append(nn.ReLU())
        self.predict.append(nn.Dropout(fcn_dropout))
        self.predict.append(nn.BatchNorm1d(fcn_hidden_size[0]))
        for p in range(1, len(fcn_hidden_size)):
            self.predict.append(nn.Linear(fcn_hidden_size[p - 1], fcn_hidden_size[p]))
            self.predict.append(nn.ReLU())
            self.predict.append(nn.Dropout(fcn_dropout))
            self.predict.append(nn.BatchNorm1d(fcn_hidden_size[p]))
        self.predict.append(nn.Linear(fcn_hidden_size[-1], out_size))
        self.predict_nn = nn.Sequential(*self.predict)

    def forward(self, g):
        """
        HyNet model:
        :param g: Hybrid graphs generated by mesh-to-hybrid graph converter
        :return: prediction for one type of constituting elements of a mesh i.e. node, edge and face
        """
        g.nodes['node'].data['geometric_feat'] = self.node_embedding(g.nodes['node'].data['geometric_feat'])
        g.nodes['edge'].data['geometric_feat'] = self.edge_embedding(g.nodes['edge'].data['geometric_feat'])
        g.nodes['face'].data['geometric_feat'] = self.face_embedding(g.nodes['face'].data['geometric_feat'])

        for gnn in self.layers:
            g = gnn(g)

        out = {self.classification_element: self.predict_nn(g.ndata['geometric_feat'][self.classification_element])}
        return out

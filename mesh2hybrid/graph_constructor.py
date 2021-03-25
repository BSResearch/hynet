import numpy as np
import pymesh
import torch
import torch_geometric
import warnings
import dgl
import math


class GraphConstructor:

    def __init__(self, input_mesh, mesh_name, data_edges, edge_area, edges_seg_labels, edges_soft_seg_labels,
                 faces_seg_labels, nodes_seg_labels, filename_to_save, graph_label):
        self.__graph = None
        self.__input_mesh = input_mesh
        self.__mesh_name = mesh_name
        self.__input_mesh.enable_connectivity()
        self.__mesh_nodes, self.__mesh_edges = pymesh.mesh_to_graph(self.__input_mesh)
        self.__mesh_faces = self.__input_mesh.faces
        self.__filename_to_save = filename_to_save
        self.__graph_label = graph_label
        self.__data_edges = data_edges
        self.__edges_seg_labels = edges_seg_labels
        self.__edges_soft_seg_labels = edges_soft_seg_labels
        self.__faces_seg_labels = faces_seg_labels
        self.__nodes_seg_labels = nodes_seg_labels
        self.__edge_area = edge_area
        self.two_face_neighbor = True

    def create_graphs(self):

        node_number_total = self.__mesh_nodes.shape[0] + self.__mesh_edges.shape[0] + self.__mesh_faces.shape[0]

        # node Features
        self.__input_mesh.add_attribute("vertex_valance")
        node_features = np.concatenate((self.__mesh_nodes, self.__input_mesh.get_vertex_attribute("vertex_valance")),
                                       axis=1)
        node_labels = self.__nodes_seg_labels - 1

        # face_features
        self.__input_mesh.add_attribute("face_normal")
        self.__input_mesh.add_attribute("face_area")
        self.__input_mesh.add_attribute("face_circumcenter")
        face_circumcenter = self.__input_mesh.get_attribute("face_circumcenter").reshape(
            (self.__mesh_faces.shape[0], 3))
        self.__input_mesh.add_attribute("face_circumscribed_radius")

        face_circumscribed_radius = np.linalg.norm(face_circumcenter - self.__mesh_nodes[self.__mesh_faces[:, 0], :],
                                                   axis=1)
        self.__input_mesh.set_attribute("face_circumscribed_radius", face_circumscribed_radius)
        face_normal = self.__input_mesh.get_attribute("face_normal").reshape((self.__mesh_faces.shape[0], 3))
        face_area = self.__input_mesh.get_attribute("face_area").reshape((self.__mesh_faces.shape[0], 1))
        max_face_area = max(face_area)
        face_area = face_area / max_face_area
        max_face_radius = max(face_circumscribed_radius)
        face_circumscribed_radius = face_circumscribed_radius / max_face_radius
        face_circumscribed_radius = face_circumscribed_radius.reshape((self.__mesh_faces.shape[0], 1))
        face_features = np.concatenate((face_normal, face_area, face_circumcenter, face_circumscribed_radius), axis=1)

        face_labels = self.__faces_seg_labels
        # edge feature
        edge_lengths = np.zeros((self.__mesh_edges.shape[0], 1))
        edge_features = np.zeros((self.__mesh_edges.shape[0], 5))
        edge_labels = -np.ones(self.__mesh_edges.shape[0], dtype=np.int32)
        edge_areas = np.zeros(self.__mesh_edges.shape[0], dtype=np.float)
        edge_soft_labels = -np.ones((self.__mesh_edges.shape[0], self.__edges_soft_seg_labels.shape[1]), dtype=np.int32)
        for edge_idx in range(self.__mesh_edges.shape[0]):
            edge_features[edge_idx] = self.get_edge_feature(edge_idx, face_normal, face_area)
            edge_lengths[edge_idx] = self.get_edge_length(edge_idx)
            if not (self.two_face_neighbor):
                break
            edge_labels[edge_idx], edge_soft_labels[edge_idx], edge_areas[edge_idx] = self.get_edge_label(edge_idx)
        if not (self.two_face_neighbor):
            return True

        # get nodes adjacent to nodes
        nodes_nodes_adjacency = self.get_vertices_influenced_by_vertices()

        # get edge nodes (edge represented by node) adjacent to nns (node represented by node)
        nodes_edges_adjacency = self.get_vertices_influenced_by_edges()

        # get face nodes ( face represented by node) adjacent to nns
        nodes_faces_adjacency = self.get_vertices_influenced_by_face()

        # get edge nodes adjacent to another edge node and sharing a single node or node and a plane
        edges_edges_sharing_node_and_plane, edges_edges_sharing_node = self.get_edges_influenced_by_edges()

        # get face nodes adjacent to edge node
        edges_faces_adjacency = self.get_edges_influenced_by_faces()

        # get nodes adjacent to faces : Each face has three adjacent nodes
        faces_nodes_adjacency = self.get_face_influenced_by_node()

        # get edge nodes adjacent to face node : Each face is adjacent to three edges
        faces_edges_adjacency = self.get_face_influenced_by_edges()

        # get faces nodes adjacent to face nodes: Each face is adjacent to three face sharing an edge
        faces_faces_adjacency = self.get_face_influenced_by_faces()

        graph_src_dst_data = {
            ('node', 'influences (nn)', 'node'): (torch.tensor(nodes_nodes_adjacency[:, 0]),
                                                  torch.tensor(nodes_nodes_adjacency[:, 1])),
            ('edge', 'influences (ne)', 'node'): (torch.tensor(nodes_edges_adjacency[:, 1]),
                                                  torch.tensor(nodes_edges_adjacency[:, 0])),
            ('face', 'influences (nf)', 'node'): (torch.tensor(nodes_faces_adjacency[:, 1]),
                                                  torch.tensor(nodes_faces_adjacency[:, 0])),
            ('node', 'influences (en)', 'edge'): (torch.tensor(nodes_edges_adjacency[:, 0]),
                                                  torch.tensor(nodes_edges_adjacency[:, 1])),
            ('edge', 'influences (ee_s_n)', 'edge'): (torch.tensor(edges_edges_sharing_node[:, 0]),
                                                      torch.tensor(edges_edges_sharing_node[:, 1])),
            ('edge', 'influences (ee_s_np)', 'edge'): (torch.tensor(edges_edges_sharing_node_and_plane[:, 0]),
                                                       torch.tensor(edges_edges_sharing_node_and_plane[:, 1])),
            ('face', 'influences (ef)', 'edge'): (torch.tensor(edges_faces_adjacency[:, 1]),
                                                  torch.tensor(edges_faces_adjacency[:, 0])),
            ('node', 'influences (fn)', 'face'): (torch.tensor(faces_nodes_adjacency[:, 1]),
                                                  torch.tensor(faces_nodes_adjacency[:, 0])),
            ('edge', 'influences (fe)', 'face'): (torch.tensor(faces_edges_adjacency[:, 1]),
                                                  torch.tensor(faces_edges_adjacency[:, 0])),
            ('face', 'influences (ff)', 'face'): (torch.tensor(faces_faces_adjacency[:, 0]),
                                                  torch.tensor(faces_faces_adjacency[:, 1]))
        }

        hybrid_graph = dgl.heterograph(graph_src_dst_data)
        hybrid_graph.ndata['edge_length'] = {
            'edge': torch.tensor(edge_lengths)
        }
        hybrid_graph.ndata['edge_area'] = {
            'edge': torch.tensor(edge_areas)
        }

        hybrid_graph.ndata['init_geometric_feat'] = {
            'node': torch.tensor(node_features),
            'edge': torch.tensor(edge_features),
            'face': torch.tensor(face_features)
        }
        hybrid_graph.ndata['geometric_feat'] = {
            'node': torch.tensor(node_features),
            'edge': torch.tensor(edge_features),
            'face': torch.tensor(face_features)
        }

        hybrid_graph.ndata['label'] = {
            'node': torch.tensor(node_labels),
            'edge': torch.tensor(edge_labels),
            'face': torch.tensor(face_labels)
        }

        hybrid_graph.ndata['edge_soft_label'] = {
            'edge': torch.tensor(edge_soft_labels)
        }

        dgl.data.utils.save_graphs(self.__filename_to_save, [hybrid_graph], self.__graph_label)
        return hybrid_graph

    def get_edge_feature(self, edge_idx, face_normal, face_area):
        n_0 = self.__mesh_edges[edge_idx, 0]
        n_1 = self.__mesh_edges[edge_idx, 1]
        n_0_adjacent_face_indices = set(self.__input_mesh.get_vertex_adjacent_faces(n_0))
        n_1_adjacent_faces_indices = set(self.__input_mesh.get_vertex_adjacent_faces(n_1))
        edge_adjacent_faces_indices = np.sort(list(n_0_adjacent_face_indices & n_1_adjacent_faces_indices))
        if len(edge_adjacent_faces_indices) == 2:
            face_idx_0 = edge_adjacent_faces_indices[0]
            face_idx_1 = edge_adjacent_faces_indices[1]
        if len(edge_adjacent_faces_indices) != 2:
            print(f"mesh {self.__mesh_name} : edge {edge_idx} has {len(edge_adjacent_faces_indices)} adjacent face.")
            self.two_face_neighbor = False
            return True
        # get dihedral angle

        face_0_normal = face_normal[face_idx_0]
        face_1_normal = face_normal[face_idx_1]
        cos_theta = min(np.dot(face_0_normal, face_1_normal), 1)
        cos_theta = max(-1, cos_theta)
        dihedral_angle = np.expand_dims(np.pi - np.arccos(cos_theta), axis=0)

        # get edge height ratio
        edge_norm = np.expand_dims(np.linalg.norm(self.__mesh_nodes[n_0] - self.__mesh_nodes[n_1]), axis=0)
        face_0_area = face_area[face_idx_0]
        face_1_area = face_area[face_idx_1]
        edge_height_ratios = edge_norm ** 2 / np.array([2 * face_0_area, 2 * face_1_area])
        edge_height_ratios = np.squeeze(edge_height_ratios)

        # get opposite angle
        opposite_vertex_in_face_idx_0 = self.__mesh_nodes[list(set(self.__mesh_faces[face_idx_0]) -
                                                               set(self.__mesh_edges[edge_idx]))[0]]
        opposite_vertex_in_face_idx_1 = self.__mesh_nodes[list(set(self.__mesh_faces[face_idx_1]) -
                                                               set(self.__mesh_edges[edge_idx]))[0]]
        edge_a = self.__mesh_nodes[n_0] - opposite_vertex_in_face_idx_0
        edge_a = edge_a / np.linalg.norm(edge_a)
        edge_b = self.__mesh_nodes[n_1] - opposite_vertex_in_face_idx_0
        edge_b = edge_b / np.linalg.norm(edge_b)
        gamma_1 = np.arccos(np.dot(edge_a, edge_b))

        edge_c = self.__mesh_nodes[n_0] - opposite_vertex_in_face_idx_1
        edge_c = edge_c / np.linalg.norm(edge_c)
        edge_d = self.__mesh_nodes[n_1] - opposite_vertex_in_face_idx_1
        edge_d = edge_d / np.linalg.norm(edge_d)
        gamma_2 = np.arccos(np.dot(edge_c, edge_d))
        opposite_angles = np.array([gamma_1, gamma_2])

        return np.concatenate((dihedral_angle, opposite_angles, edge_height_ratios))

    def get_vertices_influenced_by_vertices(self):
        nodes_nodes_adjacency = np.concatenate((self.__mesh_edges, self.__mesh_edges[:, ::-1]), axis=0)
        return nodes_nodes_adjacency

    def get_vertex_adjacent_edges(self, n_idx):
        return np.where((self.__mesh_edges[:, 0] == n_idx) | (self.__mesh_edges[:, 1] == n_idx))[0]

    def get_vertices_influenced_by_edges(self):
        nodes_edges_adjacency = np.array([], dtype=np.int32).reshape((0, 2))
        for n_idx in range(self.__mesh_nodes.shape[0]):
            node_adjacent_edges = np.expand_dims(self.get_vertex_adjacent_edges(n_idx), axis=1)
            node_edges_adjacency = np.concatenate((n_idx * np.ones((len(node_adjacent_edges), 1), dtype=np.int32),
                                                   node_adjacent_edges), axis=1)
            nodes_edges_adjacency = np.vstack((nodes_edges_adjacency, node_edges_adjacency))
        return nodes_edges_adjacency

    def get_vertices_influenced_by_face(self):
        nodes_faces_adjacency = np.array([], dtype=np.int32).reshape((0, 2))
        for n_idx in range(self.__mesh_nodes.shape[0]):
            node_adjacent_faces = np.expand_dims(self.__input_mesh.get_vertex_adjacent_faces(n_idx), axis=1)
            node_faces_adjacency = np.concatenate((n_idx * np.ones((len(node_adjacent_faces), 1), dtype=np.int32),
                                                   node_adjacent_faces), axis=1)
            nodes_faces_adjacency = np.vstack((nodes_faces_adjacency, node_faces_adjacency))
        return nodes_faces_adjacency

    def get_edge_adjacent_faces(self, edge_idx):
        n_0 = self.__mesh_edges[edge_idx, 0]
        n_1 = self.__mesh_edges[edge_idx, 1]
        n_0_adjacent_face_indices = set(self.__input_mesh.get_vertex_adjacent_faces(n_0))
        n_1_adjacent_faces_indices = set(self.__input_mesh.get_vertex_adjacent_faces(n_1))
        edge_adjacent_faces_indices = np.array(list(n_0_adjacent_face_indices & n_1_adjacent_faces_indices))
        return edge_adjacent_faces_indices

    def get_edges_influenced_by_faces(self):
        edges_faces_adjacency = np.array([], dtype=np.int32).reshape((0, 2))
        for edge_idx in range(self.__mesh_edges.shape[0]):
            edge_adjacent_faces = self.get_edge_adjacent_faces(edge_idx)
            edge_adjacent_faces = np.expand_dims(edge_adjacent_faces, axis=1)
            edge_faces_adjacency = np.concatenate((edge_idx * np.ones((len(edge_adjacent_faces), 1), dtype=np.int32),
                                                   edge_adjacent_faces), axis=1)

            edges_faces_adjacency = np.vstack((edges_faces_adjacency, edge_faces_adjacency))
        return edges_faces_adjacency

    def get_edges_influenced_by_edges(self):
        edges_edges_sharing_node = np.array([], dtype=np.int32).reshape((0, 2))
        edges_edges_sharing_node_and_plane = np.array([], dtype=np.int32).reshape((0, 2))
        for edge_idx in range(self.__mesh_edges.shape[0]):
            edge_adjacent_edges_s_np, edge_adjacent_edges_s_n = self.get_edge_adjacent_edges(edge_idx)
            edge_adjacent_edges_s_np = np.expand_dims(edge_adjacent_edges_s_np, axis=1)
            edge_edges_adjacency_s_np = np.concatenate((edge_idx * np.ones((len(edge_adjacent_edges_s_np), 1),
                                                                           dtype=np.int32),
                                                        edge_adjacent_edges_s_np), axis=1)
            edges_edges_sharing_node_and_plane = np.vstack(
                (edges_edges_sharing_node_and_plane, edge_edges_adjacency_s_np
                 ))

            edge_adjacent_edges_s_n = np.expand_dims(edge_adjacent_edges_s_n, axis=1)
            edge_edges_adjacency_s_n = np.concatenate((edge_idx * np.ones((len(edge_adjacent_edges_s_n), 1),
                                                                          dtype=np.int32),
                                                       edge_adjacent_edges_s_n), axis=1)
            edges_edges_sharing_node = np.vstack((edges_edges_sharing_node, edge_edges_adjacency_s_n))
        return edges_edges_sharing_node_and_plane, edges_edges_sharing_node

    def get_edge_adjacent_edges(self, edge_idx):
        n_0 = self.__mesh_edges[edge_idx, 0]
        n_1 = self.__mesh_edges[edge_idx, 1]
        all_adjacent_edges = (set(self.get_vertex_adjacent_edges(n_0)) | set(self.get_vertex_adjacent_edges(n_1))) - \
                             {edge_idx}
        edge_adjacent_faces = self.get_edge_adjacent_faces(edge_idx)
        face_idx_0 = edge_adjacent_faces[0]
        face_idx_1 = edge_adjacent_faces[1]

        # find opposite nodes of the edge_idx
        n_2 = list(set(self.__mesh_faces[face_idx_0]) - set(self.__mesh_edges[edge_idx]))[0]
        n_3 = list(set(self.__mesh_faces[face_idx_1]) - set(self.__mesh_edges[edge_idx]))[0]

        adjacent_edges_share_node_plane = []
        assert (not (n_0 == n_2)), f"The mesh {self.__mesh_name} is not two manifold"
        if n_0 < n_2:
            e_share_n_p_idx_0 = np.where((self.__mesh_edges[:, 0] == n_0) & (self.__mesh_edges[:, 1] == n_2))[0]
        else:
            e_share_n_p_idx_0 = np.where((self.__mesh_edges[:, 0] == n_2) & (self.__mesh_edges[:, 1] == n_0))[0]
        adjacent_edges_share_node_plane.append(e_share_n_p_idx_0[0])

        assert (n_0 != n_3), f"The mesh {self.__mesh_name} is not two manifold"
        if n_0 < n_3:
            e_share_n_p_idx_1 = np.where((self.__mesh_edges[:, 0] == n_0) & (self.__mesh_edges[:, 1] == n_3))[0]
        else:
            e_share_n_p_idx_1 = np.where((self.__mesh_edges[:, 0] == n_3) & (self.__mesh_edges[:, 1] == n_0))[0]
        adjacent_edges_share_node_plane.append(e_share_n_p_idx_1[0])

        assert (n_1 != n_2), f"The mesh {self.__mesh_name} is not two manifold"
        if n_1 < n_2:
            e_share_n_p_idx_2 = np.where((self.__mesh_edges[:, 0] == n_1) & (self.__mesh_edges[:, 1] == n_2))[0]
        else:
            e_share_n_p_idx_2 = np.where((self.__mesh_edges[:, 0] == n_2) & (self.__mesh_edges[:, 1] == n_1))[0]
        adjacent_edges_share_node_plane.append(e_share_n_p_idx_2[0])

        assert (n_1 != n_3), f"The mesh {self.__mesh_name} is not two manifold"
        if n_1 < n_3:
            e_share_n_p_idx_3 = np.where((self.__mesh_edges[:, 0] == n_1) & (self.__mesh_edges[:, 1] == n_3))[0]
        else:
            e_share_n_p_idx_3 = np.where((self.__mesh_edges[:, 0] == n_3) & (self.__mesh_edges[:, 1] == n_1))[0]
        adjacent_edges_share_node_plane.append(e_share_n_p_idx_3[0])

        adjacent_edges_share_only_node = np.array(list(all_adjacent_edges - set(adjacent_edges_share_node_plane)))

        return np.array(adjacent_edges_share_node_plane), adjacent_edges_share_only_node

    def get_face_influenced_by_node(self):
        face_idxes = np.expand_dims(np.array(range(len(self.__mesh_faces))), axis=1)
        face_idxes = np.concatenate((face_idxes, face_idxes, face_idxes), axis=1)
        face_idxes = np.expand_dims(face_idxes, axis=2)
        face_nodes = np.expand_dims(self.__mesh_faces, axis=2)
        faces_adjacent_nodes = np.concatenate((face_idxes, face_nodes), axis=2)
        faces_adjacent_nodes = faces_adjacent_nodes.reshape(
            (self.__mesh_faces.shape[0] * self.__mesh_faces.shape[1], 2))
        return faces_adjacent_nodes

    def get_face_adjacent_edges(self, face_idx):
        node_0, node_1, node_2 = self.__mesh_faces[face_idx]
        if node_0 < node_1:
            edge_0_n_0, edge_0_n_1 = node_0, node_1
        else:
            edge_0_n_0, edge_0_n_1 = node_1, node_0

        if node_1 < node_2:
            edge_1_n_0, edge_1_n_1 = node_1, node_2
        else:
            edge_1_n_0, edge_1_n_1 = node_2, node_1

        if node_2 < node_0:
            edge_2_n_0, edge_2_n_1 = node_2, node_0
        else:
            edge_2_n_0, edge_2_n_1 = node_0, node_2

        edge_0_idx = np.where((self.__mesh_edges[:, 0] == edge_0_n_0) & (self.__mesh_edges[:, 1] == edge_0_n_1))[0]
        edge_1_idx = np.where((self.__mesh_edges[:, 0] == edge_1_n_0) & (self.__mesh_edges[:, 1] == edge_1_n_1))[0]
        edge_2_idx = np.where((self.__mesh_edges[:, 0] == edge_2_n_0) & (self.__mesh_edges[:, 1] == edge_2_n_1))[0]
        return np.array([edge_0_idx, edge_1_idx, edge_2_idx])

    def get_face_influenced_by_edges(self):
        faces_edges_adjacency = np.array([], dtype=np.int32).reshape((0, 2))
        for face_idx in range(len(self.__mesh_faces)):
            adjacent_edges = self.get_face_adjacent_edges(face_idx)
            face_adjacent_edges = np.concatenate((face_idx * np.ones((len(adjacent_edges), 1), dtype=np.int32),
                                                  adjacent_edges), axis=1)
            faces_edges_adjacency = np.vstack((faces_edges_adjacency, face_adjacent_edges))
        return faces_edges_adjacency

    def get_face_influenced_by_faces(self):
        faces_faces_adjacency = np.array([], dtype=np.int32).reshape((0, 2))
        for face_idx in range(len(self.__mesh_faces)):
            adjacent_faces = self.__input_mesh.get_face_adjacent_faces(face_idx)
            adjacent_faces = np.expand_dims(adjacent_faces, 1)
            face_adjacent_faces = np.concatenate((face_idx * np.ones((len(adjacent_faces), 1), dtype=np.int32),
                                                  adjacent_faces), axis=1)
            faces_faces_adjacency = np.vstack((faces_faces_adjacency, face_adjacent_faces))
        return faces_faces_adjacency

    def get_edge_label(self, idx):
        n_0 = self.__mesh_edges[idx, 0]
        n_1 = self.__mesh_edges[idx, 1]
        data_idx = np.where((self.__data_edges[:, 0] == n_0) & (self.__data_edges[:, 1] == n_1))[0][0]
        label = self.__edges_seg_labels[data_idx] - 1
        soft_labels = self.__edges_soft_seg_labels[data_idx]
        edge_area = self.__edge_area[data_idx]
        return label, soft_labels, edge_area

    def get_edge_length(self, edge_idx):
        n_0 = self.__mesh_edges[edge_idx, 0]
        n_1 = self.__mesh_edges[edge_idx, 1]
        edge = self.__mesh_nodes[n_0] - self.__mesh_nodes[n_1]
        edge_length = np.linalg.norm(edge)
        return edge_length

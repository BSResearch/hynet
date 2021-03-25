import pymesh
from util.meshes import preprocess_mesh
from .graph_constructor import GraphConstructor
import warnings
from util.augmentation import scale_verts, rotateY3D, rotateX3D, slide_verts
from util.util import is_mesh_file, mkdir
from .mesh2hybrid_parser import Mesh2HybridParser
import os
import numpy as np


def generate_augmented_graph_from_single_mesh(mesh_files_dict, graph_dir, prevent_nonmanifold_edges=True,
                                              num_of_slide_aug_for_each_sample=0, num_of_jitter_aug_for_each_sample=0,
                                              jitter_rotation=False, mean=0, var=0.005, coef=1, num_y_rotation=0,
                                              rotate_x_90_for_each_y=True, slide_vert_percentage=0.2):
    input_mesh = pymesh.load_mesh(mesh_files_dict['mesh'])

    # Remove duplicated vertices, duplicated faces, degenerate faces
    # and, if required, faces with non-manifold edges.
    input_mesh = preprocess_mesh(input_mesh=input_mesh, prevent_nonmanifold_edges=prevent_nonmanifold_edges)

    mesh_data = np.load(mesh_files_dict['npz'], encoding='latin1', allow_pickle=True)
    mesh_edges = mesh_data['edges']

    edges_seg_labels = np.loadtxt(open(mesh_files_dict['seg'], 'r'), dtype='float64')
    edges_soft_seg_labels = np.loadtxt(open(mesh_files_dict['soft_seg']), dtype='float64')
    edges_soft_seg_labels = np.array(edges_soft_seg_labels > 0, dtype='int32')
    edge_areas = mesh_data['edge_areas']
    faces_seg_labels = np.loadtxt(open(mesh_files_dict['face_seg'], 'r'), dtype='float64')
    nodes_seg_labels = np.loadtxt(open(mesh_files_dict['node_seg'], 'r'), dtype='float64')
    mesh_name = mesh_files_dict['mesh'].split('/')[-1]

    # Generate the hybrid graph for the original mesh before augmentation
    filename_to_save = os.path.join(graph_dir,
                                    mesh_files_dict['mesh'].split('/')[-1].split('.')[0] + '_0_nef_graph.bin')
    graph_constructor = GraphConstructor(input_mesh=input_mesh, mesh_name=mesh_name,
                                         data_edges=mesh_edges, edge_area=edge_areas,
                                         edges_seg_labels=edges_seg_labels, edges_soft_seg_labels=edges_soft_seg_labels,
                                         faces_seg_labels=faces_seg_labels, nodes_seg_labels=nodes_seg_labels,
                                         filename_to_save=filename_to_save, graph_label=None)
    hybrid_graph = graph_constructor.create_graphs()

    # Augmentation:  Randomly shifting the vertices to different locations on the mesh sur-face in the
    # close-to-planar surface region
    # Generate the hybrid graph for augmented mesh
    if slide_vert_percentage != 0:
        for i in range(num_of_slide_aug_for_each_sample):
            all_edges_has_two_faces, slided_mesh = slide_verts(input_mesh, slide_vert_percentage, mesh_name)
            if all_edges_has_two_faces:
                filename_to_save = os.path.join(graph_dir, mesh_files_dict['mesh'].split('/')[-1].split('.')[0] + '_'
                                                + str(i + 1) + '_slided_nef_graph.bin')
                graph_constructor = GraphConstructor(input_mesh=slided_mesh, mesh_name=mesh_name,
                                                     data_edges=mesh_edges, edge_area=edge_areas,
                                                     edges_seg_labels=edges_seg_labels,
                                                     edges_soft_seg_labels=edges_soft_seg_labels,
                                                     faces_seg_labels=faces_seg_labels,
                                                     nodes_seg_labels=nodes_seg_labels,
                                                     filename_to_save=filename_to_save,
                                                     graph_label=None)
                hybrid_graph = graph_constructor.create_graphs()

    # Augmentation: Adding a varying Gaussian noise to each vertex of the shape
    # Generate the hybrid graph for augmented mesh
    for i in range(num_of_jitter_aug_for_each_sample):
        augmented_mesh = scale_verts(input_mesh, mean, var, coef)
        filename_to_save = os.path.join(graph_dir, mesh_files_dict['mesh'].split('/')[-1].split('.')[0] + '_'
                                        + str(i + 1) + '_jittered_nef_graph.bin')
        graph_constructor = GraphConstructor.GraphConstructor(input_mesh=augmented_mesh, mesh_name=mesh_name,
                                                              data_edges=mesh_edges,
                                                              edge_area=edge_areas,
                                                              edges_seg_labels=edges_seg_labels,
                                                              edges_soft_seg_labels=edges_soft_seg_labels,
                                                              faces_seg_labels=faces_seg_labels,
                                                              nodes_seg_labels=nodes_seg_labels,
                                                              filename_to_save=filename_to_save,
                                                              graph_label=None)
        hybrid_graph = graph_constructor.create_graphs()

    # rotate mesh about the y axis and create a a graph for every theta = (2 * np.pi / num_y_rotation) degree
    for theta in np.linspace(0, 2 * np.pi, num_y_rotation)[1:]:
        rotated_mesh_y = rotateY3D(input_mesh, theta)
        y_theta_deg = int(np.rad2deg(theta))
        filename_to_save = os.path.join(graph_dir,
                                        mesh_files_dict['mesh'].split('/')[-1].split('.')[0] + '_y_rotation_' +
                                        str(y_theta_deg) + '_nef_graph.bin')
        graph_constructor = GraphConstructor.GraphConstructor(input_mesh=rotated_mesh_y, mesh_name=mesh_name,
                                                              data_edges=mesh_edges,
                                                              edge_area=edge_areas,
                                                              edges_seg_labels=edges_seg_labels,
                                                              edges_soft_seg_labels=edges_soft_seg_labels,
                                                              faces_seg_labels=faces_seg_labels,
                                                              nodes_seg_labels=nodes_seg_labels,
                                                              filename_to_save=filename_to_save,
                                                              graph_label=None)
        hybrid_graph = graph_constructor.create_graphs()
        # slide vertices for each rotated mesh for more augmentation
        if slide_vert_percentage != 0:
            all_edges_has_two_faces, slided_mesh = slide_verts(rotated_mesh_y, slide_vert_percentage, mesh_name)
            if all_edges_has_two_faces:
                filename_to_save = os.path.join(graph_dir,
                                                mesh_files_dict['mesh'].split('/')[-1].split('.')[0] + '_y_rotation_' +
                                                str(y_theta_deg) + '_slided_nef_graph.bin')
                graph_constructor = GraphConstructor.GraphConstructor(input_mesh=slided_mesh, mesh_name=mesh_name,
                                                                      data_edges=mesh_edges,
                                                                      edge_area=edge_areas,
                                                                      edges_seg_labels=edges_seg_labels,
                                                                      edges_soft_seg_labels=edges_soft_seg_labels,
                                                                      faces_seg_labels=faces_seg_labels,
                                                                      nodes_seg_labels=nodes_seg_labels,
                                                                      filename_to_save=filename_to_save,
                                                                      graph_label=None)
                hybrid_graph = graph_constructor.create_graphs()

        if jitter_rotation:
            rotated_mesh_y_jittered = scale_verts(rotated_mesh_y, mean, var, coef)
            filename_to_save = os.path.join(graph_dir,
                                            mesh_files_dict['mesh'].split('/')[-1].split('.')[0] + '_y_rotation_' +
                                            str(y_theta_deg) + '_jittered_nef_graph.bin')
            graph_constructor = GraphConstructor.GraphConstructor(input_mesh=rotated_mesh_y_jittered,
                                                                  mesh_name=mesh_name,
                                                                  data_edges=mesh_edges,
                                                                  edge_area=edge_areas,
                                                                  edges_seg_labels=edges_seg_labels,
                                                                  edges_soft_seg_labels=edges_soft_seg_labels,
                                                                  faces_seg_labels=faces_seg_labels,
                                                                  nodes_seg_labels=nodes_seg_labels,
                                                                  filename_to_save=filename_to_save,
                                                                  graph_label=None)
            hybrid_graph = graph_constructor.create_graphs()

        if rotate_x_90_for_each_y:
            theta_x = np.pi / 2
            rotated_mesh_y_x_90 = rotateX3D(rotated_mesh_y, theta_x)
            x_theta_deg = int(np.rad2deg(theta_x))
            x_filename_to_save = os.path.join(graph_dir, mesh_files_dict['mesh'].split('/')[-1].split('.')[0] +
                                              '_y_rotation_' + str(y_theta_deg) + '_x_rotation_' + str(x_theta_deg) +
                                              '_nef_graph.bin')

            graph_constructor = GraphConstructor.GraphConstructor(input_mesh=rotated_mesh_y_x_90, mesh_name=mesh_name,
                                                                  data_edges=mesh_edges,
                                                                  edge_area=edge_areas,
                                                                  edges_seg_labels=edges_seg_labels,
                                                                  edges_soft_seg_labels=edges_soft_seg_labels,
                                                                  faces_seg_labels=faces_seg_labels,
                                                                  nodes_seg_labels=nodes_seg_labels,
                                                                  filename_to_save=x_filename_to_save,
                                                                  graph_label=None)
            hybrid_graph = graph_constructor.create_graphs()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    converter = Mesh2HybridParser().parse()

    segmentation_dir = os.path.join(converter.dataset, 'seg')
    soft_segmentation_dir = os.path.join(converter.dataset, 'sseg')
    face_segmentation_dir = os.path.join(converter.dataset, 'face_seg')
    node_segmentation_dir = os.path.join(converter.dataset, 'node_seg')
    classes_file = os.path.join(converter.dataset, 'classes.txt')
    if converter.portion == 'train':
        train_npz_dir = os.path.join(converter.dataset, 'train', 'cache')
        train_dir = os.path.join(converter.dataset, 'train')
        train_graph_dir = os.path.join(converter.hybrid_graphs, 'train')
        mkdir(train_graph_dir)
    elif converter.portion == 'test':
        test_npz_dir = os.path.join(converter.dataset, 'test', 'cache')
        test_dir = os.path.join(converter.dataset, 'test')
        test_graph_dir = os.path.join(converter.hybrid_graphs, 'test')
        mkdir(test_graph_dir)

    meshes = []
    classes = np.loadtxt(classes_file)

    if converter.portion == 'train':
        for root, _, file_names in os.walk(train_dir):
            for fname in file_names:
                if is_mesh_file(fname):
                    mesh_file = os.path.join(root, fname)
                    npz_file = os.path.join(train_npz_dir, fname.split('.')[0] + '_000.npz')
                    seg_file = os.path.join(segmentation_dir, fname.split('.')[0] + '.eseg')
                    soft_seg_file = os.path.join(soft_segmentation_dir, fname.split('.')[0] + '.seseg')
                    face_seg_file = os.path.join(face_segmentation_dir, fname.split('.')[0] + '.eseg')
                    node_seg_file = os.path.join(node_segmentation_dir, fname.split('.')[0] + '.eseg')
                    mesh_files_dict = {
                        'mesh': mesh_file,
                        'npz': npz_file,
                        'seg': seg_file,
                        'soft_seg': soft_seg_file,
                        'face_seg': face_seg_file,
                        'node_seg': node_seg_file
                    }
                    # The value for mean and var in the following is selected based on each dataset.
                    generate_augmented_graph_from_single_mesh(mesh_files_dict, train_graph_dir,
                                                              prevent_nonmanifold_edges=True,
                                                              num_of_slide_aug_for_each_sample=1,
                                                              num_of_jitter_aug_for_each_sample=1,
                                                              jitter_rotation=True, mean=0, var=0.002,
                                                              coef=1,
                                                              num_y_rotation=24, rotate_x_90_for_each_y=True,
                                                              slide_vert_percentage=0.2)

    if converter.portion == 'test':
        for root, _, file_names in os.walk(test_dir):
            for fname in file_names:
                if is_mesh_file(fname):
                    mesh_file = os.path.join(root, fname)
                    npz_file = os.path.join(test_npz_dir, fname.split('.')[0] + '_000.npz')
                    seg_file = os.path.join(segmentation_dir, fname.split('.')[0] + '.eseg')
                    soft_seg_file = os.path.join(soft_segmentation_dir, fname.split('.')[0] + '.seseg')
                    face_seg_file = os.path.join(face_segmentation_dir, fname.split('.')[0] + '.eseg')
                    node_seg_file = os.path.join(node_segmentation_dir, fname.split('.')[0] + '.eseg')
                    mesh_files_dict = {
                        'mesh': mesh_file,
                        'npz': npz_file,
                        'seg': seg_file,
                        'soft_seg': soft_seg_file,
                        'face_seg': face_seg_file,
                        'node_seg': node_seg_file
                    }

                    generate_augmented_graph_from_single_mesh(mesh_files_dict, test_graph_dir,
                                                              prevent_nonmanifold_edges=True)

from __future__ import print_function
import os

MESH_EXTENSIONS = [
    '.obj',
]

GRAPH_EXTENSIONS = [
    '.bin',
]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_mesh_file(filename):
    return any(filename.endswith(extension) for extension in MESH_EXTENSIONS)


def is_graph_file(filename):
    return any(filename.endswith(extension) for extension in GRAPH_EXTENSIONS)

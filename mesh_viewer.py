import vtk
import dgl
import warnings
import torch
import argparse
import os
from util.util import mkdir

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='view segmented mesh with predicted or ground truth edge labels.')
    parser.add_argument('--input_graph_name', type=str, help='Enter the input graph name')
    parser.add_argument('--input_folder', type=str, help='Enter the input graph folder path')
    parser.add_argument('--mode', choices={"prediction", "gt"}, default="prediction")
    parser.add_argument('--screenshot_folder_to_save', type=str, help='Path for saving screenshot')

    args = parser.parse_args()
    mkdir(args.screenshot_folder_to_save)
    graph_path = os.path.join(args.input_folder, args.input_graph_name)
    # Load input graph and edges
    graph, label = dgl.load_graphs(graph_path)
    graph = graph[0]
    subgraph = graph.edge_type_subgraph(['influences (nn)'])
    vertices = subgraph.nodes['node'].data['init_geometric_feat'][:, :3]
    src_node, dst_node = subgraph.edges()
    src_node = torch.reshape(src_node, (len(src_node), 1))
    dst_node = torch.reshape(dst_node, (len(dst_node), 1))
    directed_edges = torch.hstack((src_node, dst_node))
    edges = directed_edges[0: int(len(directed_edges) / 2)]

    if args.mode == "gt":
        edges_seg_labels = graph.nodes['edge'].data['label']
        edges_seg_labels = edges_seg_labels.type(torch.int)
        edges_seg_labels = edges_seg_labels.numpy()
    else:
        edges_seg_labels = graph.nodes['edge'].data['prediction_class']
        edges_seg_labels = edges_seg_labels.type(torch.int)
        edges_seg_labels = edges_seg_labels.numpy()

    pts = vtk.vtkPoints()
    for i in range(vertices.shape[0]):
        pts.InsertNextPoint(list(vertices[i]))

    red = [255, 0, 0]
    green = [0, 255, 0]
    blue = [0, 0, 255]
    yellow = [255, 255, 0]
    purple = [170, 93, 229]
    cyan = [0, 255, 255]
    magenta = [255, 0, 255]
    orange = [255, 140, 0]
    color_dict = {0: red, 1: green, 2: blue, 3: yellow, 4: purple, 5: cyan, 6: magenta, 7: orange}
    # Setup the colors array
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    # Add the colors we created to the colors array
    for i in range(edges_seg_labels.shape[0]):
        colors.InsertNextTypedTuple(color_dict[edges_seg_labels[i]])

    line_list = []
    for i in range(edges.shape[0]):
        n0 = edges[i][0]
        n1 = edges[i][1]
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, n0)
        line.GetPointIds().SetId(1, n1)
        line_list.append(line)

    # Create a cell array to store the lines in and add the lines to it
    lines = vtk.vtkCellArray()
    for j in range(len(line_list)):
        lines.InsertNextCell(line_list[j])

    # Create a polydata to store everything in
    linesPolyData = vtk.vtkPolyData()

    # Add the points to the dataset
    linesPolyData.SetPoints(pts)

    # Add the lines to the dataset
    linesPolyData.SetLines(lines)
    # Color the lines - associate the first component (red) of the
    # colors array with the first component of the cell array (line 0)
    # and the second component (green) of the colors array with the
    # second component of the cell array (line 1)
    linesPolyData.GetCellData().SetScalars(colors)

    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(linesPolyData)
    else:
        mapper.SetInputData(linesPolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(400, 400)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderer.AddActor(actor)
    renderer.SetBackground([255, 255, 255])
    renderWindow.Render()
    renderWindowInteractor.Start()

    screenshot_name = args.input_graph_name.split('.')[0]
    if args.mode == "gt":
        screenshot_name = screenshot_name + '_gt.png'
    else:
        screenshot_name = screenshot_name + '_prediction.png'
    path_to_save = os.path.join(args.screenshot_folder_to_save, screenshot_name)

    # save screenshot after you interact with mesh viewer window and close it. The following codes save the last
    # pose before you close the viewer window
    renderWindowInteractor.Start()
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renderWindow)
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(path_to_save)
    writer.SetInputConnection(w2if.GetOutputPort())
    writer.Write()

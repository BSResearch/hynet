import argparse
import os


class Mesh2HybridParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument('--dataset', required=True, help='path to mesh data (should have subfolder, train,'
                                                                  'test, seg, sseg, and classes.txt'
                                                                  'in case segmentation is done on node and face '
                                                                  'node_seg or face seg are required')
        self.parser.add_argument('--portion', choices={"train", "test"}, required=True,
                                 help='which portion of the data do you want to convert to hybrid graph? train or test?')
        self.parser.add_argument('--hybrid_graphs', required=True, help='path to a folder to save the hybrid graph')

    def parse(self):
        self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        return self.opt

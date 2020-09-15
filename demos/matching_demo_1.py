import os
from erd_scanner.demo_utils import draw_graph
from erd_scanner.img_to_graph import img_to_graph
import cv2
import networkx as nx
import re

def attrs_match(v_attrs, pv_attrs):
    shape_match = v_attrs['shape'] == pv_attrs['shape']
    double_lined_match = v_attrs['double_lined'] == pv_attrs['double_lined']
    name_match = len(re.findall(pattern=pv_attrs['name_pattern'], string=v_attrs['name'])) > 0

    return shape_match and double_lined_match and name_match


def find_subgraph_match(G, G_pattern):
    G_nodes = list(G.nodes(data=True))
    G_pattern_nodes = list(G_pattern.nodes(data=True))

    match_found = True
    mapping = {}
    for pv in G_pattern_nodes:
        pv_ind, pv_attrs = pv
        for v in G_nodes:
            v_ind, v_attrs = v
            if attrs_match(v_attrs, pv_attrs):
                mapping[pv_ind] = v_ind
                break

    if len(mapping.keys()) != len(G_pattern_nodes):
        match_found = False

    if match_found:
        for pe in G_pattern.edges(data=True):
            pu, pv, pattrs = pe
            if not G.has_edge(mapping[pu], mapping[pv]):
                match_found = False
                break

    if match_found:
        return G.subgraph(nodes=[v for _,v in mapping.items()])
    return None


if __name__ == "__main__":
    orig_img = cv2.imread(os.path.join("../imgs", "erd_sample_5.png"), cv2.IMREAD_GRAYSCALE)
    G = img_to_graph(img=orig_img)
    draw_graph(G)


    G_pattern_1 = nx.Graph()
    G_pattern_1.add_node(0, shape='s', double_lined=False, name_pattern=re.compile(r".*"))
    G_pattern_1.add_node(1, shape='o', double_lined=False, name_pattern=re.compile(r"FirstName"))
    G_pattern_1.add_node(2, shape='o', double_lined=False, name_pattern=re.compile(r"LastName"))
    G_pattern_1.add_node(3, shape='o', double_lined=False, name_pattern=re.compile(r"Sports"))
    G_pattern_1.add_edge(0, 1)
    G_pattern_1.add_edge(0, 2)

    # draw_graph(G_pattern_1)

    G_match_1 = find_subgraph_match(G=G, G_pattern=G_pattern_1)
    draw_graph(G_match_1)
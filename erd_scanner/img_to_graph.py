import cv2
import networkx as nx
from erd_scanner.utils import *


def img_to_graph(img, demo=False):
    _, binary_orig_img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)

    if demo:
        cv2.imshow("binary_orig_img", binary_orig_img)
        cv2.waitKey()

    # find contours
    node_contours, node_shapes, node_double_lined, node_names = get_node_contours_and_shapes(binary_img=binary_orig_img,
                                                                                             orig_img=img)

    node_mask, edge_mask = get_node_and_edge_masks(binary_img=binary_orig_img, node_contours=node_contours)
    numpy_horizontal_concat = np.concatenate((node_mask, edge_mask), axis=1)
    cv2.imshow('numpy_horizontal_concat', numpy_horizontal_concat)
    G = get_graph_from_masks(edge_mask=edge_mask, node_contours=node_contours, node_shapes=node_shapes,
                             node_double_lined=node_double_lined, node_labels=node_names)

    for v in G.nodes(data=True):
        v[1]['name'] = v[1]['name'].strip()
    return G
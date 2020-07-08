import cv2
import os
from demo_utils import *
from utils import *

if __name__ == "__main__":
    # DEMO 1
    # orig_img = cv2.imread(os.path.join("./imgs", "erd_sample_3.png"), cv2.IMREAD_GRAYSCALE)
    # _, binary_orig_img = cv2.threshold(orig_img, 160, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("binary_orig_img", binary_orig_img)
    # cv2.waitKey()
    # # find contours
    #
    # node_contours, node_shapes, node_double_lined = get_node_contours_and_shapes(binary_img=binary_orig_img)
    #
    # node_mask, edge_mask = get_node_and_edge_masks(binary_img=binary_orig_img, node_contours=node_contours)
    # numpy_horizontal_concat = np.concatenate((node_mask, edge_mask), axis=1)
    # cv2.imshow('numpy_horizontal_concat', numpy_horizontal_concat)
    # G = get_graph_from_masks(edge_mask=edge_mask, node_contours=node_contours, node_shapes=node_shapes, node_double_lined=node_double_lined)
    #
    # draw_graph(G)

    orig_img = cv2.imread(os.path.join("./imgs", "erd_sample_5.png"), cv2.IMREAD_GRAYSCALE)
    _, binary_orig_img = cv2.threshold(orig_img, 160, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("binary_orig_img", binary_orig_img)
    cv2.waitKey()
    #BW image
    bwi = cv2.bitwise_not(orig_img)
    # find contours

    node_contours, node_shapes, node_double_lined,node_ocr = get_node_contours_and_shapes(binary_img=binary_orig_img)

    node_mask, edge_mask = get_node_and_edge_masks(binary_img=binary_orig_img, node_contours=node_contours)
    numpy_horizontal_concat = np.concatenate((node_mask, edge_mask), axis=1)
    cv2.imshow('numpy_horizontal_concat', numpy_horizontal_concat)
    G = get_graph_from_masks(edge_mask=edge_mask, node_contours=node_contours, node_shapes=node_shapes,
                             node_double_lined=node_double_lined,node_OCR=node_ocr)

    draw_graph(G)
import os
# from erd_scanner.demo_utils import draw_graph
from erd_scanner.img_to_graph import img_to_graph
from ERD.graph_to_ERD import graph_to_ERD
from ERD.erd_objects import *
import cv2

if __name__ == "__main__":
    orig_img = cv2.imread(os.path.join("../imgs", "erd_sample_3.png"), cv2.IMREAD_GRAYSCALE)
    orig_img2 = cv2.imread(os.path.join("../imgs", "erd_sample_11.png"), cv2.IMREAD_GRAYSCALE)
    orig_img3 = cv2.imread(os.path.join("../imgs", "erd_sample_12.png"), cv2.IMREAD_GRAYSCALE)

    G = img_to_graph(img=orig_img, display_graphs=False)
    G_2 = img_to_graph(img=orig_img, display_graphs=False)
    G_3 = img_to_graph(img=orig_img2, display_graphs=False)
    G_4 = img_to_graph(img=orig_img3, display_graphs=False)

    ERD_1 = graph_to_ERD(G)
    ERD_2 = graph_to_ERD(G_2)
    ERD_3 = graph_to_ERD(G_3)
    ERD_4 = graph_to_ERD(G_4)

    print(f'Similarity Between Identical ERDs: {ERD_1.compare_ERD(ERD_2):.2f}%')
    print(f'Similarity Between ERDs with One Missing Attributes: {ERD_1.compare_ERD(ERD_3):.2f}%')
    print(f'Similarity Between ERDs with Three Missing Attributes: {ERD_1.compare_ERD(ERD_4):.2f}%')
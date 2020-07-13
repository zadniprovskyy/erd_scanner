import os
from erd_scanner.demo_utils import draw_graph
from erd_scanner.img_to_graph import img_to_graph
from erd_graders.sandy_grader import compare_similarity
import cv2

if __name__ == "__main__":
    orig_img = cv2.imread(os.path.join("../imgs", "erd_sample_5.png"), cv2.IMREAD_GRAYSCALE)
    G1 = img_to_graph(img=orig_img)
    draw_graph(G1)

    orig_img = cv2.imread(os.path.join("../imgs", "erd_sample_5.png"), cv2.IMREAD_GRAYSCALE)
    G2 = img_to_graph(img=orig_img)

    draw_graph(G2)

    score = compare_similarity(G1, G2)
    print("Similarity score:", score)
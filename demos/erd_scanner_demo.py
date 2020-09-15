import os
from erd_scanner.demo_utils import draw_graph
from erd_scanner.img_to_graph import img_to_graph
import cv2

if __name__ == "__main__":
    orig_img = cv2.imread(os.path.join("../imgs", "erd_sample_5.png"), cv2.IMREAD_GRAYSCALE)
    G = img_to_graph(img=orig_img)
    draw_graph(G)

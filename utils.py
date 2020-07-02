import cv2
import numpy as np
import scipy
import pytesseract as pytt
from sklearn.neighbors import KNeighborsClassifier
import os
import networkx as nx
import matplotlib.pyplot as plt
from demo_utils import draw_graph
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
 from PIL import Image
except ImportError:
 import Image

def get_contour_shape(cnt):
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if cv2.contourArea(cnt) < 0.1:
        return "line"

    elif len(approx) == 3:
        return "triangle"

    elif len(approx) == 4:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area) / rect_area
        if extent < 0.85:
            return "rhombus"
        else:
            return "rectangle"

    elif 5 < len(approx):
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        if solidity >= 0.95:
            return "ellipse"

    else:
        return "unknown"


def get_node_and_edge_masks(binary_img, node_contours):
    node_mask = np.zeros_like(binary_img)
    cv2.drawContours(node_mask, node_contours, -1, color=255, thickness=cv2.FILLED)
    kernel = np.ones((5, 5), np.uint8)
    dilated_node_mask = cv2.dilate(node_mask, kernel, iterations=2)

    filled_nodes_img = np.bitwise_or(node_mask, binary_img)
    edge_mask = cv2.bitwise_and(binary_img, cv2.bitwise_not(dilated_node_mask))

    return dilated_node_mask, edge_mask


def get_graph_from_masks(edge_mask, node_contours, node_shapes, node_double_lined):
    edge_mask_contours, _ = cv2.findContours(edge_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # nbrs will serve as a classifier to determine which node edgepoint corresponds to
    node_points = np.concatenate(node_contours, axis=0).squeeze(axis=1)
    node_inds = []
    for i in range(len(node_contours)):
        node_inds.extend([i] * node_contours[i].shape[0])

    nbrs = KNeighborsClassifier(n_neighbors=2).fit(node_points, node_inds)
    #################
    edge_points = np.concatenate(edge_mask_contours, axis=0).squeeze(axis=1)
    db = DBSCAN(eps=0.2, min_samples=30).fit(StandardScaler().fit_transform(edge_points))
    ############
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = edge_points[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = edge_points[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    ###############
    line_contours = []
    line_endpoints = []
    edges = []

    G = nx.Graph()

    for i in range(len(node_shapes)):
        shape = node_shapes[i]
        if shape == "triangle":
            shape_symbol='^'
        elif shape == "ellipse":
            shape_symbol = 'o'
        elif shape == "rectangle":
            shape_symbol = 's'
        elif shape == "rhombus":
            shape_symbol = 'D'
        else:
            shape_symbol = '*'

        G.add_node(i, shape=shape_symbol, double_lined=node_double_lined[i])

    for i in range(n_clusters_):
        rect = cv2.minAreaRect(edge_points[db.labels_ == i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # plug into the line formula to find the two endpoints, p0 and p1
        # to plot, we need pixel locations so convert to int
        p0 = tuple(box[0])
        if np.linalg.norm(box[0] - box[1]) < np.linalg.norm(box[0] - box[3]):
            p1 = tuple(box[3])
        else:
            p1 = tuple(box[1])

        line_endpoints.append((p0, p1))
        # print(edge_points[db.labels_ == i].shape[0] / np.linalg.norm(np.array(p0) - np.array(p1)) )
        # cv2.line(demo_edge_endpoints_img, p0, p1,  (0, 255, 0), 2)
        v1, v2 = nbrs.predict([p0,p1])
        G.add_edge(v1, v2, density=edge_points[db.labels_ == i].shape[0] / np.linalg.norm(np.array(p0) - np.array(p1)))

    # G.add_edges_from(edges)
    return G


def get_node_contours_and_shapes(binary_img):
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    new_contours = []

    for i in range(len(hierarchy[0])):
        h_i = hierarchy[0][i]
        ct_i = contours[i]
        next_c, prev_c, child_c, parent_c = h_i
        ct_shape = get_contour_shape(ct_i)

        if ct_shape in ['rectangle', 'ellipse', 'rhombus', 'triangle']:
            if child_c == -1:
                double_lined = False
            else:
                child_shape = get_contour_shape(contours[child_c])
                double_lined =  cv2.contourArea(contours[child_c]) >= cv2.contourArea(ct_i) * 0.9 and child_shape == ct_shape
                print(double_lined, child_shape, ct_shape)
            if parent_c == -1:
                new_contours.append((ct_i, ct_shape, double_lined))
            else:
                parent_shape = get_contour_shape(contours[parent_c])
                if parent_shape not in ['rectangle', 'ellipse', 'rhombus', 'triangle']:
                    grandpa_c = hierarchy[0][parent_c][3]
                    if grandpa_c == -1 or get_contour_shape(contours[grandpa_c]) not in ['rectangle', 'ellipse',
                                                                                         'rhombus',
                                                                                         'triangle']:
                        new_contours.append((ct_i, ct_shape, double_lined))

    new_contours = sorted(new_contours, key=lambda cnt: -cv2.contourArea(cnt[0]))

    # cnts_perim = np.array([cv2.arcLength(cnt, True) for cnt, _ in new_contours])
    node_contours_and_shapes = list(filter(lambda cnt: cv2.contourArea(cnt[0])> 0.25*cv2.contourArea(new_contours[0][0]), new_contours))

    node_contours = [x[0] for x in node_contours_and_shapes]
    node_shapes = [x[1] for x in node_contours_and_shapes]
    node_double_lined = [x[2] for x in node_contours_and_shapes]

    return node_contours, node_shapes, node_double_lined


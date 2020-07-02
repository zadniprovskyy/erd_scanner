import cv2
import numpy as np
import scipy
import pytesseract as pytt
from sklearn.neighbors import KNeighborsClassifier
import os
import networkx as nx
import matplotlib.pyplot as plt
from demo_utils import draw_graph

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


def get_graph_from_masks(edge_mask, node_contours, node_shapes):
    edge_mask_contours, _ = cv2.findContours(edge_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # demo_edge_endpoints_img = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2RGB).copy()
    edge_mask_contours, _ = cv2.findContours(edge_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # nbrs will serve as a classifier to determine which node edgepoint corresponds to
    X = np.concatenate(node_contours, axis=0).squeeze(axis=1)
    y = []
    for i in range(len(node_contours)):
        y.extend([i] * node_contours[i].shape[0])

    nbrs = KNeighborsClassifier(n_neighbors=2).fit(X, y)

    line_contours = []
    line_endpoints = []
    edges = []

    G = nx.Graph()
    for i in range(len(node_shapes)):
        shape = node_shapes[i]
        if shape == "triangle":
            G.add_node(i, shape='^')
        elif shape == "ellipse":
            G.add_node(i, shape="o")
        elif shape == "rectangle":
            G.add_node(i, shape="s")
        elif shape == "rhombus":
            G.add_node(i, shape="D")
        else:
            G.add_node(i, shape="*")
    for contour in edge_mask_contours:
        fit_line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.1, 0.1)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # plug into the line formula to find the two endpoints, p0 and p1
        # to plot, we need pixel locations so convert to int
        p0 = tuple(box[0])
        if np.linalg.norm(box[0] - box[1]) < 10:
            p1 = tuple(box[3])
        else:
            p1 = tuple(box[2])

        line_endpoints.append((p0, p1))
        # cv2.line(demo_edge_endpoints_img, p0, p1,  (0, 255, 0), 2)
        edges.append(nbrs.predict([p0,p1]))

    G.add_edges_from(edges)
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
            if parent_c == -1:
                new_contours.append((ct_i, ct_shape))
            else:
                parent_shape = get_contour_shape(contours[parent_c])
                if parent_shape not in ['rectangle', 'ellipse', 'rhombus', 'triangle']:
                    grandpa_c = hierarchy[0][parent_c][3]
                    if grandpa_c == -1 or get_contour_shape(contours[grandpa_c]) not in ['rectangle', 'ellipse',
                                                                                         'rhombus',
                                                                                         'triangle']:
                        new_contours.append((ct_i, ct_shape))
    # contours = sorted(contours, key=lambda cnt: -cv2.contourArea(cnt))

    cnts_perim = np.array([cv2.arcLength(cnt, True) for cnt, _ in new_contours])
    node_contours_and_shapes = list(filter(lambda cnt: cv2.arcLength(cnt[0], True) > 25, new_contours))

    node_contours = [x[0] for x in node_contours_and_shapes]
    node_shapes = [x[1] for x in node_contours_and_shapes]
    return node_contours, node_shapes

def line_distinguisher(cnt,solid,dotted):
    x,y,w,h = cv.boundingRect(cnt) #Find the bound for the contour in rectangle
    for line in solid:
        x1,y1,x2,y2 = line #Get coordinates of the line
        if (x-1<=x1<=x+w+1 or x-1<=x2<=x+w+1) and (y<=y1<=y+1 or y<=y2<=y+1):
            #the line is associated with the contour
            return "solid"
    for line in dotted:
        x1,y1,x2,y2 = line #Get coordinates of the line
        if (x-1<=x1<=x+w+1 or x-1<=x2<=x+w+1) and (y<=y1<=y+1 or y<=y2<=y+1):
            #the line is associated with the contour
            return "dotted"
    return "None"
    
def line_style(img):
    solid_line = [] # Stores solid line points
    dotted_line = [] #Stores dotline points
    _, threshold_img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    node_contours, node_shapes=get_node_contours_and_shapes(img)
    node_mask = np.zeros_like(threshold_img)
    cv2.drawContours(node_mask, node_contours,-1,color=255,thickness=cv2.FILLED)
    kernel = np.ones((5,5),np.uint8)
    dilated_node_mask = cv2.dilate(node_mask,kernel,iterations = 2)
    edge_mask = cv2.bitwise_and(threshold_img, cv2.bitwise_not(dilated_node_mask))
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 80  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_only = np.copy(edge_mask)*0 # image without length put on.
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edge_mask, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    for i in range(len(lines)):
        line = lines[i]
        x1,y1,x2,y2 = line[0]
        cv2.line(line_only,(x1,y1),(x2,y2),(255,0,0),2)
        solid_line.append(line[0])
    dotline_img = np.copy(img)*0 #black image with the same resolution
    dotlines =edge_mask-line_only
    dotlines_coor = cv2.HoughLinesP(dotlines, rho, theta, threshold, np.array([]),
                        30, 10)
    for i in range(len(dotlines_coor)):
        line = lines[i]
        x1,y1,x2,y2 = line[0]
        dotted_line.append(line[0])


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
import spacy
import textdistance
try:
 from PIL import Image
except ImportError:
 import Image
from sys import platform
if platform == "win32":
    pytt.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

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


def get_graph_from_masks(edge_mask, node_contours, node_shapes, node_double_lined,node_OCR):
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

        G.add_node(i, shape=shape_symbol, double_lined=node_double_lined[i],ocr = node_OCR[i])
        
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
                x,y,w,h = cv2.boundingRect(ct_i)
                node_img = binary_img[y:y+h,x:x+w].copy()
                node_img = cv2.resize(node_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                contour = side_cut(node_img,ct_shape,False)
                words = get_contour_letters(contour,ct_shape)
                print(double_lined, child_shape, ct_shape,words)
            if parent_c == -1:
                x,y,w,h = cv2.boundingRect(ct_i)
                node_img = img[y:y+h,x:x+w].copy()
                contour = side_cut(node_img,ct_shape,False)
                node_img = cv2.resize(node_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                words = get_contour_letters(contour,ct_shape)
                new_contours.append((ct_i, ct_shape, double_lined,words))
            else:
                parent_shape = get_contour_shape(contours[parent_c])
                if parent_shape not in ['rectangle', 'ellipse', 'rhombus', 'triangle']:
                    grandpa_c = hierarchy[0][parent_c][3]
                    if grandpa_c == -1 or get_contour_shape(contours[grandpa_c]) not in ['rectangle', 'ellipse',
                                                                                         'rhombus',
                                                                                         'triangle']:
                        new_contours.append((ct_i, ct_shape, double_lined,words))

    new_contours = sorted(new_contours, key=lambda cnt: -cv2.contourArea(cnt[0]))

    # cnts_perim = np.array([cv2.arcLength(cnt, True) for cnt, _ in new_contours])
    node_contours_and_shapes = list(filter(lambda cnt: cv2.contourArea(cnt[0])> 0.25*cv2.contourArea(new_contours[0][0]), new_contours))

    node_contours = [x[0] for x in node_contours_and_shapes]
    node_shapes = [x[1] for x in node_contours_and_shapes]
    node_double_lined = [x[2] for x in node_contours_and_shapes]
    node_OCR = [x[3] for x in node_contours_and_shapes]    
    return node_contours, node_shapes, node_double_lined,node_OCR


def side_cut(img,shape,demo):
  #Using pillow to open the image
  im = Image.fromarray(img)
  im_convert = im.convert('L').copy()
  #Convert it into opencv format
  open_cv_image = np.array(im_convert)
  #Shape Detection
  #IF statement for shapes
  if shape == "triangle":
    print("triangle detected")
    width, height = im_convert.size
    cropped_width = round(width * 0.40243902)
    cropped_height = round(height *0.20967742)
    cropped_bottom = round(height * 0.03335606)
    cropped = im_convert.crop((cropped_width,cropped_height,width-cropped_width,height-cropped_bottom))
    cropped = np.array(cropped)
    if(demo):
      cv2_imshow(cropped)
    return cropped
  elif shape == "rhombus":
    print("diamond detected")
    #Rotate the img for 90 convert it into rectangular shape
    rotate = im_convert.rotate(45,fillcolor=None)
    width, height = rotate.size
    coropped_size = round(width *0.1875)
    cropped = rotate.crop((coropped_size,coropped_size,width-coropped_size,height-coropped_size))
    #Rotate it back
    rotated = cropped.rotate(315)
    rotated = np.array(rotated)
    if(demo):
      cv2_imshow(rotated)
    return rotated
  elif shape == "ellipse":
    print("ellipse detected")
    #crop less side but more up and down
    width, height = im_convert.size
    cropped_width = round(width * 0.14772727)
    cropped_height = round(height * 0.2545454545)
    cropped_top = round(width * 0.125)
    cropped_bottom = round(height * 0.2)
    cropped = im_convert.crop((cropped_width,cropped_height,width-cropped_top,height-cropped_bottom))
    #converting the image from pillow format to opencv format
    cropped = np.array(cropped)
    if(demo):
      cv2_imshow(cropped)
    return cropped
  elif shape == "rectangle":
    print("rectangle detected")
    #crop equivalent size for rectangle
    width, height = im_convert.size
    cropped_width = round(width * 0.01639344)
    cropped_height = round(height * 0.03225806)
    cropped = im_convert.crop((cropped_width,cropped_height,width-cropped_width,height-cropped_height))
    cropped = np.array(cropped)
    #converting the image from pillow format to opencv format
    if(demo):
      cv2_imshow(cropped)
    return cropped
  else:
    return False


def get_contour_letters(cnt,shape):
    # # Remove horizontal
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    # detected_lines = cv2.morphologyEx(cnt, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(cnt, [c], -1, (255,255,255), 2)
    text = pytt.image_to_string(cnt,config="-l eng --psm 11")
    result = " ".join(text.split('\n'))
    # for i in range(1,len(result)):
    #     if (result[i].isupper() and result[i-1]!=" "):
    #         result = result[:i]+" "+result[i:] # Give a space for each word.
    return result

def compare_similarity(g,g_sol):
    nlp = spacy.load("en_core_web_lg")
    total_entity_sol = 0
    total_entity_submission = 0
    total = 0
    for i in range(g.number_of_nodes()):
        if g.nodes[i]['shape'] == "s":
            total_entity_submission +=1
            #node is an entity
            entity_name = g.nodes[i]['ocr']
            print("entity_name: "+entity_name)
            token = nlp(g.nodes[i]['ocr'])
            sol_entity_name = ""
            sol_entity_num = -1
            submission_num = -1
            for j in range(g_sol.number_of_nodes()):
                if g_sol.nodes[i]['shape'] == "s":
                    if(token.similarity(nlp(g_sol.nodes[i]['ocr']))>0.9 or textdistance.levenshtein.normalized_similarity(g_sol.nodes[j]['ocr'],entity_name)>0.85):
                        total_entity_sol += 1
                        #high similarity entity (similiar words or similar string)
                        print("Entity Name:"+g_sol.nodes[j]['ocr'])
                        sol_entity_name = g_sol.nodes[j]['ocr']
                        sol_entity_num = j
                        submission_num = i
                        break
            if (sol_entity_num!=-1):
                #found similar entity in solution graph
                sol_connected = list(nx.node_connected_component(g_sol, sol_entity_num))
                submission_connected = list(nx.node_connected_component(g, submission_num))
                solution_long = False
                mark = 0
                for i in submission_connected:
                    print("i: "+str(i))
                    if(g.nodes[i]['shape']=='o'):
                        #The node is an attribute
                        similarity = 0
                        token = nlp(g.nodes[i]['ocr'])
                        print("token: "+g.nodes[i]['ocr'])
                        for j in sol_connected:
                            print("j: "+str(j))
                            if(g_sol.nodes[j]['shape']=='o'):
                                print("distance: "+str(textdistance.levenshtein.normalized_similarity(g.nodes[i]['ocr'],g_sol.nodes[j]['ocr'])))
                                print("similarity: ",token.similarity(nlp(g_sol.nodes[j]['ocr'])))
                                ans_similarity = max(token.similarity(nlp(g_sol.nodes[j]['ocr'])),textdistance.levenshtein.normalized_similarity(g.nodes[i]['ocr'],g_sol.nodes[j]['ocr'])) # Get the max of the similarity mark
                                if(ans_similarity>similarity):
                                    similarity = ans_similarity
                        mark += similarity
                mark = mark / len(sol_connected) # entity mark
                total += mark
    return total
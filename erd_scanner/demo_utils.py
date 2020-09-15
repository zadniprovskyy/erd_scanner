import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(G):
    # Drawing the graph
    # First obtain the node positions using one of the layouts
    nodePos = nx.layout.spring_layout(G)

    # The rest of the code here attempts to automate the whole process by
    # first determining how many different node classes (according to
    # attribute 's') exist in the node set and then repeatedly calling
    # draw_networkx_node for each. Perhaps this part can be optimised further.

    # Get all distinct node classes according to the node shape attribute
    nodeShapes = set((aShape[1]["shape"] for aShape in G.nodes(data=True)))

    # For each node class...
    for aShape in nodeShapes:
        # ...filter and draw the subset of nodes with the same symbol in the positions that are now known through the use of the layout.
        nx.draw_networkx_nodes(G, nodePos, node_shape=aShape,
                            nodelist=[sNode[0] for sNode in filter(lambda x: x[1]["shape"] == aShape,
                            G.nodes(data=True))],
                            node_size=list(map(
                                lambda x: 600 if x[1]['double_lined'] else 200,
                                filter(lambda x: x[1]["shape"] == aShape, G.nodes(data=True))
                            ))
        )


    # Finally, draw the edges between the nodes
    nx.draw_networkx_edges(G, nodePos, edgelist=list(filter(lambda e: e[2]['density'] < 1, G.edges(data=True))), style='dashed')
    nx.draw_networkx_edges(G, nodePos, edgelist=list(filter(lambda e: e[2]['density'] >= 1, G.edges(data=True))),
                           style='solid')
    nx.draw_networkx_edges(G, nodePos, edgelist=list(filter(lambda e: e[2]['density'] >= 2.5, G.edges(data=True))),
                           style='solid', width=3)

    labels_dict = {}
    for i in range(len(G.nodes)):
        labels_dict[i] = G.nodes[i]['name']
    nx.draw_networkx_labels(G,pos=nodePos, labels=labels_dict)

    plt.plot()
    plt.show()
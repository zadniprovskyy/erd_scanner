from ERD.erd_objects import *
import networkx as nx
def graph_to_ERD(g):
    new_ERD = ERD()
    for i in range(g.number_of_nodes()):
        if g.nodes[i]['shape'] == "s":
            entity = Entity()
            entity.name = g.nodes[i]['name']
            if new_ERD.entity_in_ERD(entity):
                entity = new_ERD.get_entity_by_name(entity.name)
            else:
                new_ERD.add_entity(entity)
            entity.strong_entity = g.nodes[i]['double_lined']
            connections = list(nx.node_connected_component(g, i))
            for connected_node in connections:
                # Attribute
                if connected_node['shape'] == 'o':
                    new_attr = Attribute()
                    new_attr.name = connected_node['name']
                    new_attr.multi_valued = connected_node['double_lined']
                    new_attr.belongs_to_entity = entity
                    # TODO - Get Key Info for Entity Attributes
                    entity.add_attribute(new_attr)

                # Relationship
                if connected_node['shape'] in ['^', 'D']:
                    new_rel = Relationship()
                    new_rel.name = connected_node['name']
                    if new_ERD.relationship_in_ERD(new_rel):
                        entity.add_relationship(new_ERD.get_relationship_by_name(new_rel.name))
                        continue
                    else:
                        new_ERD.add_relationship(new_rel)
                        entity.add_relationship(new_rel)
                    new_rel.strong_relationship = connected_node['double_lined']
                    new_rel.entity_one = entity
                    # TODO - Get Cardinality
                    for neighbor in nx.neighbors(g, connected_node):
                        if neighbor['shape'] == 's':
                            second_entity = Entity()
                            second_entity.name = neighbor['name']
                            if not new_ERD.entity_in_ERD(second_entity):
                                new_rel.entity_two = second_entity
                                new_ERD.add_entity(second_entity)
                            else:
                                new_rel.entity_two = new_ERD.get_entity_by_name(second_entity.name)
                        if neighbor['shape'] == 'o':
                            # Relationship Attribute
                            new_rel_attr = Attribute()
                            new_rel_attr.name = neighbor['name']
                            new_rel_attr.multi_valued = neighbor['double_lined']
                            # TODO - Get Key Info for Relationship Attributes
                            new_rel.add_attribute(new_rel_attr)
            return new_ERD




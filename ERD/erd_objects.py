from textdistance import levenshtein as lev
from itertools import product
from config import entity_weightage, ERD_weightage, relationship_weightage
import numpy as np
class Entity:
    def __init__(self):
        self.primary_key = None
        self.name = ""
        self.number_of_attributes = 0
        self.strong_entity = True
        self.keys = []
        self.attributes = []
        self.relationships = []

    def add_attribute(self, attr):
        self.attributes.append(attr)
        self.number_of_attributes+=1
        attr.belongs_to_entity = self

    def remove_attribute(self, attr):
        self.attributes.remove(attr)
        if self.number_of_attributes>0:
            self.number_of_attributes -= 1
        attr.belongs_to_entity = None

    def add_relationship(self, rel):
        self.relationships.append(rel)

    def remove_relationship(self, rel):
        self.relationships.remove(rel)

    def compare_entities(self, other_entity):
        similarity=0
        num_correct_attributes = 0
        for attr1, attr2 in product(self.attributes, other_entity.attributes):
            if attr1.compare_attributes(attr2):
                num_correct_attributes+=1
        attribute_score = num_correct_attributes/self.number_of_attributes if self.number_of_attributes>0 else 0
        similarity += attribute_score*entity_weightage['Attributes']
        if self.strong_entity == other_entity.strong_entity:
            similarity += entity_weightage['Type']
        similarity += entity_weightage['Name']*lev.normalized_similarity(self.name, other_entity.name)
        # TODO - Compare Keys
        similarity += entity_weightage['Keys']
        # print('Entity Similarity: ', similarity)
        return similarity

    def __eq__(self, other):
        if lev.normalized_similarity(other.name, self.name)>.9:
            # TODO - Compare Keys
            if self.strong_entity == other.strong_entity and self.number_of_attributes==other.number_of_attributes:
                return True
        return False

class Attribute:
    def __init__(self):
        self.name = ""
        self.key_attribute = False
        self.strong_attribute = True
        self.multi_valued = False
        self.belongs_to_entity = None
        self.belongs_to_relationship = False

    def compare_attributes(self, other_attr):
        if lev.normalized_similarity(other_attr.name, self.name)>.9 and self==other_attr:
            return True
        return False

    def __eq__(self, other):
        if self.key_attribute == other.key_attribute and self.strong_attribute==other.strong_attribute:
            if self.multi_valued == other.multi_valued and self.belongs_to_relationship==other.belongs_to_relationship:
                return True
        return False

class Relationship:
    def __init__(self):
        self.name = ""
        self.strong_relationship = True
        self.entity_one = None
        self.entity_two = None
        self.cardinality = ""
        self.attributes = []
        self.number_of_attributes = 0

    def add_attribute(self, attr):
        self.attributes.append(attr)
        self.number_of_attributes+=1
        attr.belongs_to_entity = self
        attr.belongs_to_relationship = True

    def remove_attribute(self, attr):
        self.attributes.remove(attr)
        if self.number_of_attributes>0:
            self.number_of_attributes -= 1
        attr.belongs_to_entity = None
        attr.belongs_to_relationship = False

    def compare_relationships(self, other_rel):
        similarity = 0
        num_correct_attributes = 0
        for attr1, attr2 in product(self.attributes, other_rel.attributes):
            if attr1.compare_attributes(attr2):
                num_correct_attributes += 1
        attribute_score = num_correct_attributes / self.number_of_attributes if self.number_of_attributes > 0 else 0
        similarity += attribute_score * relationship_weightage['Attributes'] if self.number_of_attributes>0 else relationship_weightage['Attributes']
        if self.strong_relationship==other_rel.strong_relationship:
            similarity += relationship_weightage['Type']
        if self==other_rel:
            similarity += relationship_weightage['Entities']
        if self.cardinality==other_rel.cardinality:
            similarity += relationship_weightage['Entities']
        return similarity

    def __eq__(self, other):
        if self.strong_relationship==other.strong_relationship and self.number_of_attributes==other.number_of_attributes:
            if self.entity_one == other.entity_one and self.entity_two==other.entity_two:
                return True
        return False

class ERD:
    def __init__(self):
        self.entities = []
        self.relationships = []

    def compare_ERD(self, other_ERD):
        similarity_score = 0
        all_entity_similarities = []
        for ent1 in self.entities:
            max_similarity = 0
            for ent2 in other_ERD.entities:
                sim = ent1.compare_entities(ent2)
                max_similarity = sim if sim>max_similarity else max_similarity
                # print(max_similarity)
            all_entity_similarities.append(max_similarity)
        if len(self.relationships)>0:
            all_rel_similarities = []
            for rel1 in self.relationships:
                max_similarity = 0
                for rel2 in other_ERD.relationships:
                    sim = rel1.compare_relationships(rel2)
                    max_similarity = sim if sim > max_similarity else max_similarity
                all_rel_similarities.append(max_similarity)
            similarity_score += ERD_weightage['Relationships'] * np.mean(all_rel_similarities)
        else:
            similarity_score += ERD_weightage['Relationships']
        similarity_score += ERD_weightage['Entities']*np.mean(all_entity_similarities)
        return similarity_score*100

    def add_entity(self, entity):
        self.entities.append(entity)

    def remove_entity(self, entity):
        self.entities.remove(entity)

    def add_relationship(self, rel):
        self.relationships.append(rel)

    def remove_relationship(self, rel):
        self.relationships.remove(rel)

    def entity_in_ERD(self, new_entity):
        for entity in self.entities:
            if entity.name == new_entity.name:
                return True
        return False

    def get_entity_by_name(self, name):
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None

    def relationship_in_ERD(self, new_rel):
        for rel in self.relationships:
            if rel.name == new_rel.name:
                return True
        return False

    def get_relationship_by_name(self, name):
        for rel in self.relationships:
            if rel.name == name:
                return rel
        return None


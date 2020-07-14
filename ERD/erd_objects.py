
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

class Attribute:
    def __init__(self):
        self.name = ""
        self.key_attribute = False
        self.strong_attribute = True
        self.multi_valued = False
        self.belongs_to_entity = None
        self.belongs_to_relationship = False

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

class ERD:
    def __init__(self):
        self.entities = []
        self.relationships = []

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


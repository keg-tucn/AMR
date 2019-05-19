class Frameset:
    def __init__(self, lemma):
        self.lemma = lemma
        self.rolesets = []

    def add_roleset(self, roleset):
        self.rolesets.append(roleset)

    @classmethod
    def build_from_XML(cls, frame_tree):
        frameset_node = frame_tree.getroot()
        predicate_node = frameset_node._children[0]

        lemma = predicate_node.attrib["lemma"]

        instance = cls(lemma)

        for roleset in predicate_node:
            instance.add_roleset(Roleset.build_from_XML(roleset))

        return instance

    @classmethod
    def merge_framesets(cls, frameset_1, frameset_2):
        merged_frameset = cls(frameset_1.lemma)
        merged_frameset.rolesets = frameset_1.rolesets + frameset_2.rolesets

        return merged_frameset


class Roleset:
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.roles = []

    @classmethod
    def build_from_XML(cls, roleset_tree):
        instance = cls(roleset_tree.attrib["id"], roleset_tree.attrib["name"])
        roles_list = next((x for x in roleset_tree._children if x.tag == "roles"), None)
        for role in roles_list:
            instance.add_role(Role.build_from_XML(role))

        return instance

    def add_role(self, role):
        self.roles.append(role)


class Role:
    def __init__(self, index, description):
        self.index = index
        self.description = description

    @classmethod
    def build_from_XML(cls, role_tree):
        return cls(role_tree.attrib["n"], role_tree.attrib["descr"])

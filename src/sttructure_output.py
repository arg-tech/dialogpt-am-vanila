from collections import defaultdict

class ArgumentStructureGenerator:
    def __init__(self):
        pass
    
    def construct_tree(self, sentences, relations):
        """Construct an argument tree, removing only reverse connections"""
        graph = defaultdict(list)
        relation_map = {}

        # Build initial graph and store relation types
        for parent, child, relation_type in relations:
            graph[parent].append(child)
            relation_map[(parent, child)] = relation_type

        # Preserve all direct relations
        direct_relations = set((parent, child) for parent in graph for child in graph[parent])

        # Remove only one direction if both (A -> B) and (B -> A) exist
        final_relations = set(direct_relations)  # Copy initial relations
        for parent, child in direct_relations:
            if (child, parent) in direct_relations:
                # Use lexicographic order to consistently remove only one direction
                if parent < child:
                    final_relations.discard((child, parent))
                else:
                    final_relations.discard((parent, child))

        # Build the final tree
        tree = defaultdict(list)
        for parent, child in final_relations:
            if (parent, child) in relation_map:
                tree[parent].append((child, relation_map[(parent, child)]))

        return tree

    def generate_argument_structure_from_relations(self, sentences, relations):
        """Generate structured argument graph, removing only reverse connections"""
        tree = self.construct_tree(sentences, relations)
        return dict(tree)


# Example usage
generator = ArgumentStructureGenerator()
sentences = ['A', 'B', 'C', 'D']
relations = [
    ('A', 'B', 'support'),
    ('B', 'A', 'support'),
    ('A', 'C', 'attack'),
    ('C', 'A', 'attack'),
    ('B', 'D', 'support'),
    ('C', 'D', 'support'),
    ('B', 'C', 'support'),
    ('C', 'B', 'attack')  # B->C should be removed since C->B exists
]

output = generator.generate_argument_structure_from_relations(sentences, relations)
print(output)

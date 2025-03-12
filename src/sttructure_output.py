from collections import defaultdict

class ArgumentStructureGenerator:
    def __init__(self):
        pass
    
    def find_all_paths(self, graph, start, end, path=None):
        """Find all paths between start and end nodes in a directed graph"""
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = self.find_all_paths(graph, node, end, path)
                paths.extend(newpaths)
        return paths

    def construct_tree(self, sentences, relations):
        """Construct an argument tree with direct paths and remove redundant ones"""
        graph = defaultdict(list)
        relation_map = {}

        # Build initial graph and store relation types
        for parent, child, relation_type in relations:
            graph[parent].append(child)
            relation_map[(parent, child)] = relation_type

        # Detect all paths
        all_paths = {}
        for start in sentences:
            for end in sentences:
                if start != end:
                    paths = self.find_all_paths(graph, start, end)
                    if paths:
                        all_paths[(start, end)] = paths

        # Keep only the shortest connections between the same nodes
        direct_relations = set()
        for (start, end), paths in all_paths.items():
            if paths:
                shortest_path = min(paths, key=len)  # Find the shortest path
                if len(shortest_path) == 2:  # Only keep direct paths
                    direct_relations.add((shortest_path[0], shortest_path[1]))

        # Remove only one direction if both (A -> B) and (B -> A) exist
        final_relations = set(direct_relations)  # Start with the same set
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
        """Generate structured argument graph with direct relations only"""
        tree = self.construct_tree(sentences, relations)
        return dict(tree)


# Example usage
generator = ArgumentStructureGenerator()
sentences = ['A', 'B', 'C', 'D']
relations = [
    ('A', 'B', 'support'),
    ('B', 'A', 'support'),
    ('A', 'C', 'support'),
    ('C', 'A', 'attack'),
    ('B', 'D', 'support'),
    ('C', 'D', 'support'),
    ('B', 'C', 'support'),
    ('C', 'B', 'attack')  # B->C should be removed since C->B exists
]

output = generator.generate_argument_structure_from_relations(sentences, relations)
print(output)

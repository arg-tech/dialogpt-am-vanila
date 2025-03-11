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
        """Construct an argument tree while removing redundant or indirect connections"""
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

        # Keep only the shortest direct connections
        direct_relations = set()
        for (start, end), paths in all_paths.items():
            shortest_path = min(paths, key=len)
            if len(shortest_path) == 2:  # Only keep direct paths
                direct_relations.add((shortest_path[0], shortest_path[1]))

        # Remove reverse connections (B->A if A->B exists)
        final_relations = set()
        for parent, child in direct_relations:
            if (child, parent) not in direct_relations:
                final_relations.add((parent, child))

        # Remove direct connections if there's an indirect shorter path
        filtered_relations = set(final_relations)
        for (a, c) in final_relations:
            for (b, d) in final_relations:
                if a == b and c != d and (c, d) in final_relations:
                    filtered_relations.discard((a, d))  # Remove longer connection

        # Remove A -> C if A -> B -> C exists
        for (a, c) in list(filtered_relations):
            for (a2, b) in filtered_relations:
                if a == a2 and (b, c) in filtered_relations:
                    filtered_relations.discard((a, c))  # Remove A -> C if A -> B -> C exists

        # Build the final tree
        tree = defaultdict(list)
        for parent, child in filtered_relations:
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
    ('A', 'C', 'support'),
    ('B', 'D', 'support'),
    ('C', 'D', 'support'),
    ('B', 'C', 'support'),
    ('C', 'B', 'attack')  # 
]

output = generator.generate_argument_structure_from_relations(sentences, relations)
print(output)

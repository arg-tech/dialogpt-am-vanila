from collections import defaultdict
from itertools import combinations

class ArgumentStructureGenerator:
    def __init__(self):
        pass
    
    def find_all_paths(self, graph, start, end, path=None):
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
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def construct_tree(self, sentences, relations):
        graph = defaultdict(list)
        for parent, child, relation_type in relations:
            graph[parent].append(child)
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

        # Remove reverse connections (B->A if A->B exists)
        final_relations = set()
        for parent, child in direct_relations:
            if (child, parent) not in direct_relations:
                final_relations.add((parent, child))

        # Build the final tree
        tree = defaultdict(list)
        for parent, child in final_relations:
            if (parent, child) in relation_map:
                tree[parent].append((child, relation_map[(parent, child)]))

        return tree

    def get_paths(self, tree, start, end, path=None):
        if path is None:
            path = []
        path.append(start)
        if start == end:
            return [path]
        if start not in tree:
            return []
        paths = []
        for child, relation_type in tree[start]:
            if child not in path:            
                newpaths = self.get_paths(tree, child, end, path[:])
                for newpath in newpaths:
                    newpath.append(relation_type)
                    paths.append(newpath)
        return paths

    def generate_argument_structure_from_relations(self, sentences, relations):
        argument_relations = {}
        tree = self.construct_tree(sentences, relations)
        all_paths = []
        for i in range(len(sentences) - 1):
            start = sentences[i]
            end = sentences[i + 1]
            paths = self.get_paths(tree, start, end)
            all_paths.extend(paths)
        for path in all_paths:
            argument_relations[path[0]] = (path[1:])
        return argument_relations


# Example usage
generator = ArgumentStructureGenerator()
sentences = ['A', 'B', 'C', 'D']
relations = [
    ('A', 'B', 'support'),
    ('A', 'C', 'attack'),
    ('B', 'D', 'support'),
    ('C', 'D', 'support'),
    ('B', 'C', 'support'),
    ('C', 'B', 'attack')  # B->C should be removed since C->B exists
]

output = generator.generate_argument_structure_from_relations(sentences, relations)
print(output)

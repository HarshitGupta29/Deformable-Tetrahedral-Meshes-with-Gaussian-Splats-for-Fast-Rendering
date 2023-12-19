import torch

def batch_tetrahedral_volume(vertices):
    """Calculate the volume of each tetrahedron in a batch."""
    a = vertices[:, 1] - vertices[:, 0]
    b = vertices[:, 2] - vertices[:, 0]
    c = vertices[:, 3] - vertices[:, 0]
    # Using einsum for batch cross product and dot product
    cross_prod = torch.einsum('ijk,ikl->ijl', b.unsqueeze(1), c.unsqueeze(1)).squeeze(1)
    return torch.abs(torch.einsum('ij,ij->i', a, cross_prod)) / 6

def batch_max_edge_length(vertices):
    """Calculate the maximum edge length for each tetrahedron in a batch."""
    n = vertices.size(0)
    edges = [vertices[:, i] - vertices[:, j] for i in range(4) for j in range(i)]
    edges = torch.stack(edges, dim=1)  # Shape: (n, 6, 3)
    return torch.max(torch.norm(edges, dim=2), dim=1)[0]

def batch_tetrahedral_skewness(vertices):
    """Calculate the skewness of each tetrahedral cell in a batch."""
    R = batch_max_edge_length(vertices) / torch.sqrt(torch.tensor(6.0))
    V_cell = batch_tetrahedral_volume(vertices)
    V_ideal = 8 * torch.sqrt(torch.tensor(3.0)) * R**3 / 27
    return (V_ideal - V_cell) / V_ideal

# Example usage
vertices_batch = torch.tensor([
    [[0, 0, 0], [0.001, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1000, 1000, 1000], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
], dtype=torch.float)  # Shape: (2, 4, 3) for 2 tetrahedrons
skewness_batch = batch_tetrahedral_skewness(vertices_batch)
print(skewness_batch)
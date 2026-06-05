import torch

def euclidean_dist(x, y):
    """
    Compute Euclidean distance between query embeddings and prototypes.
    
    Args:
        x (torch.Tensor): Query embeddings, shape [num_query, embedding_dim]
        y (torch.Tensor): Prototypes, shape [num_classes, embedding_dim]
    
    Returns:
        torch.Tensor: Distance matrix, shape [num_query, num_classes]
    """
    n = x.size(0)  # num_query
    m = y.size(0)  # num_classes
    d = x.size(1)  # embedding_dim
    assert d == y.size(1), "Embedding dimensions must match"

    # Expand dimensions for broadcasting: [num_query, 1, embedding_dim]
    x = x.unsqueeze(1)  # Shape: [num_query, 1, embedding_dim]
    # Expand prototypes: [1, num_classes, embedding_dim]
    y = y.unsqueeze(0)  # Shape: [1, num_classes, embedding_dim]
    
    # Compute squared Euclidean distance
    dists = torch.sum((x - y) ** 2, dim=2)  # Shape: [num_query, num_classes]
    return dists
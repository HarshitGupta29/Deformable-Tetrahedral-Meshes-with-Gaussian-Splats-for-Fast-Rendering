import torch
#from pytorch3d.loss import chamfer_distance

def chamfer_dist(means, whitening, points):
    """
    Compute the Chamfer distance between two arrays of points in PyTorch.
    
    Parameters:
    - means: A tensor of shape (N, 3) containing N means of Gaussians.
    - whitening: A tensor of shape (N, 3, 3) containing whitening matrices for the Gaussians.
    - points: A tensor of shape (M, 3) containing M points in 3-dimensional space.
    
    Returns:
    - A scalar tensor containing the average Chamfer distance between the two point sets.
    """
    N, _ = means.shape
    M, _ = points.shape
    batch_factor = 1000
    batch_size = N // batch_factor

    # Initialize min_m to store indices of minimum distances
    min_m_indices = torch.empty(N, dtype=torch.long, device=means.device)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            #print('iter:', i)
            batch_means = means[i:i + batch_size]
            batch_whitening = whitening[i:i + batch_size]

            diff = batch_means[:, None, :] - points[None, :, :]
            diff = whiten(batch_whitening, diff)
            distances = torch.norm(diff, dim=2) ** 2

            # Store indices of minimum distances
            _, indices = torch.min(distances, dim=1)
            min_m_indices[i:i + batch_size] = indices

    # Use the indices to gather the corresponding points
    gathered_points = points[min_m_indices]
    #print(gathered_points.shape)
    #input()
    # Compute the Chamfer distance
    k = whiten(whitening, (means - gathered_points)[:, None, :])
    w1, _, w2 = k.shape
    k = k.reshape((w1,w2))
    chamfer_dist = torch.mean(torch.norm(k, dim=1))
    return chamfer_dist

def whiten(W, D):
    """ 
    Apply whitening matrices W to differences D.
    
    Parameters:
    - W: A tensor of shape (N, 3, 3) containing whitening matrices for the Gaussians.
    - D: A tensor of shape (N, M, 3) representing differences between points.
    
    Returns:
    - A tensor of shape (N, M, 3) after applying the whitening matrices.
    """
    return torch.einsum('nij,nmj->nmi', W, D)

def batch_whitening(batch_cov_matrix):
    """
    Compute the whitening matrix for a batch of 3x3 covariance matrices.

    Parameters:
    - batch_cov_matrix: A tensor of shape (B, 3, 3) containing a batch of B 
      covariance matrices, where each matrix is 3 x 3 dimensional.

    Returns:
    - A tensor of shape (B, 3, 3) containing the whitening matrices for the batch 
      of covariance matrices.
    """
    eig_vals, eig_vecs = torch.linalg.eigh(batch_cov_matrix, UPLO='U')
    D_inv_sqrt = torch.linalg.inv(torch.sqrt(eig_vals + 1e-10).diag_embed())
    W = torch.matmul(torch.matmul(eig_vecs, D_inv_sqrt), eig_vecs.transpose(-2, -1))
    return W
# N = 10000
# M = 300 

# means = torch.randn(N, 3)
# whitening = torch.eye(3).repeat(N, 1, 1)
# points = torch.randn(M, 3)
# c = chamfer_dist(means, whitening, points)
# print(c)
# c = chamfer_distance(means[None,:,:],points[None,:,:])
# print(c)

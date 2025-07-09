#!/usr/bin/env python

"""
Comprehensive Assessment of Rigid Poses (CARP) Dock
_             _             _     _

This script performs rigid-body ligand docking onto a protein structure using brute-force sampling of ligand rotations and translations, followed by clustering of valid poses. 
It loads protein and ligand structures, generates a grid of possible ligand positions, samples random ligand orientations, and filters out poses that clash with the protein or violate user-specified burial constraints. 
Valid ligand poses are clustered in 6D pose space (translation and rotation), and representative structures from each cluster can be written to output files. 
The script supports GPU acceleration with pytoch for computationally intensive steps and provides command-line options for customization.

Most of this script was vibe-coded, though it was validated on some test targets. 
Use at your own risk.

bfry@g.harvard.edu
"""

import json
import argparse
from pathlib import Path

import torch
import prody as pr
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from utils.burial_calc import batch_compute_fast_ligand_burial_mask_gpu
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KernelDensity


def parse_args():
    parser = argparse.ArgumentParser(description="Rigid-body ligand docking with clustering.")
    parser.add_argument('input_protein', type=str, help='Path to input protein PDB file. Should be an all-glycine backbone.')
    parser.add_argument('input_ligand', type=str, help='Path to input ligand PDB file.')
    parser.add_argument('output_dir', type=str, help='Directory to write output PDB files')
    parser.add_argument('--inside_hull', type=str, default='', help='Comma-separated ligand atom names required to be inside hull')
    parser.add_argument('--outside_hull', type=str, default='', help='Comma-separated ligand atom names required to be outside hull')
    parser.add_argument('--test_point_grid_width', type=float, default=0.5, help='Grid width for test points')
    parser.add_argument('--n_ligand_rotations', type=int, default=100, help='Number of ligand rotations to sample. More rotations (~1000) are probably better but will reduce the speed of computation.')
    parser.add_argument('--clash_distance_tolerance', type=float, default=2.5, help='Minimum allowed distance to avoid clash')
    parser.add_argument('--no_write', action='store_true', help='Do not write output files')
    parser.add_argument('--max_batch_size', type=int, default=2000, help='Max batch size for GPU operations')
    parser.add_argument('--ligand_rotation_batch_size', type=int, default=20, help='Batch size for ligand rotations')
    parser.add_argument('--search_box_padding', type=float, default=0.0, help='Padding for search box, if positive adds more volume to search, if negative removes volume.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--silent', action='store_true', help='Suppress non-error output')
    parser.add_argument('--alpha_hull_alpha', type=float, default=9.0, help='Alpha parameter for convex hull construction. Larger numbers generate more box-like hulls. Smaller numbers wrap the point cloud tighter. 9.0 is default for helical bundles. Folds with larger pockets may need larger values (~100.0)')
    parser.add_argument('--dbscan_eps', type=float, default=1.0, help='Epsilon parameter for DBscan, a larger value should produce fewer clusters, a smaller value will produce more clusters.')
    parser.add_argument('--kmeans_nclusters', type=int, default=10, help='Number of kmeans clusters to generate if using kmeans clustering')
    parser.add_argument('--clustering_algorithm', type=str, default='dbscan', choices=['dbscan', 'kmeans'], help='Clustering algorithm to use: "dbscan" or "kmeans"')
    return parser.parse_args()


def random_sample_rotations(ligand_coords, n):
    """
    Samples random rotation matrices for the ligand and applies them to the ligand coordinates.
    Returns R x L x 3 matrix of rotated ligand coords.
    """

    rotation = torch.from_numpy(Rotation.random(num=n).as_matrix()).to(ligand_coords.device, ligand_coords.dtype)
    rotated_ligands = ligand_coords @ rotation.transpose(2, 1)

    return rotated_ligands


def load_backbone_info(input_protein, device, dtype):
    protein = pr.parsePDB(str(input_protein)).select('not element H').copy()

    # Get the CA atoms and calculate the center of mass.
    protein_ca_coords = protein.ca.getCoords()
    protein_com = protein_ca_coords.mean(axis=0)

    # Center protein at the origin about the CA atoms.
    centered_coords = protein.getCoords() - protein_com
    protein.setCoords(centered_coords)
    protein.setChids('A')

    return protein, torch.from_numpy(centered_coords).to(device, dtype), torch.from_numpy(protein_ca_coords).to(device, dtype), torch.from_numpy(protein_com).to(device, dtype)


def load_ligand_info(input_ligand, device, dtype):
    ligand = pr.parsePDB(str(input_ligand)).copy()
    ligand_coords = ligand.getCoords()
    ligand_names = ligand.getNames()

    ligand_com = ligand_coords.mean(axis=0)
    ligand_coords = ligand_coords - ligand_com  
    ligand.setChids('B')

    return ligand, torch.from_numpy(ligand_coords).to(device, dtype), torch.from_numpy(ligand_com).to(device, dtype), ligand_names


def compute_test_points(rotated_core_coords, device, max_batch_size, grid_width, search_box_padding, verbose, alpha):
    min_corner = rotated_core_coords.min(dim=0).values - search_box_padding
    max_corner = rotated_core_coords.max(dim=0).values + search_box_padding

    x = torch.arange(min_corner[0], max_corner[0], grid_width, device=device)
    y = torch.arange(min_corner[1], max_corner[1], grid_width, device=device)
    z = torch.arange(min_corner[2], max_corner[2], grid_width, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=-1).to(rotated_core_coords.dtype)

    if verbose: print(f'Computed {grid_points.shape[0]} initial test grid points')

    mask = batch_compute_fast_ligand_burial_mask_gpu(rotated_core_coords, grid_points, num_rays=10, max_batch_size=max_batch_size, alpha=alpha)
    test_points = grid_points[mask]

    if verbose: print(f"Filtered to {test_points.shape[0]}, points")
    return test_points


def rotate_axes_of_principal_variation(protein_core_coords):
    # Perform PCA on protein_core_coords
    core_mean = protein_core_coords.mean(axis=0)
    core_centered = protein_core_coords - core_mean
    cov = torch.cov(core_centered.T)
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float32))

    # Sort eigenvectors by eigenvalues in descending order
    order = torch.argsort(eigvals, descending=True)
    rotation_matrix = eigvecs[:, order].to(protein_core_coords.dtype)

    # Ensure transformation doesn't flip chirality (det = +1)
    if torch.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, -1] *= -1

    return rotation_matrix, core_centered, core_mean


def compute_brute_force_rototranslations(n_ligand_rotations, ligand_coords, test_points, centered_coords, clash_distance_tolerance, max_batch_size: int = 4096, ligand_rotation_batch_size: int = 100):
    # Compute all rototranslations.
    rotations_fn = random_sample_rotations
    ligand_rotations = rotations_fn(ligand_coords, n_ligand_rotations)[:, None, ...]
    test_points_expanded = test_points[None, ..., None, :].expand(size=(1, -1, ligand_rotations.shape[-2], 3))

    all_roto_translations = []
    rotation_chunks = torch.chunk(ligand_rotations, max(ligand_rotations.shape[0] // ligand_rotation_batch_size, 1))
    for rotation_chunk in tqdm(rotation_chunks, total=len(rotation_chunks)):

        # Broadcast addition to c x T x L x 3
        chunk_roto_translations = (rotation_chunk + test_points_expanded).flatten(start_dim=0, end_dim=1)

        all_min_conf_dists = []
        for chunk in torch.chunk(chunk_roto_translations, max(chunk_roto_translations.shape[0] // max_batch_size, 1)):
            min_conformer_distance = torch.cdist(chunk, centered_coords).min(dim=-1).values.min(dim=-1).values
            all_min_conf_dists.append(min_conformer_distance)
        distances = torch.cat(all_min_conf_dists)

        # Get the rototranslations where no atoms are clashing with the backbone.
        valid_indices = (distances > clash_distance_tolerance).nonzero(as_tuple=True)

        # Remaining ligand poses not clashing with the backbone: N x L x 3 
        chunk_roto_translations = chunk_roto_translations[valid_indices]
        all_roto_translations.append(chunk_roto_translations)

    return torch.cat(all_roto_translations)


def batched_rigid_alignment(new_xyzs, ref_xyz):
    """
    new_xyzs: (B, N, 3)   # B conformations
    ref_xyz: (N, 3)       # Reference
    Returns: (B, 6)       # Rigid body transform for each batch
    """
    B, N, _ = new_xyzs.shape

    # Compute centers of mass
    new_coms = np.mean(new_xyzs, axis=1, keepdims=True)   # (B, 1, 3)
    ref_com = np.mean(ref_xyz, axis=0, keepdims=True)     # (1, 3)
    T = (new_coms[:, 0, :] - ref_com)                    # (B, 3)

    # Centered coordinates
    X = new_xyzs - new_coms   # (B, N, 3)
    R_ = ref_xyz - ref_com    # (N, 3)

    # Compute batched covariance matrices (B, 3, 3)
    Cs = np.einsum('bni,nj->bij', X, R_)

    # Singular Value Decomposition in batch
    U, S, Vt = np.linalg.svd(Cs)
    # Reflection correction to ensure det(Q)=+1
    det_sign = np.sign(np.linalg.det(np.matmul(U, Vt)))
    S_fix = np.eye(3)[np.newaxis, :, :].repeat(B, axis=0)
    S_fix[:, -1, -1] = det_sign
    Q = np.matmul(np.matmul(U, S_fix), Vt)  # (B, 3, 3)

    # Convert to rotation vectors
    rotvecs = Rotation.from_matrix(Q).as_rotvec() # (B, 3)

    # Concatenate translation and rotation
    out = np.concatenate([T, rotvecs], axis=-1)  # (B, 6)
    return out


def cluster_filtered_rototranslations(filtered_rototranslations, ligand_coords, dbscan_eps, kmeans_nclusters, clustering_algorithm):
    # Pose_vectors is (B, 6): each row is [dx,dy,dz, rx,ry,rz]
    pose_vectors = batched_rigid_alignment(filtered_rototranslations, ligand_coords)

    # Optionally, rescale translation/rotation parts:
    pose_vectors_scaled = pose_vectors.copy()
    pose_vectors_scaled[:, :3] /= 1.0   # Maybe leave translations as-is...
    pose_vectors_scaled[:, 3: ] /= 0.5  # set 30 degrees in radians roughly equal to 1.0 A

    # Clustering: support both DBSCAN and KMeans
    # By default, use DBSCAN. To use KMeans, set dbscan_eps to None and provide n_clusters.
    if clustering_algorithm == 'dbscan':
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=2)
        labels = clusterer.fit_predict(pose_vectors_scaled)
    elif clustering_algorithm == 'kmeans':
        clusterer = KMeans(n_clusters=kmeans_nclusters, random_state=0)
        labels = clusterer.fit_predict(pose_vectors_scaled)
    else:
        raise ValueError(f'No clustering algorithm named {clustering_algorithm}')

    return labels, pose_vectors_scaled


def kde_cluster_MAP_pose(X, bandwidth=0.5):
    """
    X is (N, D) array of (N) observed pose vectors in D dimensions
    Returns: index of X which is "most typical" pose according to the KDE
    """
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(X)
    log_density = kde.score_samples(X)

    # Pick pose with maximum density under KDE
    return np.argmax(log_density)


def main(
    input_protein, input_ligand, output_dir, constraints, 
    test_point_grid_width = 0.5, n_ligand_rotations=1000, clash_distance_tolerance=2.5, no_write=False, 
    max_batch_size = 2000, ligand_rotation_batch_size = 20, search_box_padding = 0.0, device: torch.device = torch.device('cuda:0'), 
    dtype: torch.dtype = torch.float32, verbose: bool = True, alpha: float = 9.0, dbscan_eps: float = 1.0, 
    kmeans_nclusters: int = 10, clustering_algorithm: str = 'dbscan'
):
    # Load starting coordinates/rigid conformers.
    protein, centered_coords, protein_ca_coords, protein_com = load_backbone_info(input_protein, device, dtype)
    ligand, ligand_coords, ligand_com, ligand_names = load_ligand_info(input_ligand, device, dtype)
    centered_ca_coords = protein_ca_coords - protein_com

    # Align the protein core to its principal axes for easier sampling and analysis.
    rotation_matrix, core_centered, core_mean = rotate_axes_of_principal_variation(centered_ca_coords)

    # Project the centered coords onto the principal axes and add back mean offset.
    rotated_core_coords = (core_centered @ rotation_matrix) + core_mean
    centered_ca_coords = ((centered_ca_coords - core_mean) @ rotation_matrix) + core_mean
    centered_coords = ((centered_coords - core_mean) @ rotation_matrix) + core_mean
    protein.setCoords(centered_coords.cpu().to(torch.float32).numpy())

    test_points = compute_test_points(rotated_core_coords, device, max_batch_size=max_batch_size, grid_width=test_point_grid_width, search_box_padding=search_box_padding, verbose=verbose, alpha=alpha)
    roto_translations = compute_brute_force_rototranslations(n_ligand_rotations, ligand_coords, test_points, centered_coords, clash_distance_tolerance, max_batch_size=max_batch_size, ligand_rotation_batch_size=ligand_rotation_batch_size)

    # compute masks tracking what atoms have constraints.
    ligand_in_hull_atoms_mask = torch.tensor([x in constraints['inside_hull'] for x in ligand_names]).bool().to(device)
    ligand_outside_hull_atoms_mask = torch.tensor([x in constraints['outside_hull'] for x in ligand_names]).bool().to(device)
    ligand_burial_mask = batch_compute_fast_ligand_burial_mask_gpu(centered_ca_coords, roto_translations.reshape(-1, 3), num_rays=10, max_batch_size=max_batch_size, alpha=alpha)
    ligand_burial_mask = ligand_burial_mask.reshape(-1, len(ligand_names))

    # Filter out poses by burial constraints.
    final_constraint_mask = (ligand_burial_mask[:, ligand_in_hull_atoms_mask].all(dim=-1) & (~ligand_burial_mask[:, ligand_outside_hull_atoms_mask]).all(dim=-1))
    filtered_rototranslations = roto_translations[final_constraint_mask]
    if verbose: print(f'Found {len(filtered_rototranslations)} roto-translations satisfying constraints.')

    if filtered_rototranslations.shape[0] == 0:
        return filtered_rototranslations, None
    elif filtered_rototranslations.shape[0] < 25:
        cluster_labels = np.zeros(filtered_rototranslations.shape[0])
    else:
        # Cluster on translation and rotation relative to reference conformer.
        cluster_labels, pose_vectors = cluster_filtered_rototranslations(filtered_rototranslations.cpu().to(torch.float32).numpy(), ligand_coords.cpu().to(torch.float32).numpy(), dbscan_eps=dbscan_eps, kmeans_nclusters=kmeans_nclusters, clustering_algorithm=clustering_algorithm)

    unique_clusters = np.unique(cluster_labels)
    if verbose: print(f'Clustered rototranslations into {len(unique_clusters)} clusters.')

    # Optionally write some cluster samples to disk
    if not no_write:
        print(f'Writing outputs to {output_dir.absolute()}')
        cluster_idx = 0
        for label in unique_clusters:

            if label == -1:
                continue  # skip noise

            if len(unique_clusters) == 1:
                for idx, coord in enumerate(filtered_rototranslations):
                    lig_coords_npy = coord.cpu().to(torch.float32).numpy()
                    ligand.setCoords(lig_coords_npy)
                    pr.writePDB(str(output_dir / f'cluster_{cluster_idx+1}_{idx}.pdb'), protein + ligand)
            else:
                idx = kde_cluster_MAP_pose(pose_vectors[cluster_labels == label])
                lig_coords_npy = filtered_rototranslations[cluster_labels == label][idx].cpu().to(torch.float32).numpy()
                ligand.setCoords(lig_coords_npy)
                pr.writePDB(str(output_dir / f'cluster_{cluster_idx+1}_{idx}.pdb'), protein + ligand)

            cluster_idx += 1

    return filtered_rototranslations, cluster_labels


if __name__ == "__main__":
    # Parse command-line arguments.
    args = parse_args()

    # Parse comma-separated lists for inside_hull and outside_hull
    inside_hull_list = [x for x in args.inside_hull.split(',') if x] if args.inside_hull else []
    outside_hull_list = [x for x in args.outside_hull.split(',') if x] if args.outside_hull else []

    constraints = {
        'inside_hull': inside_hull_list,
        'outside_hull': outside_hull_list
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=False)

    args_dict = vars(args)
    with open(output_dir / "input_args.json", "w") as f:
        json.dump(args_dict, f, indent=2)

    main(
        input_protein=args.input_protein,
        input_ligand=args.input_ligand,
        output_dir=output_dir,
        constraints=constraints,
        test_point_grid_width=args.test_point_grid_width,
        n_ligand_rotations=args.n_ligand_rotations,
        clash_distance_tolerance=args.clash_distance_tolerance,
        no_write=args.no_write,
        max_batch_size=args.max_batch_size,
        ligand_rotation_batch_size=args.ligand_rotation_batch_size,
        search_box_padding=args.search_box_padding,
        device=torch.device(args.device),
        dtype=torch.float32,
        verbose=not args.silent,
        alpha=args.alpha_hull_alpha,
        dbscan_eps=args.dbscan_eps,
        kmeans_nclusters=args.kmeans_nclusters,
        clustering_algorithm=args.clustering_algorithm,
    )
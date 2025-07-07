# Comprehensive Assessment of Rigid Poses (CARP) Dock

Can be used to quickly generate starting poses for [NISE](https://github.com/polizzilab/NISE).

### Runs with the [LASErMPNN](https://github.com/polizzilab/LASErMPNN) python environment.

This script performs rigid-body ligand docking onto a protein structure using brute-force sampling of ligand rotations and translations, followed by clustering of valid poses. 
It loads protein and ligand structures, generates a grid of possible ligand positions, samples random ligand orientations, and filters out poses that clash with the protein or violate user-specified burial constraints. 
Valid ligand poses are clustered in 6D pose space (translation and rotation), and representative structures from each cluster can be written to output files. 
The script supports GPU acceleration with pytoch for computationally intensive steps and provides command-line options for customization.


Most of this script was vibe-coded, though it was validated on some test targets. 
Use at your own risk.

____

usage:

```bash
python carp_dock.py PATH_TO_ALL_GLY_PROTEIN_BACKBONE.pdb PATH_TO_PROTONATED_LIGAND.pdb ./path_to_output_dir/
```


```bash
usage: carp_dock.py [-h] [--inside_hull INSIDE_HULL] [--outside_hull OUTSIDE_HULL] [--test_point_grid_width TEST_POINT_GRID_WIDTH] [--n_ligand_rotations N_LIGAND_ROTATIONS] [--clash_distance_tolerance CLASH_DISTANCE_TOLERANCE] [--no_write] [--max_batch_size MAX_BATCH_SIZE] [--ligand_rotation_batch_size LIGAND_ROTATION_BATCH_SIZE]
                    [--search_box_padding SEARCH_BOX_PADDING] [--device DEVICE] [--silent] [--alpha_hull_alpha ALPHA_HULL_ALPHA]
                    input_protein input_ligand output_dir

Rigid-body ligand docking with clustering.

positional arguments:
  input_protein         Path to input protein PDB file. Should be an all-glycine backbone.
  input_ligand          Path to input ligand PDB file.
  output_dir            Directory to write output PDB files

options:
  -h, --help            show this help message and exit
  --inside_hull INSIDE_HULL
                        Comma-separated ligand atom names required to be inside hull
  --outside_hull OUTSIDE_HULL
                        Comma-separated ligand atom names required to be outside hull
  --test_point_grid_width TEST_POINT_GRID_WIDTH
                        Grid width for test points
  --n_ligand_rotations N_LIGAND_ROTATIONS
                        Number of ligand rotations to sample. More rotations (~1000) are probably better but will reduce the speed of computation.
  --clash_distance_tolerance CLASH_DISTANCE_TOLERANCE
                        Minimum allowed distance to avoid clash
  --no_write            Do not write output files
  --max_batch_size MAX_BATCH_SIZE
                        Max batch size for GPU operations
  --ligand_rotation_batch_size LIGAND_ROTATION_BATCH_SIZE
                        Batch size for ligand rotations
  --search_box_padding SEARCH_BOX_PADDING
                        Padding for search box, if positive adds more volume to search, if negative removes volume.
  --device DEVICE       Torch device (e.g., "cuda:0" or "cpu")
  --silent              Suppress non-error output
  --alpha_hull_alpha ALPHA_HULL_ALPHA
                        Alpha parameter for convex hull construction. Larger numbers generate more box-like hulls. Smaller numbers wrap the point cloud tighter. 9.0 is default for helical bundles. Folds with larger pockets may need larger values (~100.0)
```

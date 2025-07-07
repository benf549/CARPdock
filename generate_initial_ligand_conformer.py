#!/usr/bin/env python

"""
This script generates a 3D conformer for a molecule from a given SMILES string,
performs geometry optimization, and saves the resulting structure as a PDB file.

Usage:
    python generate_initial_ligand_conformer.py <SMILES> <output_pdb_path>

Arguments:
    smiles            Input SMILES string representing the molecule.
    output_pdb_path   Output PDB filename to save the generated 3D conformer.

bfry@g.harvard.edu
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate initial 3D conformer from a SMILES string and save as SDF and PDB.")
    parser.add_argument("smiles", help="Input SMILES string")
    parser.add_argument("output_pdb_path", help="Output PDB filename (default: output.pdb)")
    args = parser.parse_args()

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    mol = AllChem.AddHs(mol)

    options = AllChem.ETKDGv3()
    AllChem.EmbedMolecule(mol, options)
    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)

    Chem.MolToPDBFile(mol, args.output_pdb_path)
    print('wrote', args.output_pdb_path)


if __name__ == "__main__":
    main()

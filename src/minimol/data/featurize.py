"""Atom and bond features for the GINE encoder. RDKit parses the SMILES."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

ELEMENTS = ("B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[max(0, min(size - 1, index))] = 1.0
    return values


def atom_features(atom: Chem.Atom) -> list[float]:
    symbol = atom.GetSymbol()
    element = _one_hot(ELEMENTS.index(symbol) if symbol in ELEMENTS else len(ELEMENTS), len(ELEMENTS) + 1)
    charge = int(np.clip(atom.GetFormalCharge(), -2, 2))
    charge_hot = _one_hot(charge + 2, 5)
    if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        chiral = 1
    elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        chiral = 2
    elif atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        chiral = 0
    else:
        chiral = 3
    try:
        valence = int(atom.GetTotalValence())
    except Exception:
        valence = 0
    hydrogens = int(atom.GetTotalNumHs())
    features = []
    features.extend(element)
    features.extend(charge_hot)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.extend(_one_hot(chiral, 4))
    features.extend(_one_hot(min(valence, 6), 7))
    features.extend(_one_hot(min(hydrogens, 4), 5))
    features.append(1.0 if atom.IsInRing() else 0.0)
    return features


def atom_feature_dim() -> int:
    return len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))


def bond_features(bond: Chem.Bond | None, self_loop: bool = False) -> list[float]:
    order = [0.0, 0.0, 0.0, 0.0]
    conjugated = 0.0
    in_ring = 0.0
    stereo = [0.0, 0.0, 0.0, 0.0]
    if bond is not None and not self_loop:
        kind = bond.GetBondType()
        mapping = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2,
            Chem.BondType.AROMATIC: 3,
        }
        order[mapping.get(kind, 0)] = 1.0
        conjugated = 1.0 if bond.GetIsConjugated() else 0.0
        in_ring = 1.0 if bond.IsInRing() else 0.0
        stereo_map = {
            Chem.BondStereo.STEREONONE: 0,
            Chem.BondStereo.STEREOE: 1,
            Chem.BondStereo.STEREOZ: 2,
            Chem.BondStereo.STEREOANY: 3,
        }
        stereo[stereo_map.get(bond.GetStereo(), 0)] = 1.0
    features = order + [conjugated, in_ring] + stereo + [1.0 if self_loop else 0.0]
    return features


def bond_feature_dim() -> int:
    return len(bond_features(None, self_loop=True))


@dataclass
class MolGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    rev_index: torch.Tensor


def mol_to_graph(smiles: str, self_loops: bool = True) -> MolGraph | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atoms = [atom_features(atom) for atom in mol.GetAtoms()]
    edges: list[tuple[int, int]] = []
    attrs: list[list[float]] = []
    rev: list[int] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = bond_features(bond, self_loop=False)
        index = len(edges)
        edges.append((begin, end))
        edges.append((end, begin))
        attrs.extend([feat, feat])
        rev.extend([index + 1, index])
    if self_loops:
        loop = bond_features(None, self_loop=True)
        for atom_index in range(len(atoms)):
            index = len(edges)
            edges.append((atom_index, atom_index))
            attrs.append(loop)
            rev.append(index)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
        rev_index = torch.tensor(rev, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, bond_feature_dim(), dtype=torch.float32)
        rev_index = torch.zeros(0, dtype=torch.long)
    return MolGraph(
        x=torch.tensor(atoms, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        rev_index=rev_index,
    )


def collate_graphs(graphs: list[MolGraph]) -> dict:
    xs = []
    edges = []
    attrs = []
    batches = []
    revs = []
    node_offset = 0
    edge_offset = 0
    for graph_index, graph in enumerate(graphs):
        xs.append(graph.x)
        edges.append(graph.edge_index + node_offset)
        attrs.append(graph.edge_attr)
        batches.append(torch.full((graph.x.shape[0],), graph_index, dtype=torch.long))
        revs.append(graph.rev_index + edge_offset)
        node_offset += graph.x.shape[0]
        edge_offset += graph.edge_index.shape[1]
    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat(attrs, dim=0),
        "batch": torch.cat(batches, dim=0),
        "rev_index": torch.cat(revs, dim=0),
        "n_graphs": len(graphs),
    }


def morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, array)
    return array

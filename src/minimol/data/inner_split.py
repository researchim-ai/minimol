"""Immutable inner split of the official Malinois train split.

Official validation and test stay out. Overlapping 200 bp windows share a
locus bin so a variant and its neighbour cannot land in different partitions.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from minimol.errors import MinimolError
from minimol.jsonutil import write_json
from minimol.tracking import sha256_file

SPLIT_VERSION = "malinois_inner_v1"
SPLIT_VERSION_V2 = "malinois_inner_v2"
GIANT_COMPONENT = 2000
HASH_SALT = "minimol-inner-v1"
LOCUS_BIN_BP = 200
_LOCUS = re.compile(r"^(?:chr)?([0-9XY]+):(\d+)")


def locus_group(group_id: str) -> str:
    match = _LOCUS.match(str(group_id))
    if match is None:
        return "ungrouped:" + hashlib.sha256(str(group_id).encode()).hexdigest()
    chrom, position = match.group(1), int(match.group(2))
    return f"{chrom}:{position // LOCUS_BIN_BP}"


def partition_name(group: str, salt: str = HASH_SALT) -> str:
    digest = hashlib.sha256(f"{salt}:{group}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "inner_train"
    if bucket < 90:
        return "development"
    return "calibration"


def write_inner_split(data_path: Path, out_dir: Path) -> dict:
    """Write the split once. A second call with the same table returns the existing manifest."""

    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    data_hash = sha256_file(data_path)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("data_sha256") != data_hash or existing.get("version") != SPLIT_VERSION:
            raise MinimolError("inner split manifest exists for a different table or version")
        assignments = out_dir / existing["assignments_file"]
        if sha256_file(assignments) != existing["assignments_sha256"]:
            raise MinimolError("inner split assignments no longer match the manifest")
        return existing
    frame = pd.read_parquet(data_path, columns=["split", "sequence_hash", "group_id"], filters=[("split", "==", "train")])
    objects = frame.drop_duplicates("sequence_hash")
    objects = objects.assign(locus_group=objects["group_id"].map(locus_group))
    objects = objects.assign(partition=objects["locus_group"].map(partition_name))
    # Every sequence in a locus bin follows the bin, even if group ids differ.
    assignments = objects[["sequence_hash", "locus_group", "partition"]].copy()
    path = out_dir / "assignments.parquet"
    assignments.to_parquet(path, index=False)
    counts = {str(key): int(value) for key, value in assignments["partition"].value_counts().items()}
    manifest = {
        "version": SPLIT_VERSION,
        "data_sha256": data_hash,
        "source_split": "train",
        "official_validation_used": False,
        "official_test_used": False,
        "hash_salt": HASH_SALT,
        "locus_bin_bp": LOCUS_BIN_BP,
        "group_rule": "chromosome and position//200 parsed from the oligo id; overlapping windows stay together",
        "fractions_of_groups": {"inner_train": 0.8, "development": 0.1, "calibration": 0.1},
        "n_sequences": int(len(assignments)),
        "n_locus_groups": int(assignments["locus_group"].nunique()),
        "counts": counts,
        "assignments_file": path.name,
        "assignments_sha256": sha256_file(path),
        "note": (
            "Encoders train on inner_train only. Development selects epochs. "
            "Calibration fits probability maps. This is not a homology clustering."
        ),
    }
    write_json(manifest_path, manifest)
    return manifest


def _union_find(size: int):
    parent = list(range(size))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    return find, union


def write_inner_split_v2(data_path: Path, out_dir: Path, *, giant_component: int = GIANT_COMPONENT) -> dict:
    """Interval components. Giant components are cut on genomic blocks with a purge boundary."""

    from minimol.data.sequences import reverse_complement, sequence_hash

    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    data_hash = sha256_file(data_path)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("data_sha256") != data_hash or existing.get("version") != SPLIT_VERSION_V2:
            raise MinimolError("inner split v2 manifest exists for a different table or version")
        assignments = out_dir / existing["assignments_file"]
        if sha256_file(assignments) != existing["assignments_sha256"]:
            raise MinimolError("inner split v2 assignments no longer match the manifest")
        return existing
    frame = pd.read_parquet(
        data_path,
        columns=["split", "sequence_hash", "group_id", "sequence"],
        filters=[("split", "==", "train")],
    )
    objects = frame.drop_duplicates("sequence_hash").reset_index(drop=True)
    starts, ends, chroms = [], [], []
    for group_id, sequence in zip(objects["group_id"].tolist(), objects["sequence"].tolist(), strict=True):
        match = _LOCUS.match(str(group_id))
        length = max(len(str(sequence)), 1)
        if match is None:
            chroms.append("unknown")
            starts.append(0)
            ends.append(length)
            continue
        chrom, position = match.group(1), int(match.group(2))
        half = length // 2
        chroms.append(chrom)
        starts.append(position - half)
        ends.append(position - half + length)
    objects = objects.assign(chrom=chroms, start=starts, end=ends)
    find, union = _union_find(len(objects))
    rc_owner: dict[str, int] = {}
    for index, sequence in enumerate(objects["sequence"].tolist()):
        forward = sequence_hash(sequence)
        reverse = sequence_hash(reverse_complement(sequence))
        for digest in (forward, reverse):
            previous = rc_owner.get(digest)
            if previous is None:
                rc_owner[digest] = index
            else:
                union(previous, index)
    for chrom, part in objects.groupby("chrom", sort=False):
        order = part.sort_values(["start", "end"]).index.to_list()
        if not order:
            continue
        active = order[0]
        active_end = int(objects.at[active, "end"])
        for index in order[1:]:
            if int(objects.at[index, "start"]) < active_end:
                union(active, index)
                active_end = max(active_end, int(objects.at[index, "end"]))
            else:
                active = index
                active_end = int(objects.at[index, "end"])
        del chrom
    roots = [find(index) for index in range(len(objects))]
    objects = objects.assign(component=roots)
    sizes = objects.groupby("component").size().to_dict()
    partitions = []
    n_purged = 0
    n_giant = 0
    for component, part in objects.groupby("component", sort=False):
        members = part.sort_values(["start", "end"])
        if int(sizes[component]) <= giant_component:
            name = partition_name(f"component:{component}")
            partitions.extend((index, name) for index in members.index)
            continue
        n_giant += 1
        cut_at = [int(len(members) * fraction) for fraction in (0.8, 0.9)]
        cuts = sorted({int(members.iloc[min(position, len(members) - 1)]["start"]) for position in cut_at})
        for index, row in members.iterrows():
            start, end = int(row["start"]), int(row["end"])
            if any(start < cut < end for cut in cuts):
                partitions.append((index, "purged"))
                n_purged += 1
                continue
            midpoint = (start + end) // 2
            if midpoint < cuts[0]:
                partitions.append((index, "inner_train"))
            elif len(cuts) == 1 or midpoint < cuts[-1]:
                partitions.append((index, "development"))
            else:
                partitions.append((index, "calibration"))
    part_frame = pd.DataFrame(partitions, columns=["index", "partition"]).set_index("index")
    objects = objects.join(part_frame)
    assignments = objects[["sequence_hash", "component", "partition", "chrom", "start", "end"]].rename(
        columns={"component": "locus_group"}
    )
    assignments["locus_group"] = assignments["locus_group"].astype(str)
    path = out_dir / "assignments.parquet"
    assignments.to_parquet(path, index=False)
    counts = {str(key): int(value) for key, value in assignments["partition"].value_counts().items()}
    component_sizes = sorted(int(value) for value in sizes.values())
    manifest = {
        "version": SPLIT_VERSION_V2,
        "data_sha256": data_hash,
        "source_split": "train",
        "official_validation_used": False,
        "official_test_used": False,
        "hash_salt": HASH_SALT,
        "interval_assumption": "variant coordinate from the oligo id; insert is centered there and has the observed sequence length",
        "group_rule": "connected components of overlapping inserts, plus exact and reverse-complement duplicates",
        "giant_component_threshold": giant_component,
        "n_giant_components": n_giant,
        "n_purged_sequences": n_purged,
        "component_size_max": component_sizes[-1] if component_sizes else 0,
        "component_size_p50": component_sizes[len(component_sizes) // 2] if component_sizes else 0,
        "n_sequences": int((assignments.partition != "purged").sum()),
        "n_locus_groups": int(assignments.loc[assignments.partition != "purged", "locus_group"].nunique()),
        "counts": counts,
        "assignments_file": path.name,
        "assignments_sha256": sha256_file(path),
        "note": (
            "Small components stay intact. A giant component is cut on genomic coordinates; "
            "inserts that cross a cut are purged and are not an official test."
        ),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_inner_split(out_dir: Path) -> tuple[dict, pd.DataFrame]:
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    path = out_dir / manifest["assignments_file"]
    if sha256_file(path) != manifest["assignments_sha256"]:
        raise MinimolError("inner split assignments no longer match the manifest")
    return manifest, pd.read_parquet(path)

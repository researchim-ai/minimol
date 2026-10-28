"""Exact, reverse-complement, and near-duplicate leakage across splits.

Near duplicate means k=6 Jaccard similarity >= 0.80, or Hamming identity >= 0.95
for equal-length sequences. On a ~200 nt regulatory sequence, Jaccard 0.80 means
the sequences share at least 80% of their distinct 6-mers: enough to catch
single-base variants and same-locus windows, and well above the overlap of
unrelated sequences that only share common short motifs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from minimol.data.normalize import canonicalize_split
from minimol.data.sequences import normalize_sequence, reverse_complement, sequence_hash

NEAR_DUPLICATE_RULE = (
    "Near duplicate: k=6 Jaccard >= 0.80 across splits, or Hamming identity "
    ">= 0.95 when lengths match. Reverse complements are exact leaks. "
    "The Jaccard cutoff is high on purpose so shared short motifs do not look like leakage."
)


def kmer_set(seq: str, k: int = 6) -> set[int]:
    table = {"A": 0, "C": 1, "G": 2, "T": 3}
    found: set[int] = set()
    for start in range(0, len(seq) - k + 1):
        acc = 0
        good = True
        for offset in range(k):
            value = table.get(seq[start + offset])
            if value is None:
                good = False
                break
            acc = acc * 4 + value
        if good:
            found.add(acc)
    return found


def jaccard(seq_a: str, seq_b: str, k: int = 6) -> float:
    left = kmer_set(seq_a, k)
    right = kmer_set(seq_b, k)
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def hamming_identity(seq_a: str, seq_b: str) -> float | None:
    if len(seq_a) != len(seq_b) or not seq_a:
        return None
    same = sum(char_a == char_b for char_a, char_b in zip(seq_a, seq_b, strict=True))
    return same / len(seq_a)


@dataclass
class LeakageReport:
    n_rows: int
    exact_cross_split: list[dict] = field(default_factory=list)
    rc_cross_split: list[dict] = field(default_factory=list)
    group_cross_split: list[dict] = field(default_factory=list)
    near_cross_split: list[dict] = field(default_factory=list)
    scaffold_cross_split: list[dict] = field(default_factory=list)
    inchikey_cross_split: list[dict] = field(default_factory=list)
    threshold: float = 0.8
    k: int = 6
    notes: list[str] = field(default_factory=lambda: [NEAR_DUPLICATE_RULE])

    @property
    def passed(self) -> bool:
        return not (
            self.exact_cross_split
            or self.rc_cross_split
            or self.group_cross_split
            or self.near_cross_split
            or self.scaffold_cross_split
            or self.inchikey_cross_split
        )

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_rows": self.n_rows,
            "threshold": self.threshold,
            "k": self.k,
            "exact_cross_split": self.exact_cross_split,
            "rc_cross_split": self.rc_cross_split,
            "group_cross_split": self.group_cross_split,
            "near_cross_split": self.near_cross_split,
            "scaffold_cross_split": self.scaffold_cross_split,
            "inchikey_cross_split": self.inchikey_cross_split,
            "notes": self.notes,
        }


def _pair(left: int, right: int, **extra) -> dict:
    a, b = sorted((int(left), int(right)))
    payload = {"i": a, "j": b}
    payload.update(extra)
    return payload


def scan_leakage(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    split_col: str = "split",
    group_col: str = "group_id",
    threshold: float = 0.8,
    k: int = 6,
    hamming_identity_min: float = 0.95,
    exhaustive_limit: int = 400,
) -> LeakageReport:
    report = LeakageReport(n_rows=int(len(df)), threshold=threshold, k=k)
    if split_col not in df.columns:
        report.notes.append("no split column; leakage scan skipped")
        return report
    splits = [canonicalize_split(value) for value in df[split_col].tolist()]
    if group_col in df.columns:
        grouped: dict[str, set[str]] = {}
        for group_id, split in zip(df[group_col].astype(str), splits, strict=False):
            if group_id in {"", "nan", "None"}:
                continue
            grouped.setdefault(group_id, set()).add(split)
        for group_id, seen in sorted(grouped.items()):
            if len(seen) > 1:
                report.group_cross_split.append({"group_id": group_id, "splits": sorted(seen)})
    if sequence_col in df.columns:
        sequences = []
        hashes = []
        for raw in df[sequence_col].tolist():
            try:
                text = normalize_sequence(raw)
            except ValueError:
                text = ""
            sequences.append(text)
            hashes.append(sequence_hash(text) if text else "")
        by_hash: dict[str, list[tuple[int, str]]] = {}
        for index, (digest, split) in enumerate(zip(hashes, splits, strict=False)):
            if digest:
                by_hash.setdefault(digest, []).append((index, split))
        seen_exact = set()
        for digest, rows in by_hash.items():
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    i, split_i = rows[left]
                    j, split_j = rows[right]
                    if split_i != split_j:
                        key = (i, j, "exact")
                        if key not in seen_exact:
                            seen_exact.add(key)
                            report.exact_cross_split.append(
                                _pair(i, j, sequence_hash=digest, splits=[split_i, split_j])
                            )
        seen_rc = set()
        for index, text in enumerate(sequences):
            if not text:
                continue
            digest = sequence_hash(reverse_complement(text))
            for other, split_other in by_hash.get(digest, []):
                if other == index or splits[index] == split_other:
                    continue
                key = tuple(sorted((index, other)))
                if key in seen_rc:
                    continue
                seen_rc.add(key)
                report.rc_cross_split.append(_pair(index, other, splits=[splits[index], split_other]))
        pairs = [
            (i, j)
            for i in range(len(sequences))
            for j in range(i + 1, len(sequences))
            if sequences[i] and sequences[j] and splits[i] != splits[j]
        ]
        if len(sequences) > exhaustive_limit:
            pairs = _minhash_candidates(sequences, splits, k=k, limit=exhaustive_limit)
        for i, j in pairs:
            score = jaccard(sequences[i], sequences[j], k=k)
            identity = hamming_identity(sequences[i], sequences[j])
            if score >= threshold or (identity is not None and identity >= hamming_identity_min):
                report.near_cross_split.append(
                    _pair(i, j, jaccard=round(score, 4), hamming_identity=identity, splits=[splits[i], splits[j]])
                )
    _scan_id_column(df, splits, "scaffold_id", report.scaffold_cross_split, ignore_prefixes=("acyclic:",))
    _scan_id_column(df, splits, "inchikey", report.inchikey_cross_split, ignore_prefixes=())
    return report


def _scan_id_column(df, splits, column, sink, ignore_prefixes):
    if column not in df.columns:
        return
    buckets: dict[str, set[str]] = {}
    for value, split in zip(df[column].tolist(), splits, strict=False):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value)
        if not text or any(text.startswith(prefix) for prefix in ignore_prefixes):
            continue
        buckets.setdefault(text, set()).add(split)
    for value, seen in sorted(buckets.items()):
        if len(seen) > 1:
            sink.append({column: value, "splits": sorted(seen)})


def _minhash_candidates(sequences, splits, k: int, limit: int) -> list[tuple[int, int]]:
    """Banded MinHash proposes pairs; Jaccard is recomputed by the caller."""

    del limit
    n_perm = 16
    bands = 4
    rows = n_perm // bands
    signatures = []
    for seq in sequences:
        kmers = sorted(kmer_set(seq, k)) or [0]
        signature = []
        for perm in range(n_perm):
            best = None
            for token in kmers:
                mixed = (token * (1 + perm * 17) + perm * 97) % 1_000_003
                if best is None or mixed < best:
                    best = mixed
            signature.append(int(best))
        signatures.append(signature)
    buckets: dict[tuple, list[int]] = {}
    for index, signature in enumerate(signatures):
        for band in range(bands):
            key = (band, tuple(signature[band * rows : (band + 1) * rows]))
            buckets.setdefault(key, []).append(index)
    pairs = set()
    for indexes in buckets.values():
        if len(indexes) > 40:
            continue
        for left in range(len(indexes)):
            for right in range(left + 1, len(indexes)):
                i, j = indexes[left], indexes[right]
                if splits[i] != splits[j]:
                    pairs.add(tuple(sorted((i, j))))
    return sorted(pairs)


def render_leakage_markdown(report: LeakageReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "# Leakage audit",
        "",
        f"Status: **{status}**",
        "",
        NEAR_DUPLICATE_RULE,
        "",
        f"- rows: {report.n_rows}",
        f"- exact cross-split: {len(report.exact_cross_split)}",
        f"- reverse-complement cross-split: {len(report.rc_cross_split)}",
        f"- group cross-split: {len(report.group_cross_split)}",
        f"- near cross-split: {len(report.near_cross_split)}",
        f"- scaffold cross-split: {len(report.scaffold_cross_split)}",
        f"- inchikey cross-split: {len(report.inchikey_cross_split)}",
        "",
        "Within-split variants are not failures. Only cross-split hits fail the gate.",
    ]
    return "\n".join(lines) + "\n"

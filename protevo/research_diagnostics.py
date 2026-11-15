"""Scientific views computed from public predictions and already known measurements.

Epistasis is the additive contrast on raw WT-normalized fitness. Its matched
backgrounds are drawn from the current known table. Tree dispersion is model
disagreement (sample SD across trees), with no confidence-interval claim.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .data import full_sequence, validate_reference
from .schema import AMINO_ACIDS, POSITIONS

KINDS = ("heatmap", "scatter", "epistasis", "diversity", "structure")


def scatter(state):
    points = [{"round": record["round"], "variant": row["variant"],
               "predicted_fitness": row["predicted_fitness"], "actual_fitness": row["actual_fitness"]}
              for record in state["rounds"] for row in record["recommendations"]]
    paired = [p for p in points if p["predicted_fitness"] is not None and p["actual_fitness"] is not None]
    x = [p["predicted_fitness"] for p in paired]
    y = [p["actual_fitness"] for p in paired]
    correlation = (float(spearmanr(x, y).statistic)
                   if len(paired) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0 else None)
    return {"points": points, "sampleCount": len(paired), "spearman": correlation,
            "scope": "Previously selected measurements with their pre-query predictions"}


def epistasis(state):
    known = {row["variant"]: float(row["fitness"]) for row in state["known"]}
    pairs = []
    for i, j in combinations(range(4), 2):
        contrasts = []
        for variant, actual in known.items():
            if variant[i] == "VDGV"[i] or variant[j] == "VDGV"[j]:
                continue
            left, right, base = list(variant), list(variant), list(variant)
            left[j] = "VDGV"[j]
            right[i] = "VDGV"[i]
            base[i], base[j] = "VDGV"[i], "VDGV"[j]
            a, b, zero = "".join(left), "".join(right), "".join(base)
            if all(v in known for v in (a, b, zero)):
                contrasts.append(actual - known[a] - known[b] + known[zero])
        pairs.append({"positions": [POSITIONS[i], POSITIONS[j]], "sampleCount": len(contrasts),
                      "meanEpistasis": float(np.mean(contrasts)) if contrasts else None})
    return {"pairs": pairs, "definition": "epsilon = f(ab) - f(a0) - f(0b) + f(00)",
            "background": "matched known backgrounds; the other two residues are held fixed",
            "siteReference": "VDGV", "scale": "raw WT-normalized fitness",
            "sampleCountDefinition": "distinct measured double-substitution contrasts with all four measurements"}


def diversity(state):
    variants = list(state["scores"])
    n = len(variants)
    pair_count = n * (n - 1) // 2
    differences = 0
    counts = []
    for i, position in enumerate(POSITIONS):
        residues = Counter(v[i] for v in variants)
        differences += pair_count - sum(c * (c - 1) // 2 for c in residues.values())
        counts.extend({"position": position, "residue": aa, "count": residues[aa]} for aa in AMINO_ACIDS)
    return {"sampleCount": n, "meanPairwiseHamming": differences / pair_count if pair_count else None,
            "residueCounts": counts, "scope": "Distinct candidates scored in the current round",
            "distanceScale": "Number of differing residues among the four GB1 sites (0 to 4)"}


def structure(reference, allowed_positions):
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import protein_letters_3to1

    validate_reference(reference)
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "1PGA.pdb"
    model = next(PDBParser(QUIET=True).get_structure("1PGA", str(path)).get_models())
    chain = model["A"]
    rows = [r for r in chain if r.id[0] == " "]
    if [r.id for r in rows] != [(" ", i, " ") for i in range(1, 57)]:
        raise ValueError("1PGA requires 56 standard residues without numbering insertions")
    sequence = "".join(protein_letters_3to1[r.resname] for r in rows)
    differences = [{"position": i, "structureResidue": a, "experimentResidue": b}
                   for i, (a, b) in enumerate(zip(sequence, reference), 1) if a != b]
    if differences != [{"position": 2, "structureResidue": "T", "experimentResidue": "Q"}]:
        raise ValueError("1PGA requires the documented T2/Q2 reference difference")
    residues = []
    for position, residue in enumerate(rows, 1):
        xyz = [float(v) for v in residue["CA"].coord]
        if not np.isfinite(xyz).all():
            raise ValueError("1PGA C-alpha coordinates must be finite")
        residues.append({"position": position, "residue": sequence[position - 1],
                         "x": xyz[0], "y": xyz[1], "z": xyz[2], "isMutable": position in allowed_positions})
    return {"pdbId": "1PGA", "chain": "A", "residues": residues, "referenceDifferences": differences,
            "units": "angstrom", "source": "https://www.rcsb.org/structure/1PGA",
            "scope": "Static reference C-alpha coordinates; no mutant conformation is predicted"}


def heatmap(state, background, predict):
    full_sequence(background, state["science"]["reference"])
    config = state["science"]["config"]
    positions, amino_acids = config["allowed_positions"], config["allowed_amino_acids"]
    cells = []
    for position in positions:
        i = POSITIONS.index(position)
        for aa in amino_acids:
            variant = background[:i] + aa + background[i + 1:]
            cells.append({"position": position, "aminoAcid": aa, "variant": variant})
    variants = list(dict.fromkeys([background] + [r["variant"] for r in cells]))
    cache = state.setdefault("diagnostic_cache", {})
    fresh = [v for v in variants if v not in cache]
    if len(cache) + len(fresh) > state["science"].get("diagnostic_budget", 77):
        raise ValueError("Heatmap exceeds the independent per-round diagnostic budget")
    if fresh:
        records = predict(state, fresh)
        if len(records) != len(fresh):
            raise ValueError("Diagnostic predictor returned the wrong number of records")
        cache.update(zip(fresh, records))
    baseline = cache[background]["predicted_fitness"]
    for row in cells:
        prediction = cache[row["variant"]]
        row.update(prediction)
        row["delta_from_background"] = (prediction["predicted_fitness"] - baseline
                                         if prediction["predicted_fitness"] is not None and baseline is not None else None)
    return {"background": background, "positions": positions, "aminoAcids": amino_acids,
            "cells": cells, "baselinePrediction": baseline, "uniqueVariantCount": len(variants),
            "scope": "Single-site model predictions in the explicitly specified background",
            "budgetScope": "Independent per-round diagnostic predictions; separate from candidate scoring"}

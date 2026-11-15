# variant_notation.py — 替换 + 插入/缺失（indel）的统一变体记谱解析与全长重构
#
# 记谱文法（HGVS 风格子集；分隔符支持逗号/分号/冒号/空白，与 parse_mutation_notation 一致）：
#   替换   V39D                 野生型残基 + 1-indexed 位点 + 突变残基
#   缺失   V39del               单残基缺失；V39_G41del 范围缺失
#   插入   V39_G40insAA         在 39 与 40 之间插入 AA（HGVS 要求两锚点相邻）
#   重复   V39dup               单残基重复；V39_G41dup 范围重复（等价于在其后插入该段拷贝）
#   野生型 ":" / "WT" / 空串    无突变
#
# 坐标一律为 WT 参考系的 1-indexed 蛋白编号。indel 记谱离开全长 WT 无定义
# （删除后下游位点漂移、插入需要锚点上下文），因此重构/校验函数均要求显式传入 wt。
# 与 parse_mutation_notation 的关系：本模块是其超集（含 indel token），替换行为一致；
# 旧函数保持不动，供替换库路径继续使用。
import re

import numpy as np

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

_SUB = re.compile(r"^([A-Z])(\d+)([A-Z*])$")
_DEL = re.compile(r"^([A-Z])(\d+)(?:_([A-Z])(\d+))?del$")
_INS = re.compile(r"^([A-Z])(\d+)_([A-Z])(\d+)ins([A-Z]+)$")
_DUP = re.compile(r"^([A-Z])(\d+)(?:_([A-Z])(\d+))?dup$")


def parse_variant(variant: str) -> list:
    """解析含 indel 的变体记谱，返回突变字典列表。

    替换 {"type":"sub","wt","pos","mt"}  缺失 {"type":"del","start","end"}
    插入 {"type":"ins","after","seq"}    重复 {"type":"dup","start","end"}
    野生型（":"/"WT"/空串）→ []。token 无法识别时抛 ValueError。
    """
    variant = str(variant).strip()
    if variant in (":", "", "WT", "wt"):
        return []
    mutations = []
    for token in re.split(r"[:,;\s]+", variant):
        if not token:
            continue
        if (m := _SUB.match(token)):
            mutations.append({"type": "sub", "wt": m.group(1), "pos": int(m.group(2)), "mt": m.group(3)})
        elif (m := _DEL.match(token)):
            start = int(m.group(2))
            end = int(m.group(4)) if m.group(4) else start
            mutations.append({"type": "del", "start": start, "end": end})
        elif (m := _INS.match(token)):
            mutations.append({"type": "ins", "after": int(m.group(2)), "seq": m.group(5)})
        elif (m := _DUP.match(token)):
            start = int(m.group(2))
            end = int(m.group(4)) if m.group(4) else start
            mutations.append({"type": "dup", "start": start, "end": end})
        else:
            raise ValueError(f"无法解析变体记谱 token: {token!r}（期望 V39D / V39del / V39_G40insAA / V39dup）")
    return mutations


def is_indel_token(token: str) -> bool:
    """token 是否为 indel 记谱（del/ins/dup）。"""
    return bool(_DEL.match(token) or _INS.match(token) or _DUP.match(token))


def has_indel(variant: str) -> bool:
    """变体是否含 indel 突变。"""
    variant = str(variant).strip()
    if variant in (":", "", "WT", "wt"):
        return False
    return any(is_indel_token(t) for t in re.split(r"[:,;\s]+", variant) if t)


def _touched_positions(mutations: list) -> dict:
    """每条突变触及的 WT 位点集合（重叠检测用）。

    替换/缺失触及自身跨度；插入的锚点夹在 after 与 after+1 之间，两残基都触及；
    重复在其原始跨度之外还要求锚点残基完好，触及 [start, end+1]。
    """
    touched = {}
    for i, m in enumerate(mutations):
        if m["type"] == "sub":
            span = {m["pos"]}
        elif m["type"] == "del":
            span = set(range(m["start"], m["end"] + 1))
        elif m["type"] == "ins":
            span = {m["after"], m["after"] + 1}
        else:  # dup
            span = set(range(m["start"], m["end"] + 2))
        touched[i] = span
    return touched


def validate_variant(variant: str, wt: str, alphabet: str = AA_ALPHABET) -> str:
    """校验变体记谱对给定 WT 是否合法。合法返回 None，否则返回错误说明（英文，与 tools 层风格一致）。"""
    try:
        mutations = parse_variant(variant)
    except ValueError as exc:
        return str(exc)
    n = len(wt)
    for m in mutations:
        if m["type"] == "sub":
            if not (1 <= m["pos"] <= n):
                return f"position {m['pos']} is out of bounds for reference of length {n}"
            if wt[m["pos"] - 1] != m["wt"]:
                return f"wild-type letter {m['wt']} does not match reference residue {wt[m['pos']-1]} at position {m['pos']}"
            if m["mt"] not in alphabet:
                return f"residue {m['mt']} must come from the amino-acid alphabet {alphabet}"
        elif m["type"] == "del":
            if not (1 <= m["start"] <= m["end"] <= n):
                return f"deletion range {m['start']}-{m['end']} is out of bounds for reference of length {n}"
        elif m["type"] == "ins":
            if not (1 <= m["after"] <= n - 1):
                return f"insertion anchor {m['after']} must lie strictly inside the reference (length {n})"
        else:  # dup
            if not (1 <= m["start"] <= m["end"] <= n):
                return f"duplication range {m['start']}-{m['end']} is out of bounds for reference of length {n}"
    # 锚点字母核对（插入/缺失/重复的 WT 字母须与参考序列一致）
    for token in re.split(r"[:,;\s]+", str(variant).strip()):
        if not token:
            continue
        if (m := _INS.match(token)):
            a, b = int(m.group(2)), int(m.group(4))
            if b != a + 1:
                return f"insertion anchors must be adjacent positions, got {a} and {b}"
            if not (1 <= a <= n - 1) or wt[a - 1] != m.group(1) or wt[b - 1] != m.group(3):
                return f"insertion anchors {m.group(1)}{a}/{m.group(3)}{b} do not match the reference sequence"
        elif (m := _DEL.match(token)):
            a, b = int(m.group(2)), int(m.group(4) or m.group(2))
            letter_b = m.group(3) or m.group(1)
            if wt[a - 1] != m.group(1) or wt[b - 1] != letter_b:
                return f"deletion letters {m.group(1)}{a}/{letter_b}{b} do not match the reference sequence"
        elif (m := _DUP.match(token)):
            a, b = int(m.group(2)), int(m.group(4) or m.group(2))
            letter_b = m.group(3) or m.group(1)
            if wt[a - 1] != m.group(1) or wt[b - 1] != letter_b:
                return f"duplication letters {m.group(1)}{a}/{letter_b}{b} do not match the reference sequence"
    # 重叠检测：任何两条突变共享触及位点即拒绝（插入顺序会有歧义）
    touched = _touched_positions(mutations)
    spans = list(touched.values())
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if spans[i] & spans[j]:
                return f"mutations {i+1} and {j+1} touch overlapping reference positions"
    return None


def apply_mutations(wt: str, mutations: list) -> str:
    """在 WT 坐标系上应用突变列表，返回变体全长序列。

    输入应为 parse_variant 的输出且已通过 validate_variant；函数内部仍做重叠防御。
    """
    touched = _touched_positions(mutations)
    spans = list(touched.values())
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if spans[i] & spans[j]:
                raise ValueError(f"mutations {i+1} and {j+1} touch overlapping reference positions")
    ops = []  # (主位点, 种类, 载荷)：ins 锚定于 after（含该残基后插入）
    for m in mutations:
        if m["type"] == "sub":
            ops.append((m["pos"], "sub", m["mt"]))
        elif m["type"] == "del":
            ops.append((m["start"], m["end"], "del"))
        elif m["type"] == "ins":
            ops.append((m["after"], None, "ins:" + m["seq"]))
        else:  # dup → 在跨度末位之后插入该段拷贝
            ops.append((m["end"], None, "ins:" + wt[m["start"] - 1:m["end"]]))
    ops.sort(key=lambda op: op[0])
    out, cursor = [], 1
    for primary, end, payload in ops:
        if payload == "del":
            out.append(wt[cursor - 1:primary - 1])
            cursor = end + 1
        elif isinstance(payload, str) and payload.startswith("ins:"):
            out.append(wt[cursor - 1:primary - 1])
            out.append(wt[primary - 1])          # 锚点残基
            out.append(payload[4:])              # 插入段
            cursor = primary + 1
        else:                                     # sub
            out.append(wt[cursor - 1:primary - 1])
            out.append(payload)
            cursor = primary + 1
    out.append(wt[cursor - 1:])
    return "".join(out)


def reconstruct(variant: str, wt: str) -> str:
    """记谱 → 变体全长序列（解析 + 重构一步完成；不合法时抛 ValueError）。"""
    return apply_mutations(wt, parse_variant(variant))


def composition_features(sequences: list, alphabet: str = AA_ALPHABET) -> np.ndarray:
    """变长序列的长度不变替换物：20 维氨基酸组分占比 + 1 维长度（indel 库的 CPU 基线特征）。"""
    mat = np.zeros((len(sequences), len(alphabet) + 1))
    for i, seq in enumerate(sequences):
        for aa in seq:
            idx = alphabet.find(aa)
            if idx >= 0:
                mat[i, idx] += 1
        mat[i, :len(alphabet)] /= max(len(seq), 1)
        mat[i, -1] = len(seq)
    return mat

"""variant_notation 解析/校验/重构/组分特征的单元测试（合成序列 WT=ACDEFV，无网络无 GPU）。"""
import numpy as np
import pytest

from protevo.variant_notation import (
    apply_mutations,
    composition_features,
    has_indel,
    is_indel_token,
    parse_variant,
    reconstruct,
    validate_variant,
)

WT = "ACDEFV"  # 位点 1-6：A1 C2 D3 E4 F5 V6，全部为标准氨基酸字母


# ---------- 解析 ----------

def test_parse_token_types():
    assert parse_variant("C2X") == [{"type": "sub", "wt": "C", "pos": 2, "mt": "X"}]
    assert parse_variant("C2del") == [{"type": "del", "start": 2, "end": 2}]
    assert parse_variant("C2_D3del") == [{"type": "del", "start": 2, "end": 3}]
    assert parse_variant("C2_D3insXY") == [{"type": "ins", "after": 2, "seq": "XY"}]
    assert parse_variant("D3dup") == [{"type": "dup", "start": 3, "end": 3}]
    assert parse_variant("C2_D3dup") == [{"type": "dup", "start": 2, "end": 3}]
    assert parse_variant(":") == [] and parse_variant("WT") == []
    # 混合替换 + indel，分隔符混用（含 ProteinGym 冒号）
    muts = parse_variant("C2X:D3dup,E4del")
    assert [m["type"] for m in muts] == ["sub", "dup", "del"]
    with pytest.raises(ValueError, match="无法解析"):
        parse_variant("C2dell")
    with pytest.raises(ValueError, match="无法解析"):
        parse_variant("39del")  # 缺野生型字母


def test_has_indel():
    assert has_indel("C2del") and has_indel("A1X,C2_D3insG") and has_indel("D3dup")
    assert not has_indel("A1X,C2D") and not has_indel(":") and not has_indel("WT")
    assert is_indel_token("V39del") and is_indel_token("V39_G40insAA") and not is_indel_token("V39D")


# ---------- 重构（坐标数学核心）----------

def test_apply_sub_and_single_del():
    assert apply_mutations(WT, parse_variant("C2X")) == "AXDEFV"   # 替换长度不变
    assert apply_mutations(WT, parse_variant("C2del")) == "ADEFV"

def test_apply_range_del_and_ins_and_dup():
    assert apply_mutations(WT, parse_variant("C2_D3del")) == "AEFV"
    assert apply_mutations(WT, parse_variant("C2_D3insXY")) == "ACXYDEFV"
    assert apply_mutations(WT, parse_variant("D3dup")) == "ACDDEFV"
    assert apply_mutations(WT, parse_variant("C2_D3dup")) == "ACDCDEFV"

def test_apply_combined():
    # 替换 + 缺失 + 插入，各在不相交区域
    assert apply_mutations(WT, parse_variant("A1X,C2del,F5_V6insGG")) == "XDEFGGV"

def test_apply_wt_returns_reference():
    assert reconstruct(":", WT) == WT and reconstruct("WT", WT) == WT

def test_reconstruct_length_effects():
    assert len(reconstruct("C2del", WT)) == 5        # -1
    assert len(reconstruct("C2_D3insAA", WT)) == 8   # +2
    assert len(reconstruct("D3dup", WT)) == 7        # +1

def test_apply_rejects_overlaps():
    with pytest.raises(ValueError, match="overlapping"):
        apply_mutations(WT, parse_variant("C2X,C2del"))            # 同位点替换+缺失
    with pytest.raises(ValueError, match="overlapping"):
        apply_mutations(WT, parse_variant("C2_D3insX,D3_E4insY"))  # 同锚点双插入
    with pytest.raises(ValueError, match="overlapping"):
        apply_mutations(WT, parse_variant("C2_D3del,D3del"))       # 缺失跨度与单缺失重叠


# ---------- 校验 ----------

def test_validate_legal_returns_none():
    for v in ["C2D", "C2del", "C2_D3del", "C2_D3insXY", "D3dup", "A1D:D3dup", ":", "WT"]:
        assert validate_variant(v, WT) is None, v

def test_validate_bounds_and_letters():
    assert "out of bounds" in validate_variant("C7D", WT)
    assert "do not match the reference" in validate_variant("X2del", WT)      # WT 字母不符
    assert "do not match the reference" in validate_variant("C2_X3insA", WT)  # 插入第二锚点字母不符
    assert "adjacent" in validate_variant("C2_E4insA", WT)                    # 插入锚点不相邻
    assert "must lie strictly inside" in validate_variant("V6_G7insA", WT)    # C 端外插入
    assert "out of bounds" in validate_variant("A1_C9del", WT)

def test_validate_alphabet_and_overlap():
    assert "must come from" in validate_variant("C2Z", WT)     # Z 不在字母表
    assert "overlapping" in validate_variant("C2D,C2del", WT)


# ---------- 组分特征 ----------

def test_composition_features():
    mat = composition_features(["AAAC", "DE"])
    assert mat.shape == (2, 21)
    assert mat[0, 0] == pytest.approx(0.75)      # A 占比
    assert mat[0, -1] == 4 and mat[1, -1] == 2   # 长度列
    assert mat[1, 2] == pytest.approx(0.5) and mat[1, 3] == pytest.approx(0.5)  # D=idx2, E=idx3

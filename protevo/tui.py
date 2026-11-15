# tui.py — 命令行交互界面（TUI）：交互式定向进化工作台（最简实现）
# 范式参考 OpenClaw：本地轻量、一条命令启动（python -m protevo.tui）、命令式交互、最小依赖。
"""Interactive command-line workbench: open → run → plot → export, plus first-run LLM setup wizard.

stdlib cmd.Cmd 实现（零新交互依赖）；全部功能经 protevo 公开 API 委托完成：
configure_protein / run_evolution(on_round) / setup_assistant（probe_environment /
probe_data / build_plan / run_doctor / configure_with_llm / render_doctor）。
非 tty 输出自动关闭 ANSI 颜色；stdin 非 tty 时首次引导不阻塞（打印说明直接进 REPL）。
"""
import cmd as _cmd
import getpass
import json
import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .evolve import _detect_columns, run_evolution
from .protein_config import configure_protein, parse_mutation_notation
from .setup_assistant import (
    _doctor_autofix,
    build_plan,
    configure_with_llm,
    probe_data,
    probe_environment,
    render_doctor,
    run_doctor,
)
from .variant_notation import reconstruct

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

# ---------------------------------------------------------------------------
# ANSI 颜色（stdout 非 tty 自动关闭）
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()


def _paint(code, text):
    return f"\x1b[{code}m{text}\x1b[0m" if _USE_COLOR else str(text)


def dim(text):
    return _paint("2", text)


def bold(text):
    return _paint("1", text)


def green(text):
    return _paint("32", text)


def yellow(text):
    return _paint("33", text)


def red(text):
    return _paint("31", text)


# ---------------------------------------------------------------------------
# 内置数据集（gb1 + ProteinGym 三件套，与 run_cross_protein.DATASETS 同源）
# ---------------------------------------------------------------------------
_BUILTIN_DATASETS = {
    "gb1": "GB1 四位点替换枚举库（Wu 2016）",
    "ubiq": "泛素 RL40A 单替换（Roscoe 2014）",
    "gfp": "GFP 替换（Sarkisyan 2016）",
    "blat": "TEM-1 BLAT indel（Gonzalez 2019）",
}
_GB1_CSV = PROJECT_ROOT / "data/processed/all_variants.csv"


def _builtin_entry(name):
    """内置名 → {label, path, ref, hib, rows}；目录型数据集从 meta.json 取 WT/方向/行数。"""
    if name == "gb1":
        n_rows = sum(1 for _ in _GB1_CSV.open("rb")) - 1
        return {"label": _BUILTIN_DATASETS["gb1"], "path": _GB1_CSV, "ref": "VDGV", "hib": True,
                "rows": n_rows}
    directory = PROJECT_ROOT / "data/external" / {
        "ubiq": "ubiq_roscoe_2014", "gfp": "gfp_sarkisyan", "blat": "blat_ecolx_gonzalez_2019"}[name]
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    return {"label": _BUILTIN_DATASETS[name], "path": directory / "dms.csv",
            "ref": meta.get("wt_sequence"), "hib": bool(meta.get("higher_is_better", True)),
            "rows": int(meta.get("n_variants", 0))}


# ---------------------------------------------------------------------------
# 凭据检测与 .env 写入（约定同 agent_v2.build_openai_client：包根 .env 三变量）
# ---------------------------------------------------------------------------
def _load_env(env_path=ENV_PATH):
    """把包根 .env 载入 os.environ（override=False，不覆盖已存在的真实环境变量）。"""
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path(env_path), override=False)
    except ImportError:
        pass


def credentials_complete(env_path=ENV_PATH):
    """OPENAI_API_KEY 与 OPENAI_MODEL 两变量均非空 → True（先做一次 .env 载入）。"""
    _load_env(env_path)
    return bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("OPENAI_MODEL"))


def write_env_updates(env_path, updates):
    """按 updates 增量写 .env：管理键已有行原位更新（重复行清理），没有则追加；值 None 表示
    删除该键全部行；其余行（注释/无关键）逐字保留。文件不存在则创建。返回写入路径。"""
    env_path = Path(env_path)
    lines = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.exists() else []
    seen, out = set(), []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in updates:
            if key in seen:  # 同一管理键的重复旧行一并清理，保证幂等
                continue
            seen.add(key)
            if updates[key] is not None:
                out.append(f"{key}={updates[key]}")
        else:
            out.append(line)
    out.extend(f"{key}={value}" for key, value in updates.items()
               if key not in seen and value is not None)
    env_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return env_path


# ---------------------------------------------------------------------------
# 首次启动引导（交互式三步向导；stdin 非 tty 时不阻塞，只打印说明）
# ---------------------------------------------------------------------------
def _mask(secret):
    return f"{secret[:6]}…{secret[-4:]}" if len(secret) > 12 else "****"


def run_setup_wizard(env_path=ENV_PATH):
    """三步配好 Base_URL / API Key / Model 并写入 .env；任一步可 skip，全部 skip 则不动文件。"""
    print(bold("── 首次启动设置：三步配好 LLM 凭据 ──"))
    print(dim("不配置也能用预测器/复测功能（datasets / open / predict / plot / export 全离线），"
              "仅 run（Agent 进化）与 ask（LLM 顾问）需要凭据。任一步输入 skip 跳过。"))
    updates, endpoint = {}, "unknown"

    # 1) Base URL：菜单选择
    print("1) Base URL（OpenAI 兼容端点）")
    print("   1. OpenAI 官方（不设置 OPENAI_BASE_URL）")
    print("   2. DeepSeek（预填 https://api.deepseek.com）")
    print("   3. 自定义（手输；支持 hf-mirror 类兼容端点）")
    while True:
        choice = (input("选择 [1]: ") or "1").strip()
        if choice.lower() == "skip":
            break
        if choice == "1":
            updates["OPENAI_BASE_URL"], endpoint = None, "official"
            break
        if choice == "2":
            updates["OPENAI_BASE_URL"], endpoint = "https://api.deepseek.com", "deepseek"
            break
        if choice == "3":
            url = input("Base URL: ").strip()
            if url.lower() == "skip":
                break
            if url:
                updates["OPENAI_BASE_URL"], endpoint = url, "custom"
                break
            print(red("  ✗ 不能为空（或输入 skip 跳过）"))
        else:
            print(red("  ✗ 请输入 1 / 2 / 3 / skip"))

    # 2) API Key：getpass 静默输入，回车非空校验
    while True:
        key = getpass.getpass("2) API Key（输入时隐藏）: ").strip()
        if key.lower() == "skip":
            break
        if key:
            updates["OPENAI_API_KEY"] = key
            break
        print(red("  ✗ 不能为空（或输入 skip 跳过）"))

    # 3) 模型名：按端点给建议默认值
    suggestion = {"official": "gpt-4o-mini", "deepseek": "deepseek-chat"}.get(endpoint)
    hint = f"（回车采用 {suggestion}）" if suggestion else "（该端点无默认建议，请手输）"
    while True:
        name = input(f"3) 模型名{hint}: ").strip()
        if name.lower() == "skip":
            break
        if name or suggestion:
            updates["OPENAI_MODEL"] = name or suggestion
            break
        print(red("  ✗ 不能为空（或输入 skip 跳过）"))

    if not updates:
        print(dim("全部跳过：未修改 .env（当前会话预测器功能不受影响）。"))
        return
    print(bold("── 汇总确认 ──"))
    if "OPENAI_BASE_URL" in updates:
        value = updates["OPENAI_BASE_URL"]
        print(f"  OPENAI_BASE_URL = {value if value else '（清除，走官方端点）'}")
    if "OPENAI_API_KEY" in updates:
        print(f"  OPENAI_API_KEY  = {_mask(updates['OPENAI_API_KEY'])}")
    if "OPENAI_MODEL" in updates:
        print(f"  OPENAI_MODEL    = {updates['OPENAI_MODEL']}")
    if (input("写入 .env？[y/N]: ").strip().lower() or "n") != "y":
        print(dim("已取消，未写入。"))
        return
    path = write_env_updates(env_path, updates)
    print(green(f"✅ 已写入 {path}（其他行保持不变）"))
    print(dim("复核建议：在 REPL 中运行 doctor，查看其中 llm_credentials 检查项。"))


# ---------------------------------------------------------------------------
# matplotlib（缺失时优雅降级：只出 ASCII + 安装提示）
# ---------------------------------------------------------------------------
def _matplotlib_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# 预测器短名（提示符与卡片用；matplotlib 图内文字用 ASCII，避免中文字体缺失）
_PREDICTOR_SHORT = {
    "random_forest": "RF", "ridge": "Ridge", "onehot_mlp": "MLP",
    "esm2_ridge": "ESM-Ridge", "esm2_mlp": "ESM-MLP",
    "composition_ridge": "CompRidge", "composition_mlp": "CompMLP",
    "random_forest_default": "RF",
}


class ProtevoShell(_cmd.Cmd):
    """命令式交互台：数据与配置 / 实验 / 可视化 / 回放与导出 四组命令（对照原 GUI 四面板）。"""

    def __init__(self, env_path=ENV_PATH, stdin=None, stdout=None):
        super().__init__(stdin=stdin, stdout=stdout)
        self.env_path = Path(env_path)
        self.doc_header = "命令（help <命令> 查看详细用法）"
        # 会话状态
        self.config = None          # configure_protein 产物 ProteinConfig
        self.dataset_label = None   # 内置名或 CSV 文件名（提示符显示）
        self.data_path = None
        self.n_rows = 0
        self.use_esm = False
        self.ref_used = None       # open 时采用的参考序列（None = 交给 configure_protein 推断）
        self.hib = True            # open 时采用的方向（higher_is_better）
        self.last_summary = None    # 最近一次 run_evolution 汇总
        self.last_out_dir = None
        self.model_override = None  # model 命令设置的会话级模型名
        self._client_for_test = None  # 测试注入口：(client, model) 二元组；None 走 .env
        self._refresh_prompt()

    # ---------------- 基础设施 ----------------
    def _refresh_prompt(self):
        if self.config is None:
            self.prompt = "protevo> "
            return
        info = self.config.predictor_info
        short = _PREDICTOR_SHORT.get(info.get("selected"), info.get("selected") or "?")
        rho = info.get("validation_spearman")
        rho_text = f" ρ={rho}" if rho is not None else ""
        self.prompt = f"protevo ({self.dataset_label}|{short}{rho_text})> "

    def postcmd(self, stop, line):
        self._refresh_prompt()
        return stop

    def emptyline(self):
        pass  # 空行不重复上一条命令

    def default(self, line):
        print(red(f"未知命令：{line.split()[0] if line.split() else line!r}（help 查看命令表）"))

    @staticmethod
    def _tokenize(arg):
        """Windows 友好分词：posix=False 保留反斜杠，再剥掉成对引号（--out "C:\\a b\\c.csv" 可用）。"""
        tokens = shlex.split(arg, posix=False)
        return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]

    @staticmethod
    def _parse_flags(tokens, valued, boolean=()):
        """手写轻量 flag 解析：返回 (位置参数, {flag: 值}, {flag})。"""
        positional, values, flags = [], {}, set()
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in valued:
                if i + 1 >= len(tokens):
                    raise ValueError(f"{token} 需要一个值")
                values[token] = tokens[i + 1]
                i += 2
            elif token in boolean:
                flags.add(token)
                i += 1
            else:
                positional.append(token)
                i += 1
        return positional, values, flags

    def _require_open(self, action):
        if self.config is None:
            print(yellow(f"请先 open 一个数据集再{action}：datasets 查看内置示例，"
                         "或 open <csv 路径> [--ref WT]。"))
            return False
        return True

    # ---------------- 数据与配置面板 ----------------
    def do_datasets(self, arg):
        """datasets — 列出 4 个内置示例数据集（open <名称> 直接使用）"""
        print(bold("── 内置数据集 ──"))
        for name, label in _BUILTIN_DATASETS.items():
            try:
                entry = _builtin_entry(name)
                print(f"  {name:5s} {label}\n        {entry['path'].relative_to(PROJECT_ROOT)}"
                      f"（{entry['rows']} 行，参考 {'自带' if entry['ref'] else '需补'}）")
            except (OSError, KeyError, ValueError) as exc:
                print(f"  {name:5s} {label}\n        （不可用：{exc}）")

    def do_open(self, arg):
        """open <csv|内置名> [--ref WT] [--esm] [--pool recommended|full] [--hrb higher|lower] — 配置当前数据集"""
        try:
            tokens = self._tokenize(arg)
        except ValueError as exc:
            print(red(f"参数解析失败：{exc}"))
            return
        try:
            positional, values, flags = self._parse_flags(
                tokens, valued=("--ref", "--pool", "--hrb"), boolean=("--esm",))
        except ValueError as exc:
            print(red(str(exc)))
            return
        if len(positional) != 1:
            print("用法：open <csv|内置名> [--ref WT] [--esm] [--pool recommended|full] [--hrb higher|lower]")
            return
        target = positional[0]

        ref, hib = None, True
        if target in _BUILTIN_DATASETS:
            entry = _builtin_entry(target)
            path, ref, hib = entry["path"], entry["ref"], entry["hib"]
            self.n_rows, label = entry["rows"], target
        else:
            path = Path(target)
            if not path.is_file():
                print(red(f"找不到数据文件：{target}（内置名见 datasets）"))
                return
            self.n_rows = sum(1 for _ in path.open("rb")) - 1
            label = path.stem
        if "--ref" in values:
            ref = values["--ref"]
        if "--hrb" in values:
            if values["--hrb"] not in ("higher", "lower"):
                print(red("--hrb 只接受 higher | lower"))
                return
            hib = values["--hrb"] == "higher"
        pool = values.get("--pool", "recommended")
        if pool not in ("recommended", "full"):
            print(red("--pool 只接受 recommended | full"))
            return

        print(dim(f"配置中：{path}（{self.n_rows} 行）……"))
        try:
            config = configure_protein(str(path), reference_sequence=ref,
                                       higher_is_better=hib, use_esm="--esm" in flags,
                                       predictor_pool=pool)
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(red(f"配置失败：{type(exc).__name__}: {exc}"))
            print(dim("排查建议：运行 plan（数据探测 + 规则计划）或 doctor。"))
            return
        self.config, self.dataset_label, self.data_path = config, label, path
        self.use_esm, self.ref_used, self.hib = "--esm" in flags, ref, hib
        self.last_summary, self.last_out_dir = None, None  # 换数据集后旧 run 不再关联
        self._refresh_prompt()  # onecmd 不走 postcmd，open 自行刷新提示符
        self._print_config_card()

    def _print_config_card(self):
        config, info = self.config, self.config.predictor_info
        fmt = config.variant_format
        positions = list(config.mutable_positions)
        # full / mutation_string 的位点是 0 基下标，展示按 1 基蛋白编号
        shown = [p + 1 if fmt in ("full", "mutation_string") else p for p in positions]
        span = f"{shown[0]}–{shown[-1]}" if shown else "无"
        ref_source = {"provided": "显式提供", "inferred": "众数推断"}.get(
            config.reference_source, config.reference_source)
        reference = f"{config.reference_sequence[:20]}{'…' if len(config.reference_sequence) > 20 else ''}"
        print(green("── 配置卡 ─────────────────────────────"))
        print(f"  数据:   {self.dataset_label} · {self.data_path}（{self.n_rows} 行）")
        print(f"  格式:   {fmt}（来源：{info.get('format_source', 'auto')}）")
        print(f"  位点:   {len(positions)} 个可变位点（{span}）")
        line = f"  参考:   {ref_source}（{reference}，{len(config.reference_sequence)}aa）"
        if config.reference_source == "inferred_uniform_unreliable":
            print(yellow(line))
            print(yellow("         ⚠ 均匀枚举库的众数推断不可靠，建议 open 时显式 --ref WT"))
        else:
            print(line)
        print(f"  方向:   {'higher is better' if config.fitness_higher_is_better else 'lower is better'}")
        print(f"  预测器: {info.get('selected')}（val ρ={info.get('validation_spearman')}）")
        candidates = info.get("all_candidates") or {}
        print("  候选:   " + (" | ".join(f"{n} {s}" for n, s in candidates.items()) or "（无选型记录）"))
        print(dim("  下一步：predict 预测 → run 进化 → plot / log / export。"))

    def do_info(self, arg):
        """info — 当前数据集配置卡 + 最近一次运行摘要"""
        if not self._require_open("查看 info"):
            return
        self._print_config_card()
        if self.last_summary is not None:
            summary = self.last_summary
            best = summary.get("cumulative_best") or {}
            print(f"  最近运行: {summary['out_dir']}"
                  f"（{len(summary['rounds'])} 轮，累计最优 {best.get('variant')}({best.get('fitness'):.4g})）")

    # ---------------- 实验面板 ----------------
    def do_run(self, arg):
        """run [--rounds 3] [--per-round 10] [--budget 40] [--retrain] — 多轮虚拟定向进化（每轮一行准实时摘要）"""
        if not self._require_open("运行实验"):
            return
        try:
            positional, values, flags = self._parse_flags(
                arg.split(), valued=("--rounds", "--per-round", "--budget"), boolean=("--retrain",))
        except ValueError as exc:
            print(red(str(exc)))
            return
        if positional:
            print(red(f"无法识别的参数：{positional}"))
            return
        try:
            rounds = int(values.get("--rounds", "3"))
            per_round = int(values.get("--per-round", "10"))
            budget = int(values.get("--budget", "40"))
        except ValueError:
            print(red("--rounds / --per-round / --budget 需为整数"))
            return
        if min(rounds, per_round, budget) < 1:
            print(red("轮数 / 每轮提交数 / 预算均需 ≥ 1"))
            return

        client, model = self._client_for_test if self._client_for_test else (None, None)

        def on_round(rs):
            mean = "n/a" if rs["measured_mean"] is None else f"{rs['measured_mean']:.4g}"
            best = rs["cumulative_best"]
            left = rs["max_score_budget"] - rs["score_calls_used"]
            print(green(f"▸ 第 {rs['round']}/{rounds} 轮：提交 {rs['submitted']} 条 | 回填均值 {mean} | "
                        f"累计最优 {best['variant']}({best['fitness']:.4g}) | "
                        f"预算余量 {left}/{rs['max_score_budget']}"))

        try:
            summary = run_evolution(
                self.data_path, rounds=rounds, per_round=per_round, score_budget=budget,
                # 小数据集自动收紧初始训练子集（默认 2000 超过行数会报错）
                initial_train=min(2000, max(1, self.n_rows // 2)),
                reference_sequence=self.ref_used, higher_is_better=self.hib,
                use_esm=self.use_esm, retrain_each_round="--retrain" in flags,
                client=client, model=model, on_round=on_round)
        except Exception as exc:  # MissingCredentials / ValueError 等：给出可操作的提示
            print(red(f"运行失败：{type(exc).__name__}: {exc}"))
            from .agent import MissingCredentials
            if isinstance(exc, MissingCredentials):
                print(yellow("缺少 LLM 凭据：先运行 setup 三步配置（或手动编辑包根 .env）。"))
            return
        self.last_summary, self.last_out_dir = summary, Path(summary["out_dir"])
        self._print_trajectory(summary)
        research = self._research_predictor_text()
        if research:
            print(green(f"  本次研究预测器: {research}"))
        print(dim(f"输出目录：{self.last_out_dir}（per_round.csv / summary.json / agent_round*.jsonl）"))

    @staticmethod
    def _print_trajectory(summary):
        print(bold("── 进化轨迹 ──"))
        print(f"  {'轮':<3}{'状态':<14}{'提交':<5}{'回填均值':<10}{'累计最佳':<26}评分/API")
        for rs in summary["rounds"]:
            mean = "n/a" if rs["measured_mean"] is None else f"{rs['measured_mean']:.4g}"
            best = rs["cumulative_best"]
            variant = best["variant"][:22] + "…" if len(best["variant"]) > 23 else best["variant"]
            print(f"  {rs['round']:<3}{(rs['status'] or '')[:12]:<14}{rs['submitted']:<5}{mean:<10}"
                  f"{variant}({best['fitness']:.4g})".ljust(26)
                  + f"{rs['score_calls_used']}/{rs['max_score_budget']} · {rs['api_calls_used']} 次")

    def _research_predictor_text(self):
        """本次 run 实际使用的研究预测器（读 last_summary 的 config.predictor，与落盘 summary.json 同源；
        run 会以初始训练子集重训选型，常与会话 open 时的 ρ 不同）；两者不同则并列展示。无运行记录返回 None。"""
        if self.last_summary is None:
            return None
        run = (self.last_summary.get("config") or {}).get("predictor") or {}
        if not run.get("selected"):
            return None
        run_rho = run.get("validation_spearman")
        text = f"{run['selected']}（val ρ={run_rho}）"
        session = self.config.predictor_info if self.config is not None else {}
        if (session.get("selected") != run.get("selected")
                or session.get("validation_spearman") != run_rho) and session.get("selected"):
            text += f" | 会话 open 时 {session.get('selected')}（val ρ={session.get('validation_spearman')}）"
        return text

    def do_model(self, arg):
        """model [名称] — 查看/设置本会话的 LLM 模型名（写入 os.environ，不改动 .env）"""
        name = arg.strip()
        if not name:
            current = os.environ.get("OPENAI_MODEL") or "(未设置)"
            print(f"当前模型：{current}" + (f"（会话内覆盖：{self.model_override}）" if self.model_override else ""))
            return
        os.environ["OPENAI_MODEL"] = name  # build_openai_client 读取 os.environ，会话内即时生效
        self.model_override = name
        print(green(f"✅ 本会话模型已设为 {name}（仅会话内生效；持久化请用 setup 写入 .env）"))

    def do_status(self, arg):
        """status — 会话总览：数据 / 预测器 / 凭据 / 最近运行"""
        print(bold("── 会话状态 ──"))
        if self.config is None:
            print("  数据:   未 open（datasets 查看内置示例）")
        else:
            info = self.config.predictor_info
            print(f"  数据:   {self.dataset_label} · {self.data_path}")
            print(f"  预测器: {info.get('selected')}（val ρ={info.get('validation_spearman')}，"
                  f"{len(self.config.mutable_positions)} 位点，方向 "
                  f"{'higher' if self.config.fitness_higher_is_better else 'lower'}）")
            research = self._research_predictor_text()
            if research:
                print(f"  研究模型: {research}（最近一次 run 实际使用）")
        print(f"  凭据:   {'已配置' if credentials_complete(self.env_path) else '缺失（setup 可三步配置）'}"
              f"{' · 会话模型 ' + self.model_override if self.model_override else ''}")
        if self.last_summary is not None:
            best = self.last_summary.get("cumulative_best") or {}
            print(f"  最近运行: {self.last_summary['out_dir']}（累计最优 "
                  f"{best.get('variant')}({best.get('fitness'):.4g})）")
        else:
            print("  最近运行: 无（run [--rounds N] 启动进化）")

    # ---------------- 可视化面板 ----------------
    def do_plot(self, arg):
        """plot <scatter|trajectory|heatmap|length|budget> — PNG 落输出目录 + 终端 ASCII 摘要"""
        kind = arg.strip().split()[0] if arg.strip() else ""
        kinds = ("scatter", "trajectory", "heatmap", "length", "budget")
        if kind not in kinds:
            print(f"用法：plot <{'|'.join(kinds)}>\n"
                  "  scatter/trajectory/budget 需先 run；heatmap 面向替换库、length 面向 indel 库。")
            return
        if kind in ("scatter", "trajectory", "budget") and self.last_out_dir is None:
            print(yellow("该图需要先 run 一次进化（数据来自 per_round.csv / summary.json）。"))
            return
        if kind in ("heatmap", "length") and not self._require_open("绘图"):
            return
        target = self.last_out_dir if self.last_out_dir is not None else Path("results/tui")
        target.mkdir(parents=True, exist_ok=True)
        try:
            ascii_line = getattr(self, f"_plot_{kind}")(target)
        except ValueError as exc:  # 格式不匹配等：给指引而非报错
            print(yellow(str(exc)))
            return
        except Exception as exc:
            print(red(f"绘图失败：{type(exc).__name__}: {exc}"))
            return
        print(ascii_line)

    def _plot_png(self, builder, target, name):
        """matplotlib 可用则出 PNG 返回路径，否则返回 None（调用方降级 ASCII + 提示）。"""
        try:
            plt = _matplotlib_pyplot()
        except ImportError:
            return None
        figure = builder(plt)
        png = Path(target) / f"{name}.png"
        figure.savefig(png, dpi=160)
        plt.close(figure)
        return png

    def _plot_trajectory(self, target):
        rounds = self.last_summary["rounds"]
        labels = [rs["round"] for rs in rounds]
        means = [rs["measured_mean"] for rs in rounds]
        bests = [rs["cumulative_best"]["fitness"] for rs in rounds]

        def build(plt):
            figure, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
            ax.plot(labels, [b for b in bests], marker="o", label="cumulative best")
            if any(m is not None for m in means):
                ax.plot(labels, [m if m is not None else float("nan") for m in means],
                        marker="s", label="mean measured / round")
            ax.set(xlabel="round", ylabel="fitness", title="Evolution trajectory", xticks=labels)
            ax.legend()
            ax.grid(alpha=0.25)
            return figure

        png = self._plot_png(build, target, "trajectory")
        trend = "→".join("n/a" if m is None else f"{m:.3g}" for m in means)
        best = self.last_summary["cumulative_best"]
        ascii_line = (f"轨迹: 轮均 {trend} | 累计最佳 {best['variant']}={best['fitness']:.4g}")
        return ascii_line + (f" | PNG: {png}" if png else self._no_png_hint())

    def _plot_scatter(self, target):
        frame = pd.read_csv(self.last_out_dir / "per_round.csv")
        frame = frame.dropna(subset=["measured_fitness", "predicted_fitness"])
        if frame.empty:
            return "散点: 无可绘数据（本轮回填/预测存在空值）"
        from scipy.stats import spearmanr
        if frame["predicted_fitness"].nunique() > 1 and frame["measured_fitness"].nunique() > 1:
            rho_text = f"{spearmanr(frame['predicted_fitness'], frame['measured_fitness']).statistic:.3f}"
        else:
            rho_text = "n/a（存在常数列）"

        def build(plt):
            figure, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
            ax.scatter(frame["predicted_fitness"], frame["measured_fitness"], s=28, alpha=0.75)
            ax.set(xlabel="predicted fitness", ylabel="measured fitness (oracle)",
                   title="Predicted vs measured")
            ax.grid(alpha=0.25)
            return figure

        png = self._plot_png(build, target, "scatter")
        return (f"散点: n={len(frame)} | Spearman ρ(pred,meas)={rho_text}"
                + (f" | PNG: {png}" if png else self._no_png_hint()))

    def _plot_budget(self, target):
        rounds = self.last_summary["rounds"]
        used = [rs["score_calls_used"] for rs in rounds]
        limit = rounds[0]["max_score_budget"]
        labels = [rs["round"] for rs in rounds]

        def build(plt):
            figure, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
            ax.bar(labels, used, label="score calls used")
            ax.axhline(limit, color="tab:red", ls="--", label=f"budget {limit}")
            ax.set(xlabel="round", ylabel="score calls", title="Scoring budget per round",
                   xticks=labels)
            ax.legend()
            return figure

        png = self._plot_png(build, target, "budget")
        usage = " | ".join(f"轮{r} {u}/{limit}" for r, u in zip(labels, used))
        return f"预算: {usage}" + (f" | PNG: {png}" if png else self._no_png_hint())

    def _substitution_matrix(self):
        """替换库 → (位点列表, 20×n 矩阵 mean fitness)；格式不适用时抛带指引的 ValueError。"""
        if self.config.variant_format == "indel_notation":
            raise ValueError("heatmap 面向替换库；indel 库请用 plot length（长度分布）。")
        frame = pd.read_csv(self.data_path)
        variant_col, fitness_col = _detect_columns(frame)
        fitness = dict(zip(frame[variant_col].astype(str), pd.to_numeric(frame[fitness_col])))
        cells = {}  # (位点, 残基) → [fitness…]
        for variant, value in fitness.items():
            if self.config.variant_format == "mutation_notation":
                for _wt, pos, mt in parse_mutation_notation(variant):
                    cells.setdefault((pos, mt), []).append(value)
            elif self.config.variant_format == "mutation_string":
                for i, aa in enumerate(variant):
                    cells.setdefault((i + 1, aa), []).append(value)
            else:  # full
                for pos in self.config.mutable_positions:
                    cells.setdefault((pos + 1, variant[pos]), []).append(value)
        positions = sorted({pos for pos, _ in cells})
        alphabet = self.config.aa_alphabet
        matrix = np.full((len(alphabet), len(positions)), np.nan)
        for (pos, residue), values in cells.items():
            matrix[alphabet.index(residue), positions.index(pos)] = float(np.mean(values))
        return positions, matrix

    def _plot_heatmap(self, target):
        positions, matrix = self._substitution_matrix()

        def build(plt):
            figure, ax = plt.subplots(figsize=(max(6, len(positions) * 0.22), 5), constrained_layout=True)
            image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
            ax.set(xlabel="position", ylabel="residue",
                   yticks=range(len(self.config.aa_alphabet)),
                   yticklabels=list(self.config.aa_alphabet),
                   xticks=range(0, len(positions), max(1, len(positions) // 12)),
                   xticklabels=[positions[i] for i in range(0, len(positions), max(1, len(positions) // 12))],
                   title="Mean fitness by position x residue")
            figure.colorbar(image, ax=ax, label="mean fitness")
            return figure

        png = self._plot_png(build, target, "heatmap")
        coverage = float(np.mean(~np.isnan(matrix)))
        return (f"热图: {len(positions)} 位点 × 20 残基 | 观测覆盖率 {coverage:.1%}"
                + (f" | PNG: {png}" if png else self._no_png_hint()))

    def _plot_length(self, target):
        if self.config.variant_format != "indel_notation":
            raise ValueError("length 面向 indel 库；替换库请用 plot heatmap（位点×残基热图）。")
        frame = pd.read_csv(self.data_path)
        variant_col, _ = _detect_columns(frame)
        wt_full = self.config.wt_full_sequence
        lengths = [len(reconstruct(v, wt_full)) for v in frame[variant_col].astype(str)]

        def build(plt):
            figure, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
            ax.hist(lengths, bins=min(40, max(10, len(set(lengths)))), edgecolor="white")
            ax.axvline(len(wt_full), color="tab:red", ls="--", label=f"WT {len(wt_full)}")
            ax.set(xlabel="reconstructed sequence length", ylabel="variants",
                   title="Indel length distribution")
            ax.legend()
            return figure

        png = self._plot_png(build, target, "length")
        return (f"长度分布: min {min(lengths)} | 中位 {int(np.median(lengths))} | max {max(lengths)} "
                f"| WT {len(wt_full)}"
                + (f" | PNG: {png}" if png else self._no_png_hint()))

    @staticmethod
    def _no_png_hint():
        return yellow("（matplotlib 未安装：仅 ASCII 摘要；pip install matplotlib 可出图）")

    # ---------------- 回放与导出面板 ----------------
    def do_log(self, arg):
        """log [轮次] — 回放 agent_round*.jsonl：默认逐轮摘要；指定轮次则逐条 kind/tool/参数"""
        if self.last_out_dir is None:
            print(yellow("还没有可回放的运行日志：先 run 一次。"))
            return
        files = sorted(self.last_out_dir.glob("agent_round*.jsonl"))
        if not files:
            print(yellow(f"{self.last_out_dir} 中没有 agent_round*.jsonl。"))
            return
        wanted = arg.strip()
        records = []
        for path in files:
            records += [json.loads(line) for line in
                        path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not wanted:
            print(bold("── 逐轮摘要 ──"))
            rounds = sorted({r.get("round") for r in records if r.get("round") is not None})
            for number in rounds:
                per = [r for r in records if r.get("round") == number]
                api = sum(1 for r in per if r.get("kind") == "api_call")
                # 只累计逐次 api_call 的用量：round_end 的 tokens_used 是轮末总量，一并求和会翻倍
                tokens = sum(int(r.get("tokens_used") or 0) for r in per if r.get("kind") == "api_call")
                tools = {}
                for r in per:
                    if r.get("kind") == "tool_call":
                        tools[r["tool_name"]] = tools.get(r["tool_name"], 0) + 1
                end = next((r for r in per if r.get("kind") == "round_end"), {})
                result = end.get("result") or {}
                tool_text = " ".join(f"{n}×{c}" for n, c in tools.items()) or "无"
                print(f"  第 {number} 轮：status={result.get('status')} stop={result.get('stop_reason')}"
                      f" | API {api} 次 · {tokens} tokens | 工具: {tool_text}")
            print(dim(f"逐条回放：log <轮次>（如 log 1）"))
            return
        try:
            number = int(wanted)
        except ValueError:
            print(red("轮次需为整数，如 log 1"))
            return
        per = [r for r in records if r.get("round") == number]
        if not per:
            print(yellow(f"没有第 {number} 轮的日志记录。"))
            return
        print(bold(f"── 第 {number} 轮逐条回放 ──"))
        for r in per:
            kind = r.get("kind")
            tool = r.get("tool_name")
            if kind == "round_start":
                info = r.get("result") or {}
                print(f"  [round_start] known={info.get('known_count')} "
                      f"eligible={info.get('eligible_count')} 需提交 {info.get('required_recommendations')} 条")
            elif kind == "api_call":
                result = r.get("result") or {}
                content = (result.get("content") or "")[:60]
                print(f"  [api_call] tokens={r.get('tokens_used')} finish={result.get('finish_reason')}"
                      f"{(' | ' + content + '…') if content else ''}")
            elif kind == "tool_call":
                args_text = json.dumps(r.get("arguments"), ensure_ascii=False)
                print(f"  [tool_call] {tool} {args_text[:80]}{'…' if len(args_text) > 80 else ''}")
                if tool == "submit_recommendations":
                    for item in (r.get("result") or {}).get("recommendations") or []:
                        print(f"      提交 {item.get('variant')} —— {(item.get('reason') or '')[:70]}")
            elif kind == "round_end":
                result = r.get("result") or {}
                print(f"  [round_end] status={result.get('status')} stop={result.get('stop_reason')}")
            elif kind == "api_error":
                print(red(f"  [api_error] {json.dumps(r.get('result'), ensure_ascii=False)[:100]}"))

    def do_export(self, arg):
        """export [--out DIR] — 导出 per_round.csv / summary.json / 全部适用图并打印清单"""
        if self.last_out_dir is None:
            print(yellow("还没有可导出的运行：先 run 一次。"))
            return
        try:
            _positional, values, _flags = self._parse_flags(self._tokenize(arg), valued=("--out",))
        except ValueError as exc:
            print(red(str(exc)))
            return
        out = Path(values.get("--out") or
                   Path("results/tui") / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name in ("per_round.csv", "summary.json"):
            source = self.last_out_dir / name
            if source.is_file():
                shutil.copy2(source, out / name)
                written.append(out / name)
        # agent 轮日志一并导出：缺少 jsonl 则导出目录无法回放（log 命令依赖这些文件）
        for source in sorted(self.last_out_dir.glob("agent_round*.jsonl")):
            shutil.copy2(source, out / source.name)
            written.append(out / source.name)
        # 全部适用图：运行三图 + 当前数据集的格式适用图（heatmap / length）
        applicable = ["trajectory", "scatter", "budget"]
        if self.config is not None:
            applicable.append("length" if self.config.variant_format == "indel_notation" else "heatmap")
        for kind in applicable:
            try:
                getattr(self, f"_plot_{kind}")(out)
            except Exception as exc:
                print(yellow(f"跳过 {kind} 图：{exc}"))
                continue
            png = out / f"{kind}.png"
            if png.is_file():  # 无可绘数据时（如散点全空值）不产出 PNG，也不进清单
                written.append(png)
        print(green(f"── 导出清单（{out}）──"))
        for path in written:
            print(f"  {path}（{path.stat().st_size} B）")

    # ---------------- 辅助面板（setup_assistant 委托） ----------------
    def do_plan(self, arg):
        """plan — 数据探测 + 确定性规则计划（格式判定 / 参考序列建议 / ESM 与安装命令）"""
        data_probe = None
        if self.data_path is not None:
            try:
                data_probe = probe_data(str(self.data_path),
                                        reference_sequence=self.ref_used)
            except Exception as exc:
                print(yellow(f"数据探测失败（忽略，按无数据处理）：{exc}"))
        plan = build_plan(data_probe, probe_environment())
        print(bold("── 探测 + 规则计划 ──"))
        if data_probe is not None:
            print(f"  数据: {data_probe['n_rows']} 行 · 变体列 {data_probe['variant_col']}"
                  f" · 采样 {data_probe['sampled_rows']} 行")
        print(f"  格式判定: {plan['variant_format']}（{plan['format_reason']}）")
        if plan.get("reference_note"):
            print(yellow(f"  参考序列: {plan['reference_note']}"))
        print(f"  ESM: use_esm={plan['use_esm']} · esm_model={plan['esm_model']}（{plan['esm_reason']}）"
              f" · device={plan['device']}")
        for field in ("install_core", "install_torch", "install_transformers"):
            if plan.get(field) and plan[field] != "skip":
                print(f"  安装: {plan[field]}")
        print(dim(f"  {plan['algorithms_note']}"))
        print(dim(f"  建议 configure 调用：{plan['configure_call']}"))

    def do_doctor(self, arg):
        """doctor [--execute] — 环境体检（依赖/torch/checkpoint/冒烟/LLM 凭据）；--execute 自动修复"""
        if arg.strip() and arg.strip() != "--execute":
            print("用法：doctor [--execute]")
            return
        report = run_doctor(full=True)
        if arg.strip() == "--execute":
            report = _doctor_autofix(report)
        print(render_doctor(report))

    def do_ask(self, arg):
        """ask <问题> — LLM 顾问（走 configure_with_llm；无凭据时打印降级说明与规则计划入口）"""
        question = arg.strip()
        if not question:
            print("用法：ask <问题>（如 ask 这台机器该装 CPU 版 torch 吗）")
            return
        try:
            result = configure_with_llm(data_csv=str(self.data_path) if self.data_path else None,
                                        question=question)
        except Exception as exc:
            print(red(f"LLM 顾问调用失败：{type(exc).__name__}: {exc}"))
            return
        if result["llm_used"]:
            print(bold("── LLM 顾问回答 ──"))
            print(result["answer"] or result["rationale"])
            if result["warnings"]:
                print(yellow("警告: " + "; ".join(str(w) for w in result["warnings"])))
        else:
            print(yellow(f"LLM 未参与：{result.get('note') or '未检测到 .env 凭据'}"))
            print(dim("三步配置：运行 setup；或查看规则计划：plan / 完整报告："
                      "python -m protevo.setup_assistant --doctor"))

    def do_predict(self, arg):
        """predict SEQ... — 用当前预测器预测变体（原格式进、原始标度出）"""
        if not self._require_open("预测"):
            return
        variants = arg.split()
        if not variants:
            print("用法：predict <变体1> <变体2> …（如 predict I3V K6R，格式与数据集一致）")
            return
        try:
            scores = self.config.predict_fitness(variants)
        except (ValueError, KeyError, IndexError) as exc:
            print(red(f"预测失败：{exc}（变体格式需为 {self.config.variant_format}）"))
            return
        for variant, score in zip(variants, scores):
            marker = green("▲") if score == max(scores) else " "
            print(f" {marker} {variant:<32} {score:.4g}")

    # ---------------- 引导与退出 ----------------
    def do_setup(self, arg):
        """setup — 手动重跑首次启动引导（三步配好 Base_URL / API Key / Model 并写入 .env）"""
        if sys.stdin.isatty():
            run_setup_wizard(self.env_path)
        else:
            print(yellow("当前 stdin 非交互终端，无法运行向导。请在真实终端启动 python -m protevo.tui "
                         "后运行 setup，或手动编辑包根 .env（OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL）。"))

    def do_quit(self, arg):
        """quit — 退出（EOF/Ctrl-D 同效）"""
        print("再见。")
        return True

    def do_EOF(self, arg):
        print()
        return self.do_quit(arg)


# ---------------------------------------------------------------------------
# 启动入口：python -m protevo.tui [--no-wizard]
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    no_wizard = "--no-wizard" in argv
    if hasattr(sys.stdout, "reconfigure"):  # Windows 重定向输出编码兜底
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not no_wizard and not credentials_complete(ENV_PATH):
        if sys.stdin.isatty():
            run_setup_wizard(ENV_PATH)
        else:  # 管道/测试：不阻塞，一行说明 + setup 引导后直接进 REPL
            print(yellow("未检测到 LLM 凭据（.env 缺 OPENAI_API_KEY / OPENAI_MODEL）："
                         "run（Agent 进化）与 ask（LLM 顾问）不可用，预测器/复测功能不受影响；"
                         "进入 REPL 后可运行 setup 三步配置。"))
    shell = ProtevoShell()
    shell.cmdloop(intro=bold("protevo TUI —— 命令式蛋白定向进化交互台（help 查看命令；quit 退出）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

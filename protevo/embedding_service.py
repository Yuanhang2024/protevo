# embedding_service.py — 按需 ESM-2 嵌入服务：任意新蛋白序列 → 末层 mean-pooled 1280 维向量
# 提取口径与 embed_gb1.py 完全一致：transformers 本地 checkpoint、EsmTokenizer 前后加
# BOS(<cls>)/EOS（残基 i → 模型输出行 i+1，即 BOS 偏移 +1）、末层 hidden state 取残基位置平均。
#
# 设计要点：
#   - torch/transformers 仅在 EmbeddingService.__init__ 内导入：无 torch 环境（本项目 .venv）
#     可正常 import 本模块、调用长度护栏；只有真正构造服务才要求 GPU 环境；
#   - 硬件自适应三件套（resolve_device / hardware_probe / ensure_esm_checkpoint）同为模块级
#     函数 + 惰性导入：无 torch / 无 huggingface_hub 环境均可安全调用（探针降级、解析器明确报错）；
#   - 双层缓存：内存 OrderedDict（超容量按插入序淘汰最旧）+ 可选磁盘缓存（npz 向量 + json 索引，
#     下次构造时加载，跨进程复用；实现从简，每次全量重写）；
#   - 长度护栏为模块级函数：序列 > 1022 残基显式 ValueError（ESM-2 位置编码上限，
#     1022 残基 + BOS/EOS = 1024 token），不静默截断。
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

EMBED_DIM = 1280            # ESM-2 650M（esm2_t33_650M_UR50D）隐藏维
MAX_SEQUENCE_LENGTH = 1022  # ESM-2 上限：1022 残基 + BOS/EOS = 1024 token
BATCH_SIZE = 32             # 与 embed_gb1.py 默认批量一致；EmbeddingService(batch_size=...) 的默认值锚点


def _cache_meta_path(cache_dir) -> Path:
    return Path(cache_dir) / "model.json"


def _guard_cache_model(cache_dir, model_path, esm_layer="last"):
    """磁盘缓存模型守卫：缓存目录绑定单一（模型, 嵌入层）组合。

    不同规格 ESM-2 嵌入维度不同（8M=320 / 150M=640 / 650M=1280 / 3B=2560），
    下游回归头不挑输入维度；同模型不同层（esm_layer）的向量同样互不通用——
    两者错配复用旧缓存都会**静默**拿到错误向量。此处在加载 torch 之前校验
    model.json，任一不符即拒绝。缓存目录按（模型, 层）分目录使用。
    旧档案无 esm_layer 键时按 "last" 兼容（该参数引入前只有末层一种行为）。
    """
    meta_path = _cache_meta_path(cache_dir)
    if not meta_path.exists():
        return
    recorded = json.loads(meta_path.read_text(encoding="utf-8"))
    if recorded.get("model_path") != str(model_path):
        raise ValueError(
            f"嵌入缓存目录属于其他 ESM 模型（缓存记录: {recorded.get('model_path')}，"
            f"dim={recorded.get('dim')}），当前请求: {model_path}。"
            f"请为当前模型更换缓存目录，或删除旧缓存后重算。"
        )
    recorded_layer = recorded.get("esm_layer", "last")
    if recorded_layer != esm_layer:
        raise ValueError(
            f"嵌入缓存目录属于同一模型的另一嵌入层（缓存记录: esm_layer={recorded_layer}，"
            f"dim={recorded.get('dim')}），当前请求: esm_layer={esm_layer}。"
            f"请更换缓存目录或删除旧缓存后重算。"
        )


def validate_sequence_lengths(sequences):
    """长度护栏：任一序列超过 ESM-2 上限 1022 残基 → ValueError。

    模块级函数、不依赖 torch：EmbeddingService.encode 在任何推理发生前首先调用，
    无 GPU 环境也能对护栏本身做单元测试。
    """
    for seq in sequences:
        if len(seq) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"序列长度 {len(seq)} 超过 ESM-2 上限 {MAX_SEQUENCE_LENGTH} 残基"
                f"（加 BOS/EOS 共 1024 token，超出位置编码范围）——显式拒绝，不做静默截断"
            )


def resolve_device(requested: str = "auto") -> str:
    """设备解析：'auto' → CUDA 可用（含 ROCm）选 'cuda'，否则 'cpu'。

    显式 'cuda'/'cpu' 原样返回；显式 'cuda' 但 torch.cuda.is_available() 为
    False → RuntimeError，消息按三种情况分别给出针对性提示（CPU 版 wheel /
    驱动不可见 / ROCm 运行时）。

    设计契约：正常只在有 torch 的环境中被调用（如 EmbeddingService.__init__，
    构造时 torch 已导入成功），但本函数自身惰性导入 torch——无 torch 环境直接
    调用会抛含 'torch' 的 ImportError（而非 NameError），契约可被两档测试验证。

    ROCm 说明：PyTorch ROCm 版将 hip 设备映射为 cuda 设备名，
    torch.cuda.is_available() 同样返回 True，因此 'auto' 对 ROCm 零改动生效。
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"resolve_device 需要 torch，当前环境不可用: {exc}") from exc

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            hint = ("当前为 ROCm 版 PyTorch——请确认 ROCm/hip 运行时已正确安装且 GPU 可见"
                    "（rocminfo / rocm-smi），或改用 device='cpu'")
        elif torch.version.cuda is None:
            hint = ("当前安装的是 CPU 版 PyTorch wheel——请改装 CUDA 版，例如 "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu126，"
                    "或改用 device='cpu'")
        else:
            hint = (f"当前为 CUDA {torch.version.cuda} 版 PyTorch 但未检测到可用 GPU——"
                    "请确认显卡驱动已安装且 nvidia-smi 能看到该卡，或改用 device='cpu'")
        raise RuntimeError(f"显式要求 device='cuda' 但 CUDA 不可用。{hint}")
    return requested


def hardware_probe() -> dict:
    """硬件探针：torch/CUDA 可用性与后端信息，任何环境（含无 torch）都不抛错。

    返回键：torch / cuda / backend / device_name / vram_gb / transformers。
      - 无 torch：全部降级（torch/cuda=False，backend/device_name/vram_gb=None，
        transformers 用 find_spec 探测不实际导入）
      - backend 三态：torch.version.cuda → f"CUDA {x}"；torch.version.hip →
        f"ROCm {x}"；二者皆无 → "CPU"
      - device_name / vram_gb 仅 CUDA（含 ROCm 映射）可用时取
        torch.cuda.get_device_properties(0)
    """
    try:
        import torch
    except ImportError:
        return {"torch": False, "cuda": False, "backend": None,
                "device_name": None, "vram_gb": None,
                "transformers": importlib.util.find_spec("transformers") is not None}

    cuda_ok = torch.cuda.is_available()
    if torch.version.cuda is not None:
        backend = f"CUDA {torch.version.cuda}"
    elif getattr(torch.version, "hip", None) is not None:
        backend = f"ROCm {torch.version.hip}"
    else:
        backend = "CPU"
    device_name, vram_gb = None, None
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        device_name, vram_gb = props.name, round(props.total_memory / 1024 ** 3, 1)
    return {"torch": True, "cuda": cuda_ok, "backend": backend,
            "device_name": device_name, "vram_gb": vram_gb,
            "transformers": importlib.util.find_spec("transformers") is not None}


def ensure_esm_checkpoint(path: str = None,
                          model_id: str = "facebook/esm2_t33_650M_UR50D",
                          download: bool = True) -> str:
    """确保 ESM-2 checkpoint 就绪，返回可直接传给 EmbeddingService 的本地目录。

    查找/获取顺序：
      1. path 给定且其目录含 config.json → 原样返回该路径（本机离线 checkpoint，
         如 D:/ChimeraModel_Lite/esm2_t33_650M_UR50D 即此情形，绝不触发下载）
      2. 默认缓存 ~/.cache/protevo/esm2_t33_650M 已有 config.json → 复用
      3. download=True → 惰性 import 并调用
         huggingface_hub.snapshot_download(model_id, local_dir=目标目录)
         （path 给定但未就绪时，目标目录即该 path，下载完成后它就是合法 checkpoint）

    国内镜像：huggingface_hub 原生尊重环境变量 HF_ENDPOINT，设
      HF_ENDPOINT=https://hf-mirror.com（Windows CMD: set HF_ENDPOINT=https://hf-mirror.com）
    后再调用本函数即可全程走镜像。

    huggingface_hub 缺失或网络失败 → RuntimeError，消息附手动下载命令两式
    （直连与镜像前缀的 huggingface-cli download）。
    """
    default_cache = Path.home() / ".cache" / "protevo" / "esm2_t33_650M"

    def _ready(p) -> bool:
        d = Path(p)
        d = d if d.is_dir() else d.parent  # path 可能是目录，也可能是目录下某文件
        return (d / "config.json").is_file()

    def _manual_cmd(target) -> str:
        return (f"可手动下载（两式，任选其一）：\n"
                f"  huggingface-cli download {model_id} --local-dir {target}\n"
                f"  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download {model_id} "
                f"--local-dir {target}")

    if path is not None:
        if _ready(path):
            return str(path)
        target = Path(path)
    else:
        target = default_cache
        if _ready(target):
            return str(target)

    if not download:
        raise RuntimeError(
            f"ESM-2 checkpoint 未就绪（{target} 下无 config.json）且 download=False。"
            + _manual_cmd(target)
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"huggingface_hub 不可用（{exc}），无法自动下载 ESM-2 checkpoint。"
            + _manual_cmd(target)
        ) from exc
    try:
        snapshot_download(model_id, local_dir=str(target))
    except Exception as exc:
        raise RuntimeError(
            f"ESM-2 checkpoint 自动下载失败（{type(exc).__name__}: {exc}）。"
            + _manual_cmd(target)
        ) from exc
    return str(target)


class EmbeddingService:
    """按需 ESM-2 嵌入服务：encode(序列列表) → (n, 1280) float32 mean-pooled 嵌入（默认末层，esm_layer 可选层）。"""

    def __init__(self, model_path: str, device: str = "auto",
                 cache_dir: str = None, lru_capacity: int = 200_000,
                 esm_layer="last", batch_size: int = BATCH_SIZE):
        """esm_layer：池化取哪一层 hidden state。"last"（默认）= 末层（t33 模型的第 33 层，
        与 embed_gb1.py 口径一致，默认行为零变化）；整数 i = hidden_states[i]（0 = embedding
        层输出，t33 末层为 33），越界抛 ValueError（消息含可用范围）。

        嵌入层选择应显式消融而非写死末层——FLIP 实践中不同层与不同池化方式各有适用、
        无普适赢家（见 docs/knowledge_calibration.md 第 5 条）；突变区域池化（mut-mean）
        需要变体→位点的信息贯通本服务链路，记为后续项。

        batch_size：批量推理批大小，默认取模块常量 BATCH_SIZE=32（与 embed_gb1.py 一致）。
        """
        # 缓存模型守卫先行（无需 torch）：目录与模型绑定，换模型显式拒绝而非静默复用
        if cache_dir is not None:
            _guard_cache_model(cache_dir, model_path, esm_layer)
        # torch/transformers 惰性导入：模块级 import 本文件不触发，构造时才加载
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                f"EmbeddingService 依赖 torch 与 transformers，当前环境不可用: {exc}"
            ) from exc

        self.torch = torch
        self.model_path = model_path
        self.device = resolve_device(device)   # "auto" 解析为实际设备；显式值校验后原样保留
        self.backend = hardware_probe()["backend"]  # "CUDA x" / "ROCm x" / "CPU"
        self.cache_dir = cache_dir
        self.lru_capacity = lru_capacity
        self.batch_size = batch_size
        self.esm_layer = esm_layer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval().to(self.device)
        self._validate_esm_layer()

        # 内存缓存：{序列: 向量}，超出容量按插入序淘汰最旧（简单 OrderedDict，非访问序 LRU）
        self._cache = OrderedDict()
        # 磁盘缓存（cache_dir 给定时启用）：已落盘向量全集，构造时加载
        self._disk = {}
        self._dirty = False  # 有新向量尚未写盘
        if cache_dir is not None:
            # 首次使用该目录时落模型档案（供 _guard_cache_model 校验；守卫在 torch 导入前已跑过）
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            meta_path = _cache_meta_path(cache_dir)
            if not meta_path.exists():
                meta_path.write_text(json.dumps(
                    {"model_path": str(model_path),
                     "esm_layer": self.esm_layer,
                     "dim": int(getattr(self.model.config, "hidden_size", 0) or 0)},
                    ensure_ascii=False), encoding="utf-8")
            self._load_disk_cache()

    def _validate_esm_layer(self):
        """esm_layer 校验（需模型已加载取 num_hidden_layers，故在权重加载后执行）。

        hidden_states 共 num_hidden_layers + 1 项：0 = embedding 层输出，
        1..num_hidden_layers = 各 transformer 层（末层即 -1，t33 模型为 33）。
        """
        if self.esm_layer == "last":
            return
        if not isinstance(self.esm_layer, int):
            raise ValueError(f"esm_layer 需为 'last' 或整数层号（0 = embedding 层输出），收到: {self.esm_layer!r}")
        n_layers = int(self.model.config.num_hidden_layers)
        if not 0 <= self.esm_layer <= n_layers:
            raise ValueError(
                f"esm_layer={self.esm_layer} 越界：hidden_states 共 {n_layers + 1} 项，"
                f"可用整数层号范围 0..{n_layers}（0 = embedding 层输出，{n_layers} = 末层），"
                f"或用默认 'last' 取末层。")

    def encode(self, sequences: list) -> np.ndarray:
        """批量编码：mean-pool 末层 hidden state（BOS 偏移 +1），返回 (n, 1280) float32。

        缓存未命中的序列去重后批量推理；同批重复序列只推理一次。
        """
        validate_sequence_lengths(sequences)  # 护栏最先执行：非法输入不触发任何推理
        out = np.empty((len(sequences), EMBED_DIM), dtype=np.float32)
        todo = []  # 缓存未命中、需要推理的 (输入行号, 序列)
        for i, seq in enumerate(sequences):
            vec = self._lookup(seq)
            if vec is None:
                todo.append((i, seq))
            else:
                out[i] = vec
        if todo:
            uniq = list(dict.fromkeys(seq for _, seq in todo))
            computed = self._forward(uniq)
            for i, seq in todo:
                out[i] = computed[seq]
            self._persist()
        return out

    # ---- 缓存 ----

    def _lookup(self, seq):
        """内存 → 磁盘 → None；磁盘命中时提升进内存缓存。"""
        vec = self._cache.get(seq)
        if vec is not None:
            return vec
        vec = self._disk.get(seq)
        if vec is not None:
            self._remember(seq, vec, persisted=True)
        return vec

    def _remember(self, seq, vec, persisted=False):
        """写入内存缓存（超容量按插入序淘汰最旧）；新算向量登记进磁盘缓存并置脏。"""
        self._cache[seq] = vec
        while len(self._cache) > self.lru_capacity:
            self._cache.popitem(last=False)
        if self.cache_dir is not None and not persisted and seq not in self._disk:
            self._disk[seq] = vec
            self._dirty = True

    # ---- 推理（口径与 embed_gb1.py 一致）----

    def _forward(self, sequences):
        """批量前向并写入缓存，返回 {序列: 向量}。

        取层由 esm_layer 决定："last" → hidden_states[-1]（t33 模型的第 33 层，与
        embed_gb1.py hidden_states[33] 相同）；整数 i → hidden_states[i]（0 = embedding
        层输出）。padding 只补尾端，按各序列实际长度切片，pad/BOS/EOS 行均不进入平均。
        """
        computed = {}
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start:start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)
            layer = (outputs.hidden_states[-1] if self.esm_layer == "last"
                     else outputs.hidden_states[self.esm_layer])
            hidden = layer.float().cpu().numpy()  # (B, L_max+2, 1280)
            for j, seq in enumerate(batch):
                vec = hidden[j, 1:len(seq) + 1, :].mean(axis=0)  # 残基 i → 行 i+1（BOS 偏移）
                computed[seq] = vec
                self._remember(seq, vec)
        return computed

    # ---- 磁盘持久化（实现从简：有新增即全量重写 npz + json 索引）----

    def _load_disk_cache(self):
        """构造时加载磁盘缓存：embeddings.npz（N×1280）+ index.json（{序列: 行号}）。"""
        cache = Path(self.cache_dir)
        npz_path, index_path = cache / "embeddings.npz", cache / "index.json"
        if not (npz_path.exists() and index_path.exists()):
            return
        vectors = np.load(npz_path)["embeddings"]
        for seq, row in json.loads(index_path.read_text(encoding="utf-8")).items():
            self._disk[seq] = vectors[int(row)]

    def _persist(self):
        if self.cache_dir is None or not self._dirty:
            return
        cache = Path(self.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        seqs = list(self._disk.keys())
        matrix = (np.stack([self._disk[s] for s in seqs]) if seqs
                  else np.empty((0, EMBED_DIM), dtype=np.float32))
        np.savez(cache / "embeddings.npz", embeddings=matrix)
        index = {seq: row for row, seq in enumerate(seqs)}
        (cache / "index.json").write_text(json.dumps(index), encoding="utf-8")
        self._dirty = False

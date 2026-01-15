# utils/sia_idt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Literal, List
import numpy as np


@dataclass
class SIAResult:
    pred_cid: int
    losses: Dict[int, float]  # cid -> loss (lower is better)
    probs: Optional[Dict[int, float]]  # softmax(-loss/T)
    confidence: float  # 预测置信度
    num_ties: int  # 平局数量


@dataclass
class SIAMetrics:
    """
    SIA 核心指标（精简版）

    三个核心指标：
    1. ASR: 攻击成功率
    2. Confidence: 平均预测置信度
    3. Tie Rate: 平局率
    """
    asr: float
    confidence: float
    tie_rate: float
    num_clients: int
    num_samples: int
    random_baseline: float

    def __str__(self) -> str:
        if self.tie_rate > 0.3 or self.confidence < self.random_baseline * 2:
            privacy_level = "HIGH 🛡️"
        elif self.tie_rate > 0.1 or self.confidence < 0.5:
            privacy_level = "MEDIUM 🔒"
        else:
            privacy_level = "LOW ⚠️"

        return (
            f"╔═══════════════════════════════════════════╗\n"
            f"║        SIA Attack Metrics (IDT)           ║\n"
            f"╠═══════════════════════════════════════════╣\n"
            f"║ ASR:               {self.asr:6.2%}                   ║\n"
            f"║ Random Baseline:   {self.random_baseline:6.2%}                   ║\n"
            f"║ Advantage:         {self.asr / self.random_baseline:6.2f}x                  ║\n"
            f"╠═══════════════════════════════════════════╣\n"
            f"║ Confidence:        {self.confidence:6.2%}                   ║\n"
            f"║ Tie Rate:          {self.tie_rate:6.2%}                   ║\n"
            f"╠═══════════════════════════════════════════╣\n"
            f"║ Privacy Level: {privacy_level:26s} ║\n"
            f"╠═══════════════════════════════════════════╣\n"
            f"║ Samples: {self.num_samples:5d}  |  Clients: {self.num_clients:2d}        ║\n"
            f"╚═══════════════════════════════════════════╝"
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            'asr': self.asr,
            'confidence': self.confidence,
            'tie_rate': self.tie_rate,
            'random_baseline': self.random_baseline,
            'advantage': self.asr / self.random_baseline,
        }


LossMode = Literal["cross_entropy", "margin_loss"]


class SourceInferenceAttackerIDT:
    """
    基于论文的样本级源推断攻击（IDT 版本）

    核心原理（对应论文 Theorem 2）:
        pred_cid = argmin_k ℓ(θ_k, z)
        即：预测损失最小的客户端最可能拥有该样本

    loss_mode:
      - "cross_entropy": -log p(y|x)  [推荐，对应论文]
      - "margin_loss": -[p(y|x) - max_{c≠y} p(c|x)]
    """

    def __init__(
            self,
            loss_mode: LossMode = "cross_entropy",
            temperature: float = 1.0,
            use_probs: bool = True,
            print_ties: bool = False,
            eps: float = 1e-12,
            seed: int = 42,
    ):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.loss_mode = loss_mode
        self.temperature = float(temperature)
        self.use_probs = bool(use_probs)
        self.print_ties = bool(print_ties)
        self.eps = float(eps)

    # -------------------------
    # 数据转换辅助函数
    # -------------------------
    def _to_single_graph_batch(self, data_or_batch: Any) -> Any:
        """将单个图转换为 Batch"""
        try:
            from torch_geometric.data import Batch
        except Exception as e:
            raise RuntimeError("需要 torch_geometric") from e

        if hasattr(data_or_batch, "to_data_list"):
            dl = data_or_batch.to_data_list()
            if len(dl) != 1:
                raise ValueError(f"期望单图输入，但得到 {len(dl)} 个图")
            return Batch.from_data_list(dl)

        return Batch.from_data_list([data_or_batch])

    def _iter_graphs(self, data_or_batch: Any) -> List[Any]:
        """从 Batch 或单个 Data 中提取图列表"""
        if hasattr(data_or_batch, "to_data_list"):
            return list(data_or_batch.to_data_list())
        return [data_or_batch]

    # -------------------------
    # 核心：损失计算
    # -------------------------
    def _extract_label_single(self, batch_1: Any) -> Optional[int]:
        """提取单图标签"""
        if not hasattr(batch_1, "y") or batch_1.y is None:
            return None
        y = batch_1.y
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        y = np.asarray(y).reshape(-1)
        if y.size == 0:
            return None
        return int(y[0])

    def _get_classes_mapping(self, idt: Any) -> Optional[Dict[int, int]]:
        """获取类别到 predict_proba 列索引的映射"""
        try:
            dt = idt.out_layer.dt
            classes = getattr(dt, "classes_", None)
            if classes is None:
                return None
            classes = np.asarray(classes).tolist()
            return {int(c): i for i, c in enumerate(classes)}
        except Exception:
            return None

    def _ensure_proba_2d(self, proba: Any) -> np.ndarray:
        """确保 proba 是 2D 数组"""
        p = np.asarray(proba, dtype=np.float64)
        if p.ndim == 1:
            p = np.stack([1.0 - p, p], axis=1)
        return p

    def _compute_loss_single(self, idt: Any, batch_1: Any) -> float:
        """
        计算单个样本的预测损失（对应论文中的 ℓ(θ_k, z)）

        返回值越小 => 该客户端越可能拥有此样本
        """
        if not hasattr(idt, "predict_proba"):
            raise AttributeError("需要模型实现 predict_proba(batch)")

        # 获取预测概率 (1, C)
        proba = self._ensure_proba_2d(idt.predict_proba(batch_1))
        if proba.shape[0] != 1:
            raise ValueError(f"期望单样本预测概率 (1,C)，得到 {proba.shape}")

        # 获取真实标签
        y = self._extract_label_single(batch_1)
        if y is None:
            raise ValueError("SIA 需要真实标签来计算损失")

        # 处理类别映射（sklearn 可能只训练了部分类别）
        cls2col = self._get_classes_mapping(idt)
        if cls2col is None:
            yy = int(np.clip(y, 0, proba.shape[1] - 1))
        else:
            yy = cls2col.get(int(y), None)
            if yy is None:
                return float('inf')

        # 计算损失
        if self.loss_mode == "cross_entropy":
            p_true = np.clip(proba[0, yy], self.eps, 1.0)
            return float(-np.log(p_true))

        elif self.loss_mode == "margin_loss":
            p = proba[0].copy()
            p_true = float(p[yy])
            p[yy] = -np.inf
            p_second = float(np.max(p)) if np.isfinite(np.max(p)) else 0.0
            margin = p_true - p_second
            return float(-margin)

        else:
            raise ValueError(f"未知 loss_mode: {self.loss_mode}")

    # -------------------------
    # 公共 API
    # -------------------------
    def infer_one_sample(self, idts_by_cid: Dict[int, Any], target_data: Any) -> SIAResult:
        """
        对单个样本进行源推断

        Args:
            idts_by_cid: {client_id: IDT_model} 字典
            target_data: PyG Data（单图）或包含一个图的 Batch

        Returns:
            SIAResult(pred_cid, losses, probs, confidence, num_ties)
        """
        losses: Dict[int, float] = {}

        batch_1 = self._to_single_graph_batch(target_data)
        batch_1 = batch_1.to("cpu") if hasattr(batch_1, "to") else batch_1

        # 计算每个客户端模型的损失
        for cid, idt in idts_by_cid.items():
            losses[cid] = self._compute_loss_single(idt, batch_1)

        # 选择损失最小的客户端
        min_loss = min(losses.values())
        ties = [cid for cid, v in losses.items() if np.isclose(v, min_loss, rtol=1e-12, atol=1e-12)]

        if len(ties) == 1:
            pred_cid = ties[0]

            if self.print_ties:
                # 非平局：打印所有客户端的 loss
                print("[SIA][Unique-Min] per-client losses:")
                for cid in sorted(losses.keys()):
                    mark = "  <-- min" if cid == pred_cid else ""
                    print(f"    client {cid:2d}: loss = {losses[cid]:.6f}{mark}")

        else:
            pred_cid = int(self._rng.choice(ties))
            if self.print_ties:
                print(
                    f"[SIA][Tie] {len(ties)} clients share min loss {min_loss:.6f}: "
                    f"{sorted(ties)} -> randomly pick {pred_cid} (seed={self.seed})"
                )

        # 计算后验概率分布
        probs = None
        if self.use_probs:
            cids = sorted(losses.keys())
            L = np.array([losses[c] for c in cids], dtype=np.float32)
            logits = -L / max(self.temperature, 1e-8)
            exps = np.exp(logits - logits.max())
            p = exps / (exps.sum() + 1e-12)
            probs = {c: float(p[i]) for i, c in enumerate(cids)}

        # 计算置信度
        confidence = probs[pred_cid] if probs else (1.0 / len(idts_by_cid))

        return SIAResult(
            pred_cid=pred_cid,
            losses=losses,
            probs=probs,
            confidence=confidence,
            num_ties=len(ties),
        )

    def eval_asr(
            self,
            idts_by_cid: Dict[int, Any],
            targets_by_true_cid: Dict[int, Any],
    ) -> Tuple[SIAMetrics, Dict[int, List[SIAResult]]]:
        """
        评估攻击成功率并返回3个核心指标

        Args:
            idts_by_cid: {client_id: IDT_model}
            targets_by_true_cid: {true_client_id: Data/Batch/List[Data]}

        Returns:
            (metrics, results)
            - metrics: SIAMetrics 对象
            - results: {true_cid: [SIAResult, ...]}
        """
        correct, total = 0, 0
        results: Dict[int, List[SIAResult]] = {}

        all_confidences = []
        all_num_ties = []

        for true_cid, data_or_batch in targets_by_true_cid.items():
            graphs = self._iter_graphs(data_or_batch)
            per_client_results: List[SIAResult] = []

            for g in graphs:
                res = self.infer_one_sample(idts_by_cid, g)
                per_client_results.append(res)
                total += 1
                if res.pred_cid == true_cid:
                    correct += 1

                all_confidences.append(res.confidence)
                all_num_ties.append(res.num_ties)

            results[true_cid] = per_client_results

        # 计算3个核心指标
        num_clients = len(idts_by_cid)
        asr = correct / max(total, 1)
        mean_confidence = float(np.mean(all_confidences))
        tie_rate = float(np.mean([1 if n > 1 else 0 for n in all_num_ties]))

        metrics = SIAMetrics(
            asr=asr,
            confidence=mean_confidence,
            tie_rate=tie_rate,
            num_clients=num_clients,
            num_samples=total,
            random_baseline=1.0 / num_clients,
        )

        return metrics, results


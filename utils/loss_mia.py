import numpy as np
from sklearn.metrics import roc_auc_score


class MIAttacker:

    def __init__(self):
        self.attack_model = None

    # -------------------------
    # Core: membership score
    # -------------------------
    def membership_score(self, idt, batch):
        """
        返回每个样本的 membership score，score 越大越像 member。
        优先使用：score = - per-sample loss
        退化方案：score = max confidence
        """
        # 1) 优先：如果 IDT 提供 per-sample loss
        loss = self._try_compute_per_sample_loss(idt, batch)
        if loss is not None:
            # loss 越小越像 member -> score 用 -loss
            return -loss.astype(np.float64)

        # 2) 否则：用 max confidence / 或二分类 proba
        proba = self._predict_proba(idt, batch)
        if proba is None:
            raise RuntimeError("Cannot compute membership score: neither per-sample loss nor predict_proba is available.")

        proba = np.asarray(proba)
        if proba.ndim == 1:
            # binary prob for class=1
            return proba.astype(np.float64)
        else:
            # multi-class: max confidence
            return np.max(proba, axis=1).astype(np.float64)

    # -------------------------
    # Public API (keep unchanged)
    # -------------------------
    def train_attack_model(self, idt, member_batch, nonmember_batch):
        """
        兼容你原来的调用方式，但内部不训练任何 attack model。
        直接计算 loss-based scores 并输出 AUC。
        """
        s_mem = self.membership_score(idt, member_batch)
        s_non = self.membership_score(idt, nonmember_batch)

        y = np.concatenate([np.ones_like(s_mem, dtype=np.int64),
                            np.zeros_like(s_non, dtype=np.int64)])
        scores = np.concatenate([s_mem, s_non])

        # AUC
        auc = roc_auc_score(y, scores)
        print(f"[MIA][loss-based] AUC: {auc:.4f}  (higher score => more likely member)")
        return auc

    def infer(self, idt, sample_batch):
        """
        对新样本输出 membership score（可用于画 ROC 或做阈值）。
        """
        return self.membership_score(idt, sample_batch)

    # -------------------------
    # Helpers
    # -------------------------
    def _predict_proba(self, idt, batch):
        """
        尽量以最少假设拿到每个样本的预测概率。
        """
        # 1) sklearn-style / 你自定义 IDT 若支持直接 batch 输入
        if hasattr(idt, "predict_proba"):
            try:
                return idt.predict_proba(batch)
            except Exception:
                pass

        # 2) 如果只有 predict 输出 logits/score
        if hasattr(idt, "predict"):
            try:
                scores = idt.predict(batch)
                return self._scores_to_proba(scores)
            except Exception:
                pass

        return None

    def _try_compute_per_sample_loss(self, idt, batch):
        """
        尝试获得 per-sample loss:
          - 若 idt.compute_loss(batch) 返回 (N,) 则直接用
          - 若返回标量，则无法作为 per-sample loss（返回 None）
          - 若不存在 compute_loss，则尝试用 (proba, y) 自己计算 NLL
        """
        # A) IDT 自带 compute_loss
        if hasattr(idt, "compute_loss"):
            try:
                loss = idt.compute_loss(batch)
                loss = np.asarray(loss)
                # 允许 (N,) 或 (N,1)
                if loss.ndim == 2 and loss.shape[1] == 1:
                    loss = loss[:, 0]
                # 标量 loss 无法区分样本，退化
                if loss.ndim == 0:
                    return None
                return loss
            except Exception:
                pass

        # B) 自己用 NLL 计算 per-sample loss（需要 y）
        y = self._extract_labels(batch)
        if y is None:
            return None

        proba = self._predict_proba(idt, batch)
        if proba is None:
            return None

        proba = np.asarray(proba)
        eps = 1e-12

        # 二分类：proba 为 (N,) 或 (N,2)
        if proba.ndim == 1:
            # y in {0,1}
            p1 = np.clip(proba, eps, 1.0 - eps)
            # NLL = -[y log p + (1-y) log(1-p)]
            return -(y * np.log(p1) + (1 - y) * np.log(1 - p1))

        # 多分类：proba (N,C)
        if proba.ndim == 2:
            y = y.astype(np.int64)
            # 防止 y 越界
            y = np.clip(y, 0, proba.shape[1] - 1)
            p = np.clip(proba[np.arange(len(y)), y], eps, 1.0)
            return -np.log(p)

        return None

    def _extract_labels(self, batch):
        """
        从 PyG Batch/自定义 batch 中提取图级标签。
        你当前用的是 graph classification，所以一般是 batch.y (num_graphs,).
        """
        # PyG Batch 常见：batch.y
        if hasattr(batch, "y") and batch.y is not None:
            try:
                y = batch.y
                # torch tensor -> numpy
                if hasattr(y, "detach"):
                    y = y.detach().cpu().numpy()
                y = np.asarray(y).reshape(-1)
                return y
            except Exception:
                return None

        return None

    def _scores_to_proba(self, scores):
        """
        将预测分数转换为概率：
          - (N,) -> sigmoid
          - (N,C) -> softmax
        """
        s = np.asarray(scores)

        if s.ndim == 1:
            # sigmoid
            s = np.clip(s, -50, 50)
            return 1.0 / (1.0 + np.exp(-s))

        if s.ndim == 2:
            # softmax
            s = s - np.max(s, axis=1, keepdims=True)
            e = np.exp(np.clip(s, -50, 50))
            return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

        raise ValueError(f"Unsupported score shape: {s.shape}")

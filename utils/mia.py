from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import torch

class MIAttacker:
    """
    纯 loss-based MIA：
    - 仅使用逐样本损失作为攻击特征 (N,1)
    - 优先使用 IDT 的 compute_loss_per_sample(batch)
    - 无则尝试 idt.predict_proba(batch) → 交叉熵
    - 再无则解析 (X,y) 走通用概率路径
    """

    def __init__(self, random_state=0):
        self.attack_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                solver='lbfgs',
                random_state=random_state,
            )
        )

    # ----- 工具 -----
    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _as_numpy_X_y(self, batch):
        # 解析 (X, y) —— 仅在通用概率路径需要
        X, y = None, None
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            X = batch[0]; y = batch[1] if len(batch) >= 2 else None
        elif hasattr(batch, "x"):  # PyG Data/Batch
            X = batch.x; y = getattr(batch, "y", None)
        else:
            X = batch
        X = self._to_numpy(X)
        y = self._to_numpy(y) if y is not None else None
        if y is not None and y.ndim > 1 and y.shape[-1] == 1:
            y = y.reshape(-1)
        return X, y

    def _safe_predict_proba_X(self, model, X):
        # 以 numpy X 为输入的通用路径
        if hasattr(model, "predict_proba"):
            P = model.predict_proba(X)
            P = np.atleast_2d(P)
            if P.ndim == 1:
                P = np.stack([1 - P, P], axis=1)
            return P
        if hasattr(model, "decision_function"):
            S = model.decision_function(X)
            S = np.atleast_2d(S)
            if S.shape[1] == 1:
                z = S[:, 0]; p1 = 1.0 / (1.0 + np.exp(-z))
                return np.stack([1 - p1, p1], axis=1)
            S = S - S.max(axis=1, keepdims=True)
            expS = np.exp(S)
            return expS / expS.sum(axis=1, keepdims=True)
        raise ValueError("模型缺少 predict_proba/decision_function（X 路径）。")

    def _predict_proba_generic(self, idt, batch):
        """
        先尝试 batch 作为参数（IDT 风格），失败再退到 X 路径。
        """
        if hasattr(idt, "predict_proba"):
            try:
                P = idt.predict_proba(batch)  # 先试 Batch 风格
                P = np.asarray(P)
                if P.ndim == 1:
                    P = np.stack([1 - P, P], axis=1)
                return P
            except Exception:
                pass  # 再退到 X 路径

        X, _ = self._as_numpy_X_y(batch)
        if X is None:
            raise ValueError("无法从 batch 解析出 X 以走概率路径。")
        return self._safe_predict_proba_X(idt, X)

    def _loss_from_proba(self, P, y=None, eps=1e-12):
        if y is not None:
            y = y.astype(int)
            y = np.clip(y, 0, P.shape[1]-1)
            idx = np.arange(P.shape[0])
            return -np.log(P[idx, y] + eps)
        else:
            return -np.log(P.max(axis=1) + eps)

    def _loss_via_idt_api(self, idt, batch):
        """
        优先通过 IDT 的逐样本接口拿 loss：
            - compute_loss_per_sample(batch)
            - compute_loss(batch, reduction='none') / (per_sample=True)（若你以后添加）
        """
        if hasattr(idt, "compute_loss_per_sample"):
            L = idt.compute_loss_per_sample(batch)
            return self._to_numpy(L).reshape(-1)
        if hasattr(idt, "compute_loss"):
            try:
                L = idt.compute_loss(batch, reduction='none')
                L = self._to_numpy(L).reshape(-1)
                if L.size > 1:
                    return L
            except Exception:
                pass
            try:
                L = idt.compute_loss(batch, per_sample=True)
                L = self._to_numpy(L).reshape(-1)
                if L.size > 1:
                    return L
            except Exception:
                pass
        return None

    # ----- 核心接口 -----
    def extract_features(self, idt, batch, is_member):
        """
        返回 (features, labels)
        - features: (N,1)，仅包含逐样本损失
        - labels: (N,)
        """
        try:
            # 1) 优先：IDT 的逐样本损失
            L = self._loss_via_idt_api(idt, batch)
            if L is None:
                # 2) 退路：概率 + 交叉熵
                P = self._predict_proba_generic(idt, batch)        # 允许 batch 或 X
                _, y = self._as_numpy_X_y(batch)                    # y 可能不存在
                L = self._loss_from_proba(P, y)

            features = L.reshape(-1, 1)
            labels = np.full((features.shape[0],), is_member, dtype=int)
            return features, labels

        except Exception as e:
            print(f"特征提取失败: {e}")
            return None, None

    def train_attack_model(self, idt, member_batch, nonmember_batch):
        X_mem, y_mem = self.extract_features(idt, member_batch, is_member=1)
        X_non, y_non = self.extract_features(idt, nonmember_batch, is_member=0)
        if X_mem is None or X_non is None:
            raise RuntimeError("特征提取失败，无法训练 MIA。")

        X = np.vstack([X_mem, X_non])
        y = np.concatenate([y_mem, y_non])

        self.attack_model.fit(X, y)
        auc = roc_auc_score(y, self.attack_model.predict_proba(X)[:, 1])
        print(f"[MIA-loss] Attack model AUC: {auc:.4f}")
        return auc

    def infer(self, idt, sample_batch):
        X, _ = self.extract_features(idt, sample_batch, is_member=-1)
        if X is None:
            raise RuntimeError("特征提取失败：infer() 无法进行。")
        return self.attack_model.predict_proba(X)[:, 1]

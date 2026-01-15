from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import copy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings
from scipy.cluster.hierarchy import ClusterWarning
warnings.filterwarnings("ignore", category=ClusterWarning)

# --------------------------
# helpers
# --------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _batch_to_data_list(batch: Any) -> List[Any]:
    """
    Convert various batch formats to a list of per-graph PyG Data objects if possible.
    Priority:
      1) PyG Batch/Data: has to_data_list()
      2) python list/tuple: already list-like
      3) fallback: try to split a 'DummyBatch'-like object into Data list
    """
    if batch is None:
        return []

    if hasattr(batch, "to_data_list"):
        return list(batch.to_data_list())

    if isinstance(batch, (list, tuple)):
        return list(batch)

    # Fallback: try split DummyBatch -> list[Data]
    try:
        return _split_dummybatch_to_data_list(batch)
    except Exception as e:
        raise ValueError(
            "default_shadow_factory expects member_batch/nonmember_batch to be PyG Batch (with to_data_list), "
            "or a list of Data, or a DummyBatch-like object splittable into Data."
        ) from e


def _split_dummybatch_to_data_list(db: Any) -> List[Any]:
    """
    Split a DummyBatch-like object into a list of torch_geometric.data.Data (graph-level).
    Requirements:
      - db.x: [num_nodes_total, F]
      - db.y: [num_graphs] or [num_graphs, ...]
      - db.batch: [num_nodes_total] with graph id per node
      - optional db.edge_index: [2, num_edges_total] with global node indices
    """
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as e:
        raise RuntimeError("torch and torch_geometric are required to split DummyBatch.") from e

    if not (hasattr(db, "x") and hasattr(db, "y") and hasattr(db, "batch")):
        raise ValueError("Object does not look like DummyBatch (missing x/y/batch).")

    x = db.x
    y = db.y
    batch_vec = db.batch
    edge_index = getattr(db, "edge_index", None)

    # infer num graphs
    if hasattr(y, "numel"):
        num_graphs = int(y.numel())
    else:
        y = torch.as_tensor(y)
        num_graphs = int(y.numel())

    data_list: List[Data] = []
    for gid in range(num_graphs):
        node_mask = (batch_vec == gid)
        if node_mask.sum().item() == 0:
            continue

        # map global node indices -> local indices
        global_idx = node_mask.nonzero(as_tuple=False).view(-1)
        local_map = -torch.ones((x.size(0),), dtype=torch.long, device=x.device)
        local_map[global_idx] = torch.arange(global_idx.numel(), device=x.device)

        x_g = x[global_idx]
        y_g = y.view(-1)[gid].view(1)

        if edge_index is not None:
            src = edge_index[0]
            dst = edge_index[1]
            e_mask = node_mask[src] & node_mask[dst]
            e = edge_index[:, e_mask]
            e_local = torch.stack([local_map[e[0]], local_map[e[1]]], dim=0)
        else:
            e_local = None

        d = Data(x=x_g, y=y_g)
        if e_local is not None:
            d.edge_index = e_local
        data_list.append(d)

    if len(data_list) == 0:
        raise ValueError("Split resulted in empty data_list.")
    return data_list


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _infer_num_samples(batch: Any) -> int:
    if batch is None:
        return 0
    if hasattr(batch, "batch_size") and isinstance(getattr(batch, "batch_size"), int):
        return int(batch.batch_size)
    if hasattr(batch, "to_data_list"):
        return len(batch.to_data_list())
    try:
        return len(batch)
    except Exception as e:
        raise ValueError(
            "Cannot infer number of samples from batch. Provide batch_size / to_data_list() / __len__()."
        ) from e


def _split_data_list(data_list: List[Any], rng: np.random.Generator) -> Tuple[List[Any], List[Any]]:
    """Randomly split data_list into two halves."""
    idx = np.arange(len(data_list))
    rng.shuffle(idx)
    mid = len(idx) // 2
    a = [data_list[i] for i in idx[:mid]]
    b = [data_list[i] for i in idx[mid:]]
    return a, b


# --------------------------
# default shadow factory
# --------------------------

def _train_shadow_idt_from_target(
    target_idt: Any,
    train_batch: Any,
    *,
    values_source: Optional[Any] = None,
) -> Any:
    """
    Train a shadow IDT cloned from target_idt on train_batch.

    values_source:
      - None: will try `from model.idt import get_activations` and call get_activations(train_batch, model_or_int)
              where model_or_int is:
                * values_source if provided
                * else tries to use `target_idt.width` as an int (fallback)
      - int or torch.nn.Module: directly used as get_activations(train_batch, values_source)
    """
    shadow = copy.deepcopy(target_idt)

    # Get values for IDT.fit(batch, values, y)
    if values_source is None:
        # best-effort: use target_idt.width as int (matches your get_activations(int) path)
        values_source = getattr(target_idt, "width", 1)

    try:
        from model.idt import get_activations  # your project function
        values = get_activations(train_batch, values_source)
    except Exception as e:
        raise RuntimeError(
            "Failed to build IDT 'values' for shadow training. "
            "Make sure model.idt.get_activations is importable, or pass values_source explicitly."
        ) from e

    shadow.fit(train_batch, values, train_batch.y)
    return shadow


def _train_shadow_torch_model_from_target(
    target_model: Any,
    train_batch: Any,
    *,
    shadow_epochs: int = 5,
    shadow_lr: float = 1e-3,
    device: Optional[str] = None,
) -> Any:
    """
    Train a shadow torch model cloned from target_model on train_batch.
    Works with your GNN.forward() returning log_softmax (log-probabilities).
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError("PyTorch is required to train torch shadow models.") from e

    shadow = copy.deepcopy(target_model)
    shadow.train()

    if device is None:
        try:
            device = next(shadow.parameters()).device
        except Exception:
            device = torch.device("cpu")

    shadow.to(device)
    b = train_batch.to(device)

    opt = torch.optim.Adam(shadow.parameters(), lr=shadow_lr)

    for _ in range(int(shadow_epochs)):
        opt.zero_grad()
        out = shadow(b)
        # If model returns log-prob (like your GNN), use NLL. Otherwise, use CE.
        if hasattr(out, "dim") and out.dim() == 2:
            # heuristic: if values roughly negative and exp sums to ~1 -> log-prob
            loss = F.nll_loss(out, b.y) if out.max().item() <= 0.0 else F.cross_entropy(out, b.y)
        else:
            loss = F.cross_entropy(out, b.y)
        loss.backward()
        opt.step()

    shadow.eval()
    return shadow


def default_shadow_factory(
    target_model: Any,
    member_batch: Any,
    nonmember_batch: Any,
    *,
    num_shadows: int = 5,
    seed: int = 0,
    # for torch models
    shadow_epochs: int = 5,
    shadow_lr: float = 1e-3,
    device: Optional[str] = None,
    # for IDT models
    values_source: Optional[Any] = None,
):
    """
    A self-contained shadow_factory usable by both:
    - main(IDT) and
    - baselines(GNN)
    """
    try:
        import torch
        from torch_geometric.data import Batch  # type: ignore
    except Exception as e:
        raise RuntimeError("torch and torch_geometric are required for default_shadow_factory.") from e

    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1) decide a single device
    # -----------------------------
    if device is None:
        # safest default: CPU
        dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    # -----------------------------
    # 2) move Data objects to dev BEFORE batching
    # -----------------------------
    def _move_data_list_to_device(data_list, dev):
        moved = []
        for d in data_list:
            # PyG Data supports .to(device)
            try:
                moved.append(d.to(dev))
            except Exception:
                # fallback: manual move (rarely needed)
                d2 = d.clone()
                for k, v in d2:
                    if torch.is_tensor(v):
                        d2[k] = v.to(dev)
                moved.append(d2)
        return moved

    pool = _batch_to_data_list(member_batch) + _batch_to_data_list(nonmember_batch)
    if len(pool) < 2:
        raise ValueError("Not enough samples to build shadows (need >=2 total).")

    pool = _move_data_list_to_device(pool, dev)

    shadow_models: List[Any] = []
    shadow_member_batches: List[Iterable[Any]] = []
    shadow_nonmember_batches: List[Iterable[Any]] = []

    for s in range(int(num_shadows)):
        train_list, holdout_list = _split_data_list(pool, rng)
        if len(train_list) == 0 or len(holdout_list) == 0:
            mid = max(1, len(pool) // 2)
            train_list, holdout_list = pool[:mid], pool[mid:]

        # IMPORTANT: now train_list/holdout_list are already on the same device
        train_b = Batch.from_data_list(train_list)
        holdout_b = Batch.from_data_list(holdout_list)

        # -----------------------------
        # 3) train shadow model
        # -----------------------------
        if hasattr(target_model, "fit") and hasattr(target_model, "predict"):
            # IDT training should be on CPU in your implementation
            # so we move batches to CPU for IDT
            train_cpu = train_b.to("cpu")
            shadow = _train_shadow_idt_from_target(target_model, train_cpu, values_source=values_source)

            # keep batches CPU as well for querying IDT
            train_b = train_cpu
            holdout_b = holdout_b.to("cpu")

        else:
            # torch model (e.g., GNN) can be trained on dev
            shadow = _train_shadow_torch_model_from_target(
                target_model,
                train_b,
                shadow_epochs=shadow_epochs,
                shadow_lr=shadow_lr,
                device=str(dev),
            )

        shadow_models.append(shadow)
        shadow_member_batches.append([train_b])
        shadow_nonmember_batches.append([holdout_b])

    return shadow_models, shadow_member_batches, shadow_nonmember_batches



# --------------------------
# Shadow-MIA attacker
# --------------------------

@dataclass
class AttackDataset:
    X: np.ndarray
    y: np.ndarray


ShadowFactory = Callable[..., Tuple[Sequence[Any], Sequence[Iterable[Any]], Sequence[Iterable[Any]]]]


class ShadowMIAttacker:
    """
    Shadow-model MIA attacker with a loss-based-like API:

        attacker = ShadowMIAttacker(shadow_factory=default_shadow_factory, ...)
        stats = attacker.train_attack_model(target_model, member_batch, nonmember_batch, num_shadows=5, ...)
        print(stats["auc_target_transfer"])

    Returned stats:
      - auc_shadow_train: AUC on shadow training data (sanity)
      - auc_shadow_eval: AUC on shadow eval split (sanity)
      - auc_target_transfer: AUC on target_model for (member_batch vs nonmember_batch) (main metric)
    """

    def __init__(
        self,
        shadow_factory: Optional[ShadowFactory] = None,
        attack_model: Optional[LogisticRegression] = None,
        include_loss_feature: bool = True,
        max_samples_per_shadow: Optional[int] = None,
        random_state: int = 0,
    ):
        self.shadow_factory = shadow_factory
        self.attack_model = attack_model or LogisticRegression(max_iter=2000, random_state=random_state)
        self.include_loss_feature = include_loss_feature
        self.max_samples_per_shadow = max_samples_per_shadow
        self.random_state = random_state

    # -------- feature extraction --------

    def extract_features(self, model: Any, batch: Any, is_member: int) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(batch)
        else:
            pred_scores = model.predict(batch)
            pred_scores = _as_numpy(pred_scores)
            pred_proba = _sigmoid(pred_scores)

        pred_proba = _as_numpy(pred_proba)
        pred_feat = pred_proba.reshape(-1, 1) if pred_proba.ndim == 1 else pred_proba

        n = _infer_num_samples(batch)

        if self.include_loss_feature and hasattr(model, "compute_loss"):
            loss = _as_numpy(model.compute_loss(batch)).reshape(-1, 1)
            if loss.shape[0] != n:
                if loss.size == 1:
                    loss = np.full((n, 1), float(loss.item()))
                else:
                    raise ValueError(f"compute_loss returned {loss.shape[0]} entries, but batch has {n}.")
            X = np.concatenate([pred_feat, loss], axis=1)
        else:
            X = pred_feat

        y = np.full((n,), int(is_member), dtype=np.int64) if is_member in (0, 1) else np.full((n,), -1, dtype=np.int64)
        return X, y

    # -------- build dataset from shadows --------

    def _collect_for_one_shadow(self, model: Any, mem_batches: Iterable[Any], non_batches: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
        Xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for b in mem_batches:
            X_b, y_b = self.extract_features(model, b, is_member=1)
            Xs.append(X_b); ys.append(y_b)
        for b in non_batches:
            X_b, y_b = self.extract_features(model, b, is_member=0)
            Xs.append(X_b); ys.append(y_b)

        X = np.vstack(Xs) if Xs else np.zeros((0, 1), dtype=np.float32)
        y = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
        return X, y

    def build_attack_dataset(
        self,
        shadow_models: Sequence[Any],
        shadow_member_batches: Sequence[Iterable[Any]],
        shadow_nonmember_batches: Sequence[Iterable[Any]],
    ) -> AttackDataset:
        if not (len(shadow_models) == len(shadow_member_batches) == len(shadow_nonmember_batches)):
            raise ValueError("shadow_models, shadow_member_batches, shadow_nonmember_batches must have the same length.")

        rng = np.random.default_rng(self.random_state)
        X_all: List[np.ndarray] = []
        y_all: List[np.ndarray] = []

        for m, mem_it, non_it in zip(shadow_models, shadow_member_batches, shadow_nonmember_batches):
            X_s, y_s = self._collect_for_one_shadow(m, mem_it, non_it)

            if self.max_samples_per_shadow is not None and X_s.shape[0] > self.max_samples_per_shadow:
                mem_idx = np.where(y_s == 1)[0]
                non_idx = np.where(y_s == 0)[0]
                k = max(1, self.max_samples_per_shadow // 2)
                k_mem = min(k, mem_idx.size)
                k_non = min(k, non_idx.size)
                chosen = np.concatenate([
                    rng.choice(mem_idx, size=k_mem, replace=False) if k_mem > 0 else np.array([], dtype=int),
                    rng.choice(non_idx, size=k_non, replace=False) if k_non > 0 else np.array([], dtype=int),
                ])
                rng.shuffle(chosen)
                X_s, y_s = X_s[chosen], y_s[chosen]

            X_all.append(X_s); y_all.append(y_s)

        X = np.vstack(X_all) if X_all else np.zeros((0, 1), dtype=np.float32)
        y = np.concatenate(y_all) if y_all else np.zeros((0,), dtype=np.int64)
        return AttackDataset(X=X, y=y)

    # -------- training & inference --------

    def train_attack_model_from_shadows(
        self,
        shadow_models: Sequence[Any],
        shadow_member_batches: Sequence[Iterable[Any]],
        shadow_nonmember_batches: Sequence[Iterable[Any]],
        eval_split: float = 0.2,
    ) -> Dict[str, float]:
        ds = self.build_attack_dataset(shadow_models, shadow_member_batches, shadow_nonmember_batches)
        if ds.X.shape[0] == 0:
            raise ValueError("Empty attack dataset from shadows. Check your shadow_factory outputs.")

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(ds.X.shape[0])
        X, y = ds.X[idx], ds.y[idx]

        if eval_split and 0.0 < eval_split < 1.0:
            n_eval = max(1, int(round(X.shape[0] * eval_split)))
            X_eval, y_eval = X[:n_eval], y[:n_eval]
            X_train, y_train = X[n_eval:], y[n_eval:]
        else:
            X_train, y_train = X, y
            X_eval = y_eval = None

        self.attack_model.fit(X_train, y_train)

        out: Dict[str, float] = {}
        out["auc_shadow_train"] = float(roc_auc_score(y_train, self.attack_model.predict_proba(X_train)[:, 1]))
        if X_eval is not None:
            out["auc_shadow_eval"] = float(roc_auc_score(y_eval, self.attack_model.predict_proba(X_eval)[:, 1]))
        return out

    def train_attack_model(
        self,
        target_model: Any,
        member_batch: Any,
        nonmember_batch: Any,
        eval_split: float = 0.2,
        **shadow_kwargs: Any,
    ) -> Dict[str, float]:
        if self.shadow_factory is None:
            raise ValueError(
                "ShadowMIAttacker requires shadow_factory. Use default_shadow_factory or provide your own."
            )

        shadow_models, shadow_member_batches, shadow_nonmember_batches = self.shadow_factory(
            target_model, member_batch, nonmember_batch, **shadow_kwargs
        )

        stats = self.train_attack_model_from_shadows(
            shadow_models=shadow_models,
            shadow_member_batches=shadow_member_batches,
            shadow_nonmember_batches=shadow_nonmember_batches,
            eval_split=eval_split,
        )

        # main metric: transfer to the target model (member vs nonmember)
        try:
            X_mem, y_mem = self.extract_features(target_model, member_batch, is_member=1)
            X_non, y_non = self.extract_features(target_model, nonmember_batch, is_member=0)
            X_t = np.vstack([X_mem, X_non])
            y_t = np.concatenate([y_mem, y_non])
            p_t = self.attack_model.predict_proba(X_t)[:, 1]
            stats["auc_target_transfer"] = float(roc_auc_score(y_t, p_t))
        except Exception:
            stats["auc_target_transfer"] = float("nan")

        return stats

    def infer(self, target_model: Any, sample_batch: Any) -> np.ndarray:
        X, _ = self.extract_features(target_model, sample_batch, is_member=-1)
        return self.attack_model.predict_proba(X)[:, 1]

# visualize_boundary_gnn_vs_idt.py
from __future__ import annotations

import os
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from torch_geometric.data import Batch

# ====== your project imports ======
from data.dataset import data_label_skew
from model.client import GNNClient


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class BoundaryResult:
    client_id: int
    gnn_surrogate_train_fidelity: float
    gnn_surrogate_val_fidelity: float
    idt_surrogate_train_fidelity: float
    idt_surrogate_val_fidelity: float
    idt_tree_depth: int
    idt_n_leaves: int
    idt_used_features: int
    idt_total_features: int
    idt_feature_usage_ratio: float
    save_path: str


def to_cpu_batch(batch: Batch) -> Batch:
    return batch.to("cpu")


def pick_val_batch(client) -> Batch:
    """
    优先使用 train_val_batch（更接近“训练分布下的验证”），
    若不存在则尝试 val_batch。
    """
    if hasattr(client, "train_val_batch") and client.train_val_batch is not None:
        return to_cpu_batch(client.train_val_batch)
    if hasattr(client, "val_batch") and client.val_batch is not None:
        return to_cpu_batch(client.val_batch)
    raise AttributeError(
        f"Client {client.client_id} has neither train_val_batch nor val_batch."
    )


def get_graph_embeddings_and_preds(
    client,
    eval_batch: Batch
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回:
      H: (N, D) GNN图级表示（pooling后，分类头前）
      y_true: (N,)
      y_gnn: (N,) GNN预测
      y_idt: (N,) IDT预测
    """
    model = client.model
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        b_dev = eval_batch.to(device)
        H = model.get_graph_representation(b_dev).detach().cpu().numpy()  # (N, D)
        logits = model(b_dev)  # log_softmax
        y_gnn = logits.argmax(dim=1).detach().cpu().numpy()

    y_true = eval_batch.y.detach().cpu().numpy()
    y_idt = client.idt.predict(eval_batch)  # numpy

    return H, y_true, y_gnn, y_idt


def train_surrogates_on_2d(
    Z_train: np.ndarray,
    y_gnn_train: np.ndarray,
    y_idt_train: np.ndarray,
    seed: int = 42,
):
    """
    在2D空间拟合两个代理器：
      - GNN surrogate: MLP
      - IDT surrogate: DecisionTree
    """
    # GNN surrogate
    gnn_sur = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=2000,
            random_state=seed,
        ))
    ])
    gnn_sur.fit(Z_train, y_gnn_train)

    # IDT surrogate (使用树来保留离散切分特性)
    idt_sur = DecisionTreeClassifier(
        max_depth=6,     # 可调；太浅会过度简化，太深会过拟合
        min_samples_leaf=5,
        random_state=seed
    )
    idt_sur.fit(Z_train, y_idt_train)

    # train fidelity（对teacher labels）
    gnn_train_fit = accuracy_score(y_gnn_train, gnn_sur.predict(Z_train))
    idt_train_fit = accuracy_score(y_idt_train, idt_sur.predict(Z_train))

    return gnn_sur, idt_sur, gnn_train_fit, idt_train_fit


def make_meshgrid(Z: np.ndarray, grid_size: int = 300, pad_ratio: float = 0.08):
    x_min, x_max = Z[:, 0].min(), Z[:, 0].max()
    y_min, y_max = Z[:, 1].min(), Z[:, 1].max()

    dx = (x_max - x_min) * pad_ratio + 1e-6
    dy = (y_max - y_min) * pad_ratio + 1e-6

    xx, yy = np.meshgrid(
        np.linspace(x_min - dx, x_max + dx, grid_size),
        np.linspace(y_min - dy, y_max + dy, grid_size),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_boundaries(
    Z_plot: np.ndarray,
    y_true_plot: np.ndarray,
    gnn_sur,
    idt_sur,
    client_id: int,
    save_path: str,
    title_prefix: str = ""
):
    xx, yy, grid = make_meshgrid(Z_plot, grid_size=350)

    pred_gnn_grid = gnn_sur.predict(grid).reshape(xx.shape)
    pred_idt_grid = idt_sur.predict(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)

    # ---- Left: GNN surrogate boundary ----
    ax = axes[0]
    ax.contourf(xx, yy, pred_gnn_grid, alpha=0.25)
    ax.scatter(Z_plot[:, 0], Z_plot[:, 1], c=y_true_plot, s=12, alpha=0.8)
    ax.contour(xx, yy, pred_gnn_grid, linewidths=0.8)
    ax.set_title("GNN Boundary (2D surrogate)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # ---- Right: IDT surrogate boundary ----
    ax = axes[1]
    ax.contourf(xx, yy, pred_idt_grid, alpha=0.25)
    ax.scatter(Z_plot[:, 0], Z_plot[:, 1], c=y_true_plot, s=12, alpha=0.8)
    ax.contour(xx, yy, pred_idt_grid, linewidths=0.8)
    ax.set_title("IDT Boundary (2D surrogate)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if title_prefix:
        fig.suptitle(f"{title_prefix} | Client {client_id}", y=1.02, fontsize=12)
    else:
        fig.suptitle(f"Client {client_id}: GNN vs IDT Decision Boundary", y=1.02, fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def extract_idt_structure_stats(client) -> Dict[str, float]:
    """
    使用输出层决策树统计“有限特征/离散结构”指标
    """
    dt = client.idt.out_layer.dt
    tree = dt.tree_

    # sklearn中内部节点特征index，叶子为-2
    used = set(int(f) for f in tree.feature if f >= 0)

    n_features_total = int(dt.n_features_in_)
    n_used = len(used)

    stats = {
        "tree_depth": int(tree.max_depth),
        "n_leaves": int(tree.n_leaves),
        "used_features": int(n_used),
        "total_features": int(n_features_total),
        "feature_usage_ratio": float(n_used / max(1, n_features_total)),
    }
    return stats


# ---------------------------
# Main pipeline
# ---------------------------
def run(args):
    set_seed(args.seed)

    # 1) same non-iid setup as your main
    label_allocation = {
        0: [0, 1],
        1: [0],
        2: [0],
        3: [1],
        4: [0],
    }

    num_features, num_classes, train_loaders, val_loaders, train_val_batches, test_batches = data_label_skew(
        args.dataset,
        kfold=args.kfold,
        seed=args.seed,
        return_split=True,
        label_allocation=label_allocation,
        samples_per_label=args.samples_per_label
    )

    # 2) init clients
    clients = []
    for cid in range(args.kfold):
        bundle = (
            num_features, num_classes,
            train_loaders[cid], val_loaders[cid],
            train_val_batches[cid], test_batches[cid]
        )
        client = GNNClient(cid, bundle, args, device=cid % args.devices)
        clients.append(client)

    # 3) train + distill
    for client in clients:
        print(f"[Client {client.client_id}] Training GNN...")
        client.train(conv="GCN")
        print(f"[Client {client.client_id}] Distilling to IDT...")
        client.distill_to_idt(use_pred=True)

    # 4) visualize per client
    os.makedirs(args.out_dir, exist_ok=True)
    all_results: List[BoundaryResult] = []

    for client in clients:
        cid = client.client_id

        # 用 test batch 画图（可视化）
        plot_batch = to_cpu_batch(client.test_batch)

        # 用 val batch 做“泛化 fidelity”
        val_batch = pick_val_batch(client)

        # 4.1 train/plot split data
        H_plot, y_true_plot, y_gnn_plot, y_idt_plot = get_graph_embeddings_and_preds(client, plot_batch)

        # 4.2 val split data
        H_val, y_true_val, y_gnn_val, y_idt_val = get_graph_embeddings_and_preds(client, val_batch)

        # 4.3 shared 2D space: PCA仅在 plot/train 上拟合，再变换到 val
        pca = PCA(n_components=2, random_state=args.seed)
        Z_plot = pca.fit_transform(H_plot)
        Z_val = pca.transform(H_val)

        # 4.4 fit surrogates on plot/train split
        gnn_sur, idt_sur, gnn_train_fit, idt_train_fit = train_surrogates_on_2d(
            Z_plot, y_gnn_plot, y_idt_plot, seed=args.seed + cid
        )

        # 4.5 val fidelity (teacher consistency on val split)
        gnn_val_fit = accuracy_score(y_gnn_val, gnn_sur.predict(Z_val))
        idt_val_fit = accuracy_score(y_idt_val, idt_sur.predict(Z_val))

        # 4.6 plot
        save_path = os.path.join(
            args.out_dir, f"{args.dataset}_client{cid}_gnn_vs_idt_boundary.png"
        )
        plot_boundaries(
            Z_plot=Z_plot,
            y_true_plot=y_true_plot,
            gnn_sur=gnn_sur,
            idt_sur=idt_sur,
            client_id=cid,
            save_path=save_path,
            title_prefix=f"{args.dataset}"
        )

        # 4.7 idt structure stats
        s = extract_idt_structure_stats(client)

        r = BoundaryResult(
            client_id=cid,
            gnn_surrogate_train_fidelity=float(gnn_train_fit),
            gnn_surrogate_val_fidelity=float(gnn_val_fit),
            idt_surrogate_train_fidelity=float(idt_train_fit),
            idt_surrogate_val_fidelity=float(idt_val_fit),
            idt_tree_depth=s["tree_depth"],
            idt_n_leaves=s["n_leaves"],
            idt_used_features=s["used_features"],
            idt_total_features=s["total_features"],
            idt_feature_usage_ratio=s["feature_usage_ratio"],
            save_path=save_path,
        )
        all_results.append(r)

        print(
            f"[Client {cid}] saved -> {save_path}\n"
            f"  surrogate fidelity (teacher-consistency):\n"
            f"    GNN train={r.gnn_surrogate_train_fidelity:.3f}, val={r.gnn_surrogate_val_fidelity:.3f}\n"
            f"    IDT train={r.idt_surrogate_train_fidelity:.3f}, val={r.idt_surrogate_val_fidelity:.3f}\n"
            f"  IDT structure: depth={r.idt_tree_depth}, leaves={r.idt_n_leaves}, "
            f"used_feat={r.idt_used_features}/{r.idt_total_features} "
            f"({r.idt_feature_usage_ratio:.3f})"
        )

    # 5) summary csv
    csv_path = os.path.join(args.out_dir, f"{args.dataset}_boundary_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "client_id,"
            "gnn_surrogate_train_fidelity,gnn_surrogate_val_fidelity,"
            "idt_surrogate_train_fidelity,idt_surrogate_val_fidelity,"
            "idt_tree_depth,idt_n_leaves,idt_used_features,idt_total_features,idt_feature_usage_ratio,save_path\n"
        )
        for r in all_results:
            f.write(
                f"{r.client_id},"
                f"{r.gnn_surrogate_train_fidelity:.6f},{r.gnn_surrogate_val_fidelity:.6f},"
                f"{r.idt_surrogate_train_fidelity:.6f},{r.idt_surrogate_val_fidelity:.6f},"
                f"{r.idt_tree_depth},{r.idt_n_leaves},{r.idt_used_features},"
                f"{r.idt_total_features},{r.idt_feature_usage_ratio:.6f},{r.save_path}\n"
            )

    print(f"\nDone. Summary saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize GNN vs IDT boundary in shared 2D space")

    # same key args as your main
    parser.add_argument("--dataset", type=str, default="EMLC0")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--layer_depth", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--ccp_alpha", type=float, default=1e-3)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--samples_per_label", type=int, default=1000)

    # extra
    parser.add_argument("--out_dir", type=str, default="./boundary_vis")

    args = parser.parse_args()
    run(args)

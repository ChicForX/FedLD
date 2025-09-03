from scipy.sparse import coo_matrix
import networkx as nx
from networkx.generators import random_graphs, lattice, small, classic
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from collections import Counter, defaultdict
import os
import pickle


def _generate_EMLC(name, rng):
    u0 = (rng.random((13, 1)) < 0.5).astype(np.float32)
    u1 = np.ones((13, 1), dtype=np.float32)
    graph = nx.erdos_renyi_graph(13, 0.5, seed=rng.choice(2**32))
    adj = nx.adjacency_matrix(graph).toarray()
    edge_index = _adj_to_edge_index(adj)

    if name == 'EMLC0':
        has_more_than_half_u0 = (u0.sum() > 6)
        return Data(x=torch.tensor(u0), edge_index=edge_index, y=int(has_more_than_half_u0))
    elif name == 'EMLC1':
        has_lt_4_or_gt_9_neighbors = (edge_index[0].bincount() < 4) | (edge_index[0].bincount() > 9)
        return Data(x=torch.tensor(u1), edge_index=edge_index, y=int(has_lt_4_or_gt_9_neighbors.max()))
    elif name == 'EMLC2':
        degrees = adj.sum(1)
        has_gt_6_neighbors = degrees > 6
        more_than_half_neighbours_with_gt_6_neighbors = ((adj @ has_gt_6_neighbors) / degrees.clip(1)) > 0.5
        return Data(x=torch.tensor(u1), edge_index=edge_index,
                    y=(torch.tensor(more_than_half_neighbours_with_gt_6_neighbors).float().mean() > 0.5).long())
    else:
        raise ValueError(f"Unknown EMLC name: {name}")


def _generate_BAMultiShapes(rng):
    nb_node_ba = 40
    r = rng.choice(2)

    if r == 0:
        g = _generate_class1(nb_random_edges=1, nb_node_ba=nb_node_ba, rng=rng)
        return Data(x=torch.ones((len(g.nodes()), 1)), edge_index=_adj_to_edge_index(nx.adjacency_matrix(g).toarray()), y=0)
    else:
        g = _generate_class0(nb_random_edges=1, nb_node_ba=nb_node_ba, rng=rng)
        return Data(x=torch.ones((len(g.nodes()), 1)), edge_index=_adj_to_edge_index(nx.adjacency_matrix(g).toarray()), y=1)


def _generate_class1(nb_random_edges, nb_node_ba, rng):
    r = rng.choice(3)

    if r == 0:  # W + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 6 - 9, 1, seed=rng.choice(2**32))
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1, g2, nb_random_edges, rng)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12, g3, nb_random_edges)
    elif r == 1:  # W + H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 6 - 5, 1, seed=rng.choice(2**32))
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1, g2, nb_random_edges, rng)
        g3 = small.house_graph()
        g123 = _merge_graphs(g12, g3, nb_random_edges, rng)
    elif r == 2:  # H + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 5 - 9, 1, seed=rng.choice(2**32))
        g2 = small.house_graph()
        g12 = _merge_graphs(g1, g2, nb_random_edges, rng)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12, g3, nb_random_edges, rng)
    return g123


def _generate_class0(nb_random_edges, nb_node_ba, rng):
    r = rng.choice(4)

    if r > 3:
        return random_graphs.barabasi_albert_graph(nb_node_ba, 1, seed=rng.choice(2**32))
    if r == 0:  # W
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 6, 1, seed=rng.choice(2**32))
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1, g2, nb_random_edges, rng)
    if r == 1:  # H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 5, 1, seed=rng.choice(2**32))
        g2 = small.house_graph()
        g12 = _merge_graphs(g1, g2, nb_random_edges, rng)
    if r == 2:  # G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 9, 1, seed=rng.choice(2**32))
        g2 = lattice.grid_2d_graph(3, 3)
        g12 = _merge_graphs(g1, g2, nb_random_edges, rng)
    if r == 3:  # All
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 9 - 5 - 6, 1, seed=rng.choice(2**32))
        g2 = small.house_graph()
        g12 = _merge_graphs(g1, g2, nb_random_edges, rng)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12, g3, nb_random_edges)
        g4 = classic.wheel_graph(6)
        g12 = _merge_graphs(g123, g4, nb_random_edges)
    return g12


def _merge_graphs(g1, g2, nb_random_edges=1, rng=np.random.default_rng(0)):
    mapping = {n: max(g1.nodes()) + i + 1 for i, n in enumerate(g2.nodes())}
    g2 = nx.relabel_nodes(g2, mapping)
    g12 = nx.union(g1, g2)
    for _ in range(nb_random_edges):
        e1 = rng.choice(list(g1.nodes()))
        e2 = rng.choice(list(g2.nodes()))
        g12.add_edge(e1, e2)
    return g12


def _adj_to_edge_index(adj):
    matrix = coo_matrix(adj)
    return torch.stack([
        torch.tensor(matrix.row, dtype=torch.long),
        torch.tensor(matrix.col, dtype=torch.long)
    ])


def data(name, kfold=5, seed=0, return_split=False):
    rng = np.random.default_rng(seed)

    if name == 'BAMultiShapes':
        datalist = [_generate_BAMultiShapes(rng) for _ in range(8000)]
    elif name in ['EMLC0', 'EMLC1', 'EMLC2']:
        datalist = [_generate_EMLC(name, rng) for _ in range(5000)]
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    n_clients = kfold
    client_data = [[] for _ in range(n_clients)]

    for i, data in enumerate(datalist):
        client_data[i % n_clients].append(data)

    # 输出为每个 client 构造 Dataloader 和 Batch
    client_loaders = []
    client_batches = []

    for cdata in client_data:
        loader = DataLoader(cdata, batch_size=len(cdata), shuffle=True)
        batch = Batch.from_data_list(cdata)
        client_loaders.append(loader)
        client_batches.append(batch)

    num_features = datalist[0].x.shape[1]
    num_classes = len(torch.unique(torch.tensor([d.y for d in datalist])))

    if return_split:
        client_val_loaders = []
        client_train_val_batches = []
        client_test_batches = []

        for cdata in client_data:
            split = int(0.8 * len(cdata))
            train_val = cdata[:split]
            test = cdata[split:]

            val_loader = DataLoader(test, batch_size=len(test), shuffle=False)
            train_val_batch = Batch.from_data_list(train_val)
            test_batch = Batch.from_data_list(test)

            client_val_loaders.append(val_loader)
            client_train_val_batches.append(train_val_batch)
            client_test_batches.append(test_batch)

        return (
            num_features, num_classes,
            client_loaders,  # 训练 loader（全量）
            client_val_loaders,
            client_train_val_batches,
            client_test_batches
        )
    else:
        all_batch = Batch.from_data_list(datalist)
        return num_features, num_classes, DataLoader(datalist, batch_size=len(datalist)), all_batch


def data_label_skew(name, kfold=5, seed=0, return_split=False,
                              label_allocation=None, samples_per_label=200, verbose=True):
    rng = np.random.default_rng(seed)

    # 默认标签分配
    if label_allocation is None:
        label_allocation = {i: [i % 2] for i in range(kfold)}

    # 计算总的需求量
    all_needed_labels = set()
    max_needed_per_label = 0

    for client_labels in label_allocation.values():
        all_needed_labels.update(client_labels)
        max_needed_per_label = max(max_needed_per_label,
                                   len([cid for cid, labels in label_allocation.items()
                                        if any(l in client_labels for l in labels)]) * samples_per_label)

    print(f"Need labels: {sorted(all_needed_labels)}, max per label: {max_needed_per_label}")

    # 生成数据集 - 这里会应用数量限制！
    if name == 'BAMultiShapes':
        # 对于合成数据集，我们可以精确控制生成数量
        total_needed = sum(len(labels) * samples_per_label for labels in label_allocation.values())
        datalist = [_generate_BAMultiShapes(rng) for _ in range(min(8000, total_needed * 2))]
    elif name in ['EMLC0', 'EMLC1', 'EMLC2']:
        total_needed = sum(len(labels) * samples_per_label for labels in label_allocation.values())
        datalist = [_generate_EMLC(name, rng) for _ in range(min(5000, total_needed * 2))]
    elif name in ['AIDS', 'BZR', 'PROTEINS', 'Tox21_AhR_training', 'Tox21_HSE_training', 'SF-295']:
        # 对于TU数据集，使用限制加载
        datalist = _load_TUDataset(
            name,
            max_samples_per_label=max_needed_per_label,
            target_labels=list(all_needed_labels),
            cache_suffix=f"kfold{kfold}_samples{samples_per_label}"
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # 其余逻辑保持不变...
    client_data = [[] for _ in range(kfold)]
    client_label_counts = [{} for _ in range(kfold)]

    for i, data in enumerate(datalist):
        label = int(data.y)

        eligible_clients = [cid for cid, allowed_labels in label_allocation.items()
                            if label in allowed_labels]

        if not eligible_clients:
            continue

        target_client = eligible_clients[i % len(eligible_clients)]
        current_count = client_label_counts[target_client].get(label, 0)

        if current_count < samples_per_label:
            client_data[target_client].append(data)
            client_label_counts[target_client][label] = current_count + 1

    # 打印分布信息
    if verbose:
        for cid, cdata in enumerate(client_data):
            y_dist = Counter(int(d.y) for d in cdata)
            print(f"[Client {cid}] Label distribution: {dict(y_dist)}, Total: {len(cdata)}")

    # 构造结果...（保持原有逻辑）
    client_loaders = []
    client_batches = []

    for cdata in client_data:
        if len(cdata) == 0:
            print(f"Warning: Client has no data!")
            continue
        loader = DataLoader(cdata, batch_size=len(cdata), shuffle=True)
        batch = Batch.from_data_list(cdata)
        client_loaders.append(loader)
        client_batches.append(batch)

    num_features = datalist[0].x.shape[1] if datalist else 1  # 改为1避免0维度问题
    num_classes = len(torch.unique(torch.tensor([d.y for d in datalist]))) if datalist else 2  # 改为2避免0类别问题

    if return_split:
        client_val_loaders = []
        client_train_val_batches = []
        client_test_batches = []

        for cdata in client_data:
            if len(cdata) == 0:
                continue
            split = int(0.8 * len(cdata))
            train_val = cdata[:split]
            test = cdata[split:]

            val_loader = DataLoader(test, batch_size=len(test), shuffle=False)
            train_val_batch = Batch.from_data_list(train_val)
            test_batch = Batch.from_data_list(test)

            client_val_loaders.append(val_loader)
            client_train_val_batches.append(train_val_batch)
            client_test_batches.append(test_batch)

        return (
            num_features, num_classes,
            client_loaders,  # 训练 loader（全量）
            client_val_loaders,
            client_train_val_batches,
            client_test_batches
        )
    else:
        all_batch = Batch.from_data_list(datalist)
        return num_features, num_classes, DataLoader(datalist, batch_size=len(datalist)), all_batch


def data_label_ratio(name, kfold=5, seed=0, return_split=False,
                     label_ratio_allocation=None, samples_per_label=1000, verbose=True):
    """
    根据标签比例分配数据给客户端

    先计算需求，再生成/加载数据，最后分配

    Args:
        name: 数据集名称
        kfold: 客户端数量
        seed: 随机种子
        return_split: 是否返回训练/验证分割
        label_ratio_allocation: 字典，格式为 {client_id: {label: ratio, ...}, ...}
        samples_per_label: 每个客户端的基准样本数
        verbose: 是否打印详细信息
    """
    rng = np.random.default_rng(seed)

    # Step 1: 设置默认标签比例分配
    if label_ratio_allocation is None:
        label_ratio_allocation = {}
        for i in range(kfold):
            if i % 2 == 0:
                label_ratio_allocation[i] = {0: 0.8, 1: 0.2}
            else:
                label_ratio_allocation[i] = {0: 0.2, 1: 0.8}

    # 验证比例分配
    for client_id, ratios in label_ratio_allocation.items():
        total_ratio = sum(ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Client {client_id} ratios sum to {total_ratio}, should be 1.0")

    # Step 2: 计算每个客户端每个标签的精确需求
    client_label_needs = {}
    total_label_needs = defaultdict(int)
    all_labels = set()

    for client_id in range(kfold):
        ratios = label_ratio_allocation.get(client_id, {0: 0.5, 1: 0.5})
        client_label_needs[client_id] = {}

        for label, ratio in ratios.items():
            needed = max(1, int(samples_per_label * ratio))  # 至少1个样本
            client_label_needs[client_id][label] = needed
            total_label_needs[label] += needed
            all_labels.add(label)

    # 计算总预期样本数
    total_expected_samples = sum(total_label_needs.values())

    if verbose:
        print(f"=== 需求计算 ===")
        print(f"原始基准样本数: {samples_per_label}")
        print(f"标签数量: {len(all_labels)}")
        print(f"预期总样本数: {total_expected_samples}")
        print(f"每个标签总需求: {dict(total_label_needs)}")
        for client_id, needs in client_label_needs.items():
            total_client_samples = sum(needs.values())
            print(f"  客户端 {client_id}: {needs} (总计: {total_client_samples})")

    # Step 3: 生成/加载数据
    if verbose:
        print(f"\n=== 数据生成/加载 ===")

    # 先用一个较大的数来生成数据，然后根据实际情况调整
    if name == 'BAMultiShapes':
        datalist = [_generate_BAMultiShapes(rng) for _ in range(10000)]
    elif name in ['EMLC0', 'EMLC1', 'EMLC2']:
        datalist = [_generate_EMLC(name, rng) for _ in range(8000)]
    elif name in ['AIDS', 'BZR', 'PROTEINS', 'Tox21_AhR_training', 'Tox21_HSE_training', 'ENZYMES']:
        # 对于真实数据集，先加载全部看有多少
        datalist = _load_TUDataset(name, max_samples_per_label=None, target_labels=list(all_labels))
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    if len(datalist) == 0:
        raise ValueError(f"No data loaded for dataset {name}")

    # Step 4: 检查数据集实际大小和充足性
    actual_data_size = len(datalist)
    if verbose:
        print(f"数据集实际大小: {actual_data_size}")

    if actual_data_size < total_expected_samples:
        print(f"Warning: 数据集大小 {actual_data_size} 小于预期需求 {total_expected_samples}")
        print(f"将根据实际数据量按比例调整分配")

    # Step 5: 按标签分组数据并检查充足性
    label_pools = defaultdict(list)
    for data in datalist:
        label = int(data.y)
        if label in all_labels:
            label_pools[label].append(data)

    if verbose:
        print(f"各标签实际数据量:")
        for label in sorted(all_labels):
            available = len(label_pools[label])
            needed = total_label_needs[label]
            status = "✓" if available >= needed else "✗"
            print(f"  标签 {label}: 有 {available}, 需要 {needed} {status}")

    # 检查是否有足够数据
    for label in all_labels:
        if len(label_pools[label]) < total_label_needs[label]:
            ratio = len(label_pools[label]) / total_label_needs[label]
            print(f"Warning: 标签 {label} 数据不足，按比例 {ratio:.3f} 调整所有客户端需求")

            # 按比例调整所有客户端的该标签需求
            for client_id in range(kfold):
                if label in client_label_needs[client_id]:
                    old_need = client_label_needs[client_id][label]
                    new_need = max(1, int(old_need * ratio))
                    client_label_needs[client_id][label] = new_need

    # 打乱数据池
    for label in label_pools:
        rng.shuffle(label_pools[label])

    # Step 6: 精确分配数据
    if verbose:
        print(f"\n=== 数据分配 ===")

    client_data = [[] for _ in range(kfold)]
    label_pool_indices = {label: 0 for label in all_labels}

    # 按客户端和标签精确分配
    for client_id in range(kfold):
        if verbose:
            print(f"分配客户端 {client_id}:")

        for label, needed_count in client_label_needs[client_id].items():
            available_count = len(label_pools[label]) - label_pool_indices[label]
            actual_take = min(needed_count, available_count)

            if actual_take > 0:
                start_idx = label_pool_indices[label]
                end_idx = start_idx + actual_take
                taken_data = label_pools[label][start_idx:end_idx]
                client_data[client_id].extend(taken_data)
                label_pool_indices[label] += actual_take

            if verbose:
                print(f"  标签 {label}: 需要 {needed_count}, 实际分配 {actual_take}")

    # Step 7: 验证分配结果
    for client_id, cdata in enumerate(client_data):
        if len(cdata) == 0:
            raise ValueError(f"客户端 {client_id} 没有分配到任何数据！")

    if verbose:
        print(f"\n=== 分配结果 ===")
        for client_id, cdata in enumerate(client_data):
            label_counts = Counter(int(d.y) for d in cdata)
            total_count = len(cdata)
            ratios = {label: count / total_count for label, count in label_counts.items()}
            print(f"客户端 {client_id}: 总数={total_count}, 计数={dict(label_counts)}, 比例={ratios}")

    # Step 8: 构造返回结果
    client_loaders = []
    client_batches = []

    for cdata in client_data:
        loader = DataLoader(cdata, batch_size=len(cdata), shuffle=True)
        batch = Batch.from_data_list(cdata)
        client_loaders.append(loader)
        client_batches.append(batch)

    # 计算特征数和类别数
    num_features = datalist[0].x.shape[1] if datalist and hasattr(datalist[0], 'x') else 1
    num_classes = len(torch.unique(torch.tensor([d.y for d in datalist]))) if datalist else 2

    if return_split:
        client_val_loaders = []
        client_train_val_batches = []
        client_test_batches = []

        for cdata in client_data:
            # 80-20分割
            split_idx = max(1, int(0.8 * len(cdata)))
            train_data = cdata[:split_idx] if split_idx < len(cdata) else cdata[:-1]
            test_data = cdata[split_idx:] if split_idx < len(cdata) else cdata[-1:]

            # 确保都不为空
            if len(train_data) == 0:
                train_data = [cdata[0]]
            if len(test_data) == 0:
                test_data = [cdata[-1]]

            val_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
            train_batch = Batch.from_data_list(train_data)
            test_batch = Batch.from_data_list(test_data)

            client_val_loaders.append(val_loader)
            client_train_val_batches.append(train_batch)
            client_test_batches.append(test_batch)

        return (
            num_features, num_classes,
            client_loaders,
            client_val_loaders,
            client_train_val_batches,
            client_test_batches
        )
    else:
        all_batch = Batch.from_data_list(datalist)
        return num_features, num_classes, DataLoader(datalist, batch_size=len(datalist)), all_batch


def data_split_for_centralized(
    name,
    kfold=5,
    cv_split=0,
    seed=0,
    batch_size=32,
    val_batch_size=None,
    num_workers=0,
    pin_memory=True,
    shuffle_train=True,
    persistent_workers=None,
):
    """
    生成 centralized 训练/验证 DataLoader（小批次），并同时返回整合的 train_val_batch 与 test_batch 供蒸馏/可视化使用。

    Args:
        name: 数据集名称
        kfold: 用于划分 test fold 的份数
        cv_split: 选用第几个 fold 作为测试集
        seed: 随机种子
        batch_size: 训练集 batch 大小
        val_batch_size: 验证集 batch 大小（默认与训练一致）
        num_workers: DataLoader 的工作线程数（Windows 建议 0~4）
        pin_memory: 是否 pin memory（GPU 训练建议 True）
        shuffle_train: 训练集是否打乱
        persistent_workers: 是否常驻 worker（None 时按 num_workers>0 自动推断）

    Returns:
        num_features, num_classes, train_loader, val_loader, train_val_batch, test_batch
    """
    import numpy as np

    if val_batch_size is None:
        val_batch_size = batch_size

    rng = np.random.default_rng(seed)

    # —— 构造完整数据列表（与你原先一致）——
    if name == 'BAMultiShapes':
        datalist = [_generate_BAMultiShapes(rng) for _ in range(8000)]
    elif name in ['EMLC0', 'EMLC1', 'EMLC2']:
        datalist = [_generate_EMLC(name, rng) for _ in range(5000)]
    elif name in ['AIDS', 'BZR', 'PROTEINS', 'Tox21_AhR_training', 'Tox21_HSE_training', 'SF-295']:
        datalist = _load_TUDataset(name)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # —— k-fold 切分 test fold ——
    n_test = len(datalist) // kfold
    test_data = datalist[cv_split * n_test : (cv_split + 1) * n_test]
    train_val_data = datalist[:cv_split * n_test] + datalist[(cv_split + 1) * n_test:]

    # —— 从 train_val 再随机切出一个 val ——
    gen = torch.Generator().manual_seed(seed)
    train_len = len(train_val_data) - n_test
    val_len = n_test
    train_data, val_data = torch.utils.data.random_split(train_val_data, [train_len, val_len], generator=gen)

    # —— 小批次 DataLoader（关键信息）——
    # 为避免“只有 1 个 batch”的情况，这里做个 min 保护，最多取数据量大小
    train_bs = min(batch_size, max(1, len(train_data)))
    val_bs   = min(val_batch_size, max(1, len(val_data)))

    if persistent_workers is None:
        persistent_workers = (num_workers > 0)

    train_loader = DataLoader(
        train_data, batch_size=train_bs, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_data, batch_size=val_bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    # —— 保持向后兼容：仍返回整合的大 Batch ——
    train_val_batch = Batch.from_data_list(train_val_data)
    test_batch      = Batch.from_data_list(test_data)

    num_features = datalist[0].x.shape[1]
    num_classes  = len(torch.unique(torch.tensor([d.y for d in datalist])))

    return num_features, num_classes, train_loader, val_loader, train_val_batch, test_batch


def _load_TUDataset(name, root='D:/workspace/python/fed_logic_distill/data',
                               max_samples_per_label=None, target_labels=None,
                               use_cache=True, cache_suffix=""):
    """
    支持按标签数量限制的TUDataset加载器

    Args:
        name: 数据集名称
        root: 数据根目录
        max_samples_per_label: 每个标签最多加载多少个样本
        target_labels: 只加载指定标签的数据，None表示加载所有标签
        use_cache: 是否使用缓存
        cache_suffix: 缓存文件后缀，用于区分不同的加载配置
    """

    # 构建缓存文件名
    cache_parts = [name]
    if max_samples_per_label:
        cache_parts.append(f"max{max_samples_per_label}")
    if target_labels:
        cache_parts.append(f"labels{'_'.join(map(str, sorted(target_labels)))}")
    if cache_suffix:
        cache_parts.append(cache_suffix)

    cache_file = os.path.join(root, f"{'_'.join(cache_parts)}_limited.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached limited dataset: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Loading dataset {name} with limits: max_per_label={max_samples_per_label}, target_labels={target_labels}")

    prefix = os.path.join(root, name)
    path = lambda fname: os.path.join(prefix, f'{name}_{fname}.txt')

    # 首先只读取标签，确定我们需要哪些数据
    with open(path('graph_labels'), 'r') as f:
        graph_labels = [int(line.strip()) for line in f]

    # 标签归一化
    unique_labels = sorted(set(graph_labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    normalized_labels = [label_map[label] for label in graph_labels]

    # 确定目标标签
    if target_labels is None:
        target_labels = set(range(len(unique_labels)))
    else:
        target_labels = set(target_labels)

    # 找出我们需要加载的图的索引
    label_counts = Counter()
    selected_graph_ids = []

    for graph_id, label in enumerate(normalized_labels):
        if label in target_labels:
            if max_samples_per_label is None or label_counts[label] < max_samples_per_label:
                selected_graph_ids.append(graph_id)
                label_counts[label] += 1

                # 如果所有标签都已经收集够了，提前退出
                if (max_samples_per_label and
                        all(label_counts[l] >= max_samples_per_label for l in target_labels)):
                    break

    print(f"Selected {len(selected_graph_ids)} graphs from {len(graph_labels)} total")
    print(f"Label distribution: {dict(label_counts)}")

    # 如果没有需要的数据，直接返回
    if not selected_graph_ids:
        return []

    # 现在只加载我们需要的数据
    selected_graph_ids_set = set(selected_graph_ids)

    # 读取其他必要文件
    with open(path('graph_indicator'), 'r') as f:
        graph_indicator = [int(line.strip()) for line in f]

    with open(path('A'), 'r') as f:
        edges_raw = [list(map(int, line.split(','))) for line in f]

    # 读取节点特征
    has_node_labels = os.path.exists(path('node_labels'))
    has_node_attrs = os.path.exists(path('node_attributes'))

    if has_node_labels:
        with open(path('node_labels'), 'r') as f:
            node_labels = torch.tensor([int(line.strip()) for line in f],
                                       dtype=torch.float).unsqueeze(1)
    if has_node_attrs:
        with open(path('node_attributes'), 'r') as f:
            node_attrs = torch.tensor([[float(x) for x in line.strip().split(',')] for line in f],
                                      dtype=torch.float)

    # 合并特征
    if has_node_labels and has_node_attrs:
        all_features = torch.cat([node_attrs, node_labels], dim=1)
    elif has_node_attrs:
        all_features = node_attrs
    elif has_node_labels:
        all_features = node_labels
    else:
        raise ValueError("No node features found in TUDataset.")

    # 构建节点到图的映射，只关心我们需要的图
    needed_nodes = set()
    graph_nodes = defaultdict(list)

    for node_idx, graph_id in enumerate(graph_indicator):
        graph_id_0based = graph_id - 1
        if graph_id_0based in selected_graph_ids_set:
            graph_nodes[graph_id_0based].append(node_idx)
            needed_nodes.add(node_idx)

    # 只处理涉及到我们需要节点的边
    graph_edges = defaultdict(list)
    node_to_graph = {}

    for node_idx, graph_id in enumerate(graph_indicator):
        if node_idx in needed_nodes:
            node_to_graph[node_idx] = graph_id - 1

    for u, v in edges_raw:
        u_idx, v_idx = u - 1, v - 1
        if u_idx in needed_nodes and v_idx in needed_nodes:
            graph_id = node_to_graph[u_idx]
            if node_to_graph[v_idx] == graph_id:  # 验证边的两个节点属于同一个图
                graph_edges[graph_id].append((u_idx, v_idx))

    # 构建数据对象，按照selected_graph_ids的顺序
    datalist = []
    for graph_id in selected_graph_ids:
        node_indices = graph_nodes[graph_id]

        # 节点特征
        node_feats = all_features[node_indices]

        # 构建节点重映射
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_indices)}

        # 处理边
        edges_for_graph = graph_edges[graph_id]
        if not edges_for_graph:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            remapped_edges = [[node_id_map[u], node_id_map[v]] for u, v in edges_for_graph]
            edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()

        # 创建数据对象
        label = torch.tensor(normalized_labels[graph_id], dtype=torch.long)
        datalist.append(Data(x=node_feats, edge_index=edge_index, y=label))

    # 缓存结果
    if use_cache:
        print(f"Caching limited dataset to: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(datalist, f)

    return datalist




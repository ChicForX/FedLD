import re
import matplotlib
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Batch


class IDT:
    def __init__(self, width=10, sample_size=20, layer_depth=3, max_depth=5, ccp_alpha=.0):
        self.width = width
        self.sample_size = sample_size
        self.layer_depth = layer_depth
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.layer = None
        self.out_layer = None

    def fit(self, batch, values, y, sample_weight=None):
        # 统一到 CPU，避免 torch 操作（如 torch.isin）出现 cuda:0 / cpu 不一致
        batch = batch.to("cpu")

        if self.sample_size is None or self.sample_size > batch.num_graphs:
            self.sample_size = batch.num_graphs
        self.layer = []

        # 构建全局邻接与节点特征（CPU → NumPy）
        adj_full = to_scipy_sparse_matrix(batch.edge_index, num_nodes=batch.x.shape[0]).tobsr()
        x = batch.x.numpy()
        node_sample_weight = sample_weight[batch.batch] if sample_weight is not None else None
        depth_indices = [x.shape[1]]

        for i, value in enumerate(values):
            new_layers = []
            depth_indices_new = []

            for _ in range(self.width):
                new_layers.append(IDTInnerLayer(self.layer_depth, depth_indices))

                # Step 1: graph-level sampling（在 CPU）
                graph_indices = np.random.choice(batch.num_graphs, size=self.sample_size, replace=False)
                graph_indices = torch.tensor(graph_indices, dtype=torch.long)  # CPU tensor

                # Step 2: 由选中图索引构建子 batch
                small_batch_list = batch.index_select(graph_indices)
                small_batch = Batch.from_data_list(small_batch_list)

                # Step 3: 找到原 batch 中属于这些图的节点索引
                node_mask = torch.isin(batch.batch, graph_indices)  # 两者均在 CPU
                node_indices = node_mask.nonzero(as_tuple=False).view(-1)

                # Step 4: 计算子 batch 的邻接矩阵
                adj = to_scipy_sparse_matrix(small_batch.edge_index, num_nodes=small_batch.x.shape[0]).tobsr()

                # Step 5: 训练该子层
                new_layers[-1].fit(
                    x[node_indices],
                    value[node_indices],
                    adj,
                    node_sample_weight[node_indices] if node_sample_weight is not None else None
                )

                depth_indices_new.append(2 ** (self.layer_depth + 1) - 1)

            # 该层输出拼接到特征
            x_new = np.concatenate([layer.predict(x, adj_full) for layer in new_layers], axis=1)
            x = np.concatenate([x, x_new], axis=1)
            self.layer += new_layers
            depth_indices += depth_indices_new

        # 最终分类层
        self.out_layer = IDTFinalLayer(
            self.max_depth,
            self.ccp_alpha,
            depth_indices
        )
        self.out_layer.fit(x, batch.batch, y)
        return self

    def predict(self, batch):
        x = batch.x.cpu().numpy()
        adj = to_scipy_sparse_matrix(batch.edge_index.cpu(), num_nodes=x.shape[0]).tobsr()
        for i in range(len(self.layer) // self.width):
            x_new = np.concatenate(
                [self.layer[j].predict(x, adj) for j in range(i * self.width, (i + 1) * self.width)],
                axis=1
            )
            x = np.concatenate([x, x_new], axis=1)
        return self.out_layer.predict(x, batch.batch.cpu())

    def predict_proba(self, batch):
        """
        返回图级 (num_graphs, num_classes) 概率。
        与 predict 相同的特征构建流程，但末端调用输出层 dt 的 predict_proba。
        期望 batch 为 PyG 的 Batch（含 .x/.edge_index/.batch）
        """
        # 逐层构造节点级特征
        x = batch.x.cpu().numpy()
        adj = to_scipy_sparse_matrix(batch.edge_index.cpu(), num_nodes=x.shape[0]).tobsr()
        for i in range(len(self.layer) // self.width):
            x_new = np.concatenate(
                [self.layer[j].predict(x, adj) for j in range(i * self.width, (i + 1) * self.width)],
                axis=1
            )
            x = np.concatenate([x, x_new], axis=1)

        # 图级池化 → 输出层概率
        X_graph = _pool(x, batch.batch.cpu())
        return self.out_layer.dt.predict_proba(X_graph)

    def compute_loss_per_sample(self, batch, eps=1e-12):
        """
        返回逐样本交叉熵损失 (num_graphs,)：
          -log P_y；若 y 的类别在训练中未出现，则回退为 -log max_c P_c
        """
        P = self.predict_proba(batch)  # (N, C)
        y = batch.y.cpu().numpy().astype(int)
        classes = self.out_layer.dt.classes_
        class_to_col = {c: i for i, c in enumerate(classes)}
        losses = np.empty(len(y), dtype=float)
        for i, yi in enumerate(y):
            j = class_to_col.get(yi, None)
            if j is None:  # 未见过类别 → 无标签代理损失
                losses[i] = -np.log(P[i].max() + eps)
            else:
                losses[i] = -np.log(P[i, j] + eps)
        return losses

    def accuracy(self, batch):
        prediction = self.predict(batch)
        return (prediction == batch.y.cpu().numpy()).mean()

    def prune(self):
        relevant = self.out_layer._relevant()
        for i, layer in enumerate(self.layer[::-1]):
            _relevant = {index for (depth, index) in relevant if depth == len(self.layer) - i - 1}
            layer._prune_irrelevant(_relevant)
            layer._remove_redunant()
            relevant |= layer._relevant()
        return self

    def save_image(self, path):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(
            ncols=self.width, nrows=len(self.layer) // self.width + 1,
            figsize=(4 * self.width, 4 * (len(self.layer) // self.width + 1))
        )
        for i, layer in enumerate(self.layer):
            layer.plot(axs[i // self.width, i % self.width], i)
        self.out_layer.plot(axs[-1, 0], len(self.layer))
        for i in range(1, self.width):
            plt.delaxes(axs[-1, i])
        fig.savefig(path)
        plt.close(fig)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(
            ncols=self.width, nrows=len(self.layer) // self.width + 1,
            figsize=(4 * self.width, 4 * (len(self.layer) // self.width + 1))
        )
        for i, layer in enumerate(self.layer):
            layer.plot(axs[i // self.width, i % self.width], i)
        self.out_layer.plot(axs[-1, 0], len(self.layer))
        for i in range(1, self.width):
            plt.delaxes(axs[-1, i])
        plt.show()

    def fidelity(self, batch, model):
        prediction = self.predict(batch)
        with torch.no_grad():
            device = next(model.parameters()).device
            model_pred = model(batch.to(device)).argmax(dim=1).detach().cpu().numpy()
        return (prediction == model_pred).mean()

    def f1_score(self, batch, average='macro'):
        prediction = self.predict(batch)
        return f1_score(batch.y.cpu().numpy(), prediction, average=average)

    def save_output_layer_spacious(self, path, figsize=(14, 10)):
        """
        保存输出层决策树，使用开阔的间距
        """
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        if self.out_layer is None or self.out_layer.dt is None:
            print(f"No output layer decision tree to save for {path}")
            return

        try:
            fig, ax = plt.subplots(figsize=figsize)
            plot_tree(
                self.out_layer.dt,
                ax=ax,
                filled=True,
                rounded=True,
                fontsize=10,
                proportion=False,
                impurity=False,
                precision=2
            )
            ax.set_title("IDT Output Layer Decision Tree", fontsize=14, fontweight='bold', pad=20)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f"Output layer decision tree saved to: {path}")
        except Exception as e:
            print(f"Error saving output layer for {path}: {e}")
            if 'fig' in locals():
                plt.close(fig)

    def plot_output_layer_spacious(self):
        """
        显示输出层决策树，使用开阔的间距
        """
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        if self.out_layer is None or self.out_layer.dt is None:
            print("No output layer decision tree to display")
            return

        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            plot_tree(
                self.out_layer.dt,
                ax=ax,
                filled=True,
                rounded=True,
                fontsize=10,
                proportion=False,
                impurity=False,
                precision=2
            )
            ax.set_title("IDT Output Layer Decision Tree", fontsize=14, fontweight='bold', pad=20)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
            plt.show()
        except Exception as e:
            print(f"Error displaying output layer: {e}")
            if 'fig' in locals():
                plt.close(fig)


class IDTInnerLayer:
    def __init__(self, max_depth, depth_indices):
        self.max_depth = max_depth
        self.depth_indices = [index for index in depth_indices]
        self.n_features_in = sum(depth_indices)
        self.dt = None
        self.leaf_indices = None
        self.leaf_values = None
        self.leaf_formulas = None

    def fit(self, x, y, adj, sample_weight=None):
        if self.n_features_in == 0:
            x = np.ones((x.shape[0], 1))
        x_neigh = adj @ x
        deg = adj.sum(axis=1)
        x = np.asarray(np.concatenate([
            x, x_neigh, x_neigh / deg.clip(1e-6, None)
        ], axis=1))
        self.dt = DecisionTreeRegressor(max_depth=self.max_depth, splitter='random')
        self.dt.fit(x, y, sample_weight=sample_weight)
        leaves = _leaves(self.dt.tree_)
        leaf_values = [self.dt.tree_.value[i, :, 0] for i in leaves]
        if len(leaves) == 1:
            combinations = [{0}]
        else:
            clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
            clustering.fit(leaf_values)
            combinations = _agglomerate_labels(len(leaves), clustering)
        self.leaf_indices = np.array([
            leaves.index(i) if i in leaves else -1
            for i in range(self.dt.tree_.node_count)
        ])
        self.leaf_values = np.array([
            [i in combination for combination in combinations]
            for i in self.leaf_indices if i != -1
        ])
        self.leaf_formulas = [
            [i for (i, b) in enumerate(self.leaf_values[j]) if b]
            for j in range(len(self.leaf_values))
        ]
        return self

    def predict(self, x, adj=None):
        if self.n_features_in == 0:
            x = np.ones((x.shape[0], 1))
        x_neigh = adj @ x
        x = np.asarray(np.concatenate([
            x, x_neigh, x_neigh / (adj.sum(axis=1)).clip(1e-6, None)
        ], axis=1))
        pred = self.dt.apply(x)
        return self.leaf_values[self.leaf_indices[pred]]

    def fit_predict(self, x, y, adj=None):
        self.fit(x, y, adj)
        return self.predict(x, adj)

    def _relevant(self):
        return {_feature_depth_index(feature, self.depth_indices) for feature in self.dt.tree_.feature if feature != -2}

    def _prune_irrelevant(self, relevant):
        self.leaf_formulas = [
            [i for i in formulas if i in relevant]
            for formulas in self.leaf_formulas
        ]

    def _remove_redunant(self):
        flag = True
        while flag:
            flag = False
            for parent, (left, right) in enumerate(zip(self.dt.tree_.children_left, self.dt.tree_.children_right)):
                if self.leaf_indices[left] != -1 and self.leaf_indices[right] != -1 and left != -1 and right != -1 and \
                        self.leaf_formulas[self.leaf_indices[left]] == self.leaf_formulas[self.leaf_indices[right]]:
                    self._merge_leaves(parent, left, right)
                    flag = True
                    break

    def _merge_leaves(self, parent, left, right):
        self.dt.tree_.children_left[parent] = -1
        self.dt.tree_.children_right[parent] = -1
        self.leaf_indices[parent] = self.leaf_indices[left]
        self.leaf_indices[left] = -1
        self.leaf_indices[right] = -1
        self.dt.tree_.feature[parent] = -2
        self.dt.tree_.threshold[parent] = -2

    def plot(self, ax, n=0):
        from sklearn.tree import plot_tree
        plot_tree(self.dt, ax=ax)
        leaf_counter = 0
        for obj in ax.properties()['children']:
            if type(obj) == matplotlib.text.Annotation:
                obj.set_fontsize(8)
                txt = obj.get_text().splitlines()[0]
                match = re.match(r'X\[(\d+)\] <= (\d+\.\d+)', txt)
                if match:
                    feature, threshold = match.groups()
                    feature = int(feature)
                    formula = _feature_formula(feature, self.depth_indices)
                    threshold = float(threshold)
                    if feature < self.n_features_in:
                        obj.set_text(fr'$I{formula} > 0$')
                    elif feature < 2 * self.n_features_in:
                        obj.set_text(fr'$A{formula} > {int(threshold)}$')
                    elif feature < 3 * self.n_features_in:
                        obj.set_text(fr'$A{formula} > {threshold}$')
                else:
                    valid_leaf_indices = [i for i in self.leaf_indices if i != -1]
                    if leaf_counter < len(valid_leaf_indices):
                        leaf_index = valid_leaf_indices[leaf_counter]
                        try:
                            formula = self.leaf_formulas[leaf_index]
                            if formula:
                                txt = r"$" + r", ".join([fr"M_{{{i}}}^{{{n}}}" for i in formula]) + r"$"
                            else:
                                txt = r"$\mathrm{Leaf}$"
                        except (IndexError, TypeError) as e:
                            print(f"Exception during plot: {e}")
                            txt = r"$\mathrm{Leaf}$"
                        obj.set_text(txt)
                    else:
                        obj.set_text(r"$\mathrm{Leaf}$")
                    leaf_counter += 1


class IDTFinalLayer:
    def __init__(self, max_depth, ccp_alpha, depth_indices):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.depth_indices = [index for index in depth_indices]
        self.n_features_in = sum(depth_indices)
        self.dt = None
        self.leaf_indices = None
        self.leaf_values = None

    def fit(self, x, batch, y, sample_weight=None):
        x = _pool(x, batch)
        self.dt = DecisionTreeClassifier(max_depth=self.max_depth, ccp_alpha=self.ccp_alpha)
        self.dt.fit(x, y, sample_weight=sample_weight)

    def predict(self, x, batch):
        x = _pool(x, batch)
        return self.dt.predict(x)

    def fit_predict(self, x, batch, y):
        self.fit(x, batch, y)
        return self.predict(x, batch)

    def _relevant(self):
        return {_feature_depth_index(feature, self.depth_indices) for feature in self.dt.tree_.feature if feature != -2}

    def plot(self, ax, n=0):
        from sklearn.tree import plot_tree
        plot_tree(self.dt, ax=ax)
        for obj in ax.properties()['children']:
            if type(obj) == matplotlib.text.Annotation:
                obj.set_fontsize(8)
                txt = obj.get_text().splitlines()[0]
                match = re.match(r'X\[(\d+)\] <= (\d+\.\d+)', txt)
                if match:
                    feature, threshold = match.groups()
                    feature = int(feature)
                    formula = _feature_formula(feature, self.depth_indices)
                    threshold = float(threshold)
                    if feature < self.n_features_in:
                        obj.set_text(fr'$1{formula} > {threshold}$')
                    else:
                        obj.set_text(fr'$1{formula} > {int(threshold)}$')
                if 'True' in txt:
                    obj.set_text(txt.replace('True', 'False'))
                elif 'False' in txt:
                    obj.set_text(txt.replace('False', 'True'))


def _pool(x, batch):
    # 保证 batch 在 CPU，与 x 张量一致
    batch = batch.cpu()
    return np.concatenate([
        global_mean_pool(torch.tensor(x), batch).cpu().numpy(),
        global_add_pool(torch.tensor(x), batch).cpu().numpy()
    ], axis=1)


def _agglomerate_labels(n_labels, clustering):
    agglomerated_features = [{i} for i in range(n_labels)]
    for i, j in clustering.children_:
        agglomerated_features.append(agglomerated_features[i] | agglomerated_features[j])
    return agglomerated_features


def _leaves(tree, node_id=0):
    if tree.children_left[node_id] == tree.children_right[node_id] and \
            (node_id in tree.children_left or node_id in tree.children_right or node_id == 0):
        return [node_id]
    else:
        left = _leaves(tree, tree.children_left[node_id])
        right = _leaves(tree, tree.children_right[node_id])
        return left + right


def get_activations(batch, model_or_int):
    if isinstance(model_or_int, int):
        return _get_values_int(batch, model_or_int)
    if isinstance(model_or_int, torch.nn.Module):
        return _get_values_model(batch, model_or_int)
    raise ValueError(f"Unknown type {type(model_or_int)}")


def _get_values_int(batch, int):
    labels = batch.y.cpu().numpy().max() + 1
    values = []
    for datapoint in batch.to_data_list():
        onehot = np.eye(labels)[datapoint.y.cpu().numpy()]
        values.append(onehot.repeat(datapoint.x.shape[0], axis=0))
    values = np.concatenate(values, axis=0).squeeze()
    values = [values] * int
    return values


def _get_values_model(batch, model):
    values = []
    device = next(model.parameters()).device
    hook = model.act.register_forward_hook(
        lambda mod, inp, out: values.append(out.detach().cpu().numpy().squeeze())
    )
    with torch.no_grad():
        model(batch.to(device))   # 触发 hook；不需要对输出再 .numpy()
    hook.remove()
    return values[:-1]


def _feature_formula(index, depth_indices):
    depth, index = _feature_depth_index(index, depth_indices)
    if depth == -1:
        return fr'U_{{{index}}}'
    else:
        return fr'\chi_{{{index}}}^{{{depth}}}'


def _feature_depth_index(index, depth_indices):
    index = index % sum(depth_indices)
    depth = -1
    for i in depth_indices:
        if index < i:
            return depth, index
        index -= i
        depth += 1

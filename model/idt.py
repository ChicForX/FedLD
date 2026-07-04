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
from joblib import Parallel, delayed


def _fit_one_inner_layer(layer_depth, depth_indices, x, value,
                         node_indices, adj, node_sample_weight, min_leaf=1):
    layer = IDTInnerLayer(layer_depth, depth_indices, min_leaf=min_leaf)
    layer.fit(
        x[node_indices],
        value[node_indices],
        adj,
        node_sample_weight[node_indices] if node_sample_weight is not None else None,
    )
    return layer


def _max_abs_corr_against_selected(candidate_features, selected_features, eps=1e-12):
    if selected_features is None or len(selected_features) == 0:
        return 0.0

    cand = np.asarray(candidate_features, dtype=np.float64)
    if cand.ndim == 1:
        cand = cand.reshape(-1, 1)

    selected = np.concatenate(selected_features, axis=1).astype(np.float64)
    if selected.ndim == 1:
        selected = selected.reshape(-1, 1)

    cand_std = cand.std(axis=0)
    sel_std = selected.std(axis=0)
    cand = cand[:, cand_std > eps]
    selected = selected[:, sel_std > eps]

    if cand.shape[1] == 0 or selected.shape[1] == 0:
        return 0.0

    cand = cand - cand.mean(axis=0, keepdims=True)
    selected = selected - selected.mean(axis=0, keepdims=True)
    cand = cand / (cand.std(axis=0, keepdims=True) + eps)
    selected = selected / (selected.std(axis=0, keepdims=True) + eps)

    corr = np.abs(cand.T @ selected) / max(1, cand.shape[0] - 1)
    if corr.size == 0:
        return 0.0
    return float(np.nanmax(corr))


def _candidate_layer_score(layer, feature_block, selected_feature_blocks,
                           r2_threshold=0.01, diversity_threshold=0.05):
    r2 = max(0.0, float(getattr(layer, "fit_r2_", 0.0)))
    max_corr = _max_abs_corr_against_selected(feature_block, selected_feature_blocks)
    diversity = 1.0 - max_corr
    score = r2 * max(0.0, diversity)
    usable = (r2 >= r2_threshold) and (diversity >= diversity_threshold)

    return {
        "score": float(score),
        "r2": float(r2),
        "diversity": float(diversity),
        "max_corr": float(max_corr),
        "usable": bool(usable),
    }


class IDT:
    def __init__(self, width=10, sample_size=20, layer_depth=3, max_depth=5, ccp_alpha=.0,
                 min_leaf_inner=1, min_leaf_final=1,
                 adaptive_width=False, width_min=4, width_step=4, width_patience=2,
                 adapt_r2_threshold=0.01, adapt_diversity_threshold=0.05,
                 adapt_target_gain_threshold=0.002,
                 verbose_adaptive_width=False):
        self.width = width
        self.sample_size = sample_size
        self.layer_depth = layer_depth
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.min_leaf_inner = min_leaf_inner
        self.min_leaf_final = min_leaf_final
        self.adaptive_width = adaptive_width
        self.width_min = max(1, int(width_min))
        self.width_step = max(1, int(width_step))
        self.width_patience = max(1, int(width_patience))
        self.adapt_r2_threshold = float(adapt_r2_threshold)
        self.adapt_diversity_threshold = float(adapt_diversity_threshold)
        self.adapt_target_gain_threshold = float(adapt_target_gain_threshold)
        self.verbose_adaptive_width = bool(verbose_adaptive_width)
        self.layer = None
        self.out_layer = None
        self.layer_widths = []
        self.adaptive_width_history = []

    def fit(self, batch, values, y, sample_weight=None):
        batch = batch.to("cpu")

        if self.sample_size is None or self.sample_size > batch.num_graphs:
            self.sample_size = batch.num_graphs
        self.layer = []
        self.layer_widths = []
        self.adaptive_width_history = []

        adj_full = to_scipy_sparse_matrix(batch.edge_index, num_nodes=batch.x.shape[0]).tobsr()
        x = batch.x.numpy()
        node_sample_weight = sample_weight[batch.batch] if sample_weight is not None else None
        depth_indices = [x.shape[1]]

        for i, value in enumerate(values):
            if self.adaptive_width:
                new_layers, x_new, layer_history = self._fit_one_value_adaptive_width(
                    layer_id=i,
                    batch=batch,
                    x=x,
                    value=value,
                    depth_indices=depth_indices,
                    adj_full=adj_full,
                    node_sample_weight=node_sample_weight,
                    y_graph=y,
                )
                effective_width = len(new_layers)
                depth_indices_new = [2 ** (self.layer_depth + 1) - 1] * effective_width
                self.adaptive_width_history.append(layer_history)

                if self.verbose_adaptive_width:
                    print(
                        f"[IDT] Adaptive width layer={i}: "
                        f"effective_width={effective_width}, "
                        f"candidate_budget={self.width}, "
                        f"mean_r2={layer_history.get('mean_r2', 0.0):.4f}, "
                        f"mean_diversity={layer_history.get('mean_diversity', 0.0):.4f}"
                    )
            else:
                samples = []
                for _ in range(self.width):
                    graph_indices = np.random.choice(batch.num_graphs, size=self.sample_size, replace=False)
                    graph_indices = torch.tensor(graph_indices, dtype=torch.long)

                    small_batch = Batch.from_data_list(batch.index_select(graph_indices))
                    node_mask = torch.isin(batch.batch, graph_indices)
                    node_indices = node_mask.nonzero(as_tuple=False).view(-1).numpy()
                    adj = to_scipy_sparse_matrix(small_batch.edge_index, num_nodes=small_batch.x.shape[0]).tobsr()
                    samples.append((node_indices, adj))

                new_layers = Parallel(n_jobs=2, prefer="threads")(
                    delayed(_fit_one_inner_layer)(
                        self.layer_depth,
                        depth_indices,
                        x,
                        value,
                        node_indices,
                        adj,
                        node_sample_weight,
                        self.min_leaf_inner,
                    )
                    for node_indices, adj in samples
                )

                x_new = np.concatenate([layer.predict(x, adj_full) for layer in new_layers], axis=1)
                depth_indices_new = [2 ** (self.layer_depth + 1) - 1] * self.width

            x = np.concatenate([x, x_new], axis=1)
            self.layer += new_layers
            self.layer_widths.append(len(new_layers))
            depth_indices += depth_indices_new

        if self.layer_widths:
            self.width = max(self.layer_widths)

        self.out_layer = IDTFinalLayer(
            self.max_depth,
            self.ccp_alpha,
            depth_indices,
            min_leaf=self.min_leaf_final,
        )
        self.out_layer.fit(x, batch.batch, y)
        return self

    def _temporary_final_score(self, x_node, batch_index, y_graph):
        y_arr = np.asarray(y_graph).reshape(-1)
        if len(y_arr) == 0:
            return 0.0

        Xg = _pool(x_node, batch_index)
        try:
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                ccp_alpha=self.ccp_alpha,
                min_samples_leaf=self.min_leaf_final,
                min_samples_split=max(2, 2 * self.min_leaf_final),
            )
            clf.fit(Xg, y_arr)
            pred = clf.predict(Xg)
            return float((pred == y_arr).mean())
        except Exception:
            return 0.0

    def _fit_one_value_adaptive_width(self, layer_id, batch, x, value, depth_indices,
                                      adj_full, node_sample_weight, y_graph):
        selected_layers = []
        selected_feature_blocks = []
        selected_stats = []

        best_layers = None
        best_feature_blocks = None
        best_stats = None
        best_score = -1.0

        max_width = max(self.width_min, int(self.width))
        no_gain_rounds = 0
        total_candidates = 0
        base_score = self._temporary_final_score(x, batch.batch.cpu(), y_graph)

        while len(selected_layers) < max_width:
            remaining = max_width - len(selected_layers)
            cur_step = min(self.width_step, remaining)

            samples = []
            for _ in range(cur_step):
                graph_indices = np.random.choice(batch.num_graphs, size=self.sample_size, replace=False)
                graph_indices = torch.tensor(graph_indices, dtype=torch.long)
                small_batch = Batch.from_data_list(batch.index_select(graph_indices))
                node_mask = torch.isin(batch.batch, graph_indices)
                node_indices = node_mask.nonzero(as_tuple=False).view(-1).numpy()
                adj = to_scipy_sparse_matrix(small_batch.edge_index, num_nodes=small_batch.x.shape[0]).tobsr()
                samples.append((node_indices, adj))

            candidate_layers = Parallel(n_jobs=2, prefer="threads")(
                delayed(_fit_one_inner_layer)(
                    self.layer_depth,
                    depth_indices,
                    x,
                    value,
                    node_indices,
                    adj,
                    node_sample_weight,
                    self.min_leaf_inner,
                )
                for node_indices, adj in samples
            )

            total_candidates += len(candidate_layers)

            candidate_records = []
            for layer in candidate_layers:
                feature_block = layer.predict(x, adj_full)
                stat = _candidate_layer_score(
                    layer,
                    feature_block,
                    selected_feature_blocks,
                    r2_threshold=self.adapt_r2_threshold,
                    diversity_threshold=self.adapt_diversity_threshold,
                )
                candidate_records.append((stat["score"], layer, feature_block, stat))

            candidate_records.sort(key=lambda t: t[0], reverse=True)
            for _, layer, feature_block, stat in candidate_records:
                selected_layers.append(layer)
                selected_feature_blocks.append(feature_block)
                selected_stats.append(stat)
                if len(selected_layers) >= max_width:
                    break

            x_candidate = np.concatenate([x] + selected_feature_blocks, axis=1)
            current_score = self._temporary_final_score(x_candidate, batch.batch.cpu(), y_graph)

            if len(selected_layers) < self.width_min:
                continue

            if best_layers is None:
                best_layers = list(selected_layers)
                best_feature_blocks = list(selected_feature_blocks)
                best_stats = list(selected_stats)
                best_score = current_score
                no_gain_rounds = 0
                continue

            gain = current_score - best_score
            if gain >= self.adapt_target_gain_threshold:
                best_layers = list(selected_layers)
                best_feature_blocks = list(selected_feature_blocks)
                best_stats = list(selected_stats)
                best_score = current_score
                no_gain_rounds = 0
            else:
                no_gain_rounds += 1

            if no_gain_rounds >= self.width_patience:
                break

        if best_layers is not None:
            selected_layers = best_layers
            selected_feature_blocks = best_feature_blocks
            selected_stats = best_stats

        if len(selected_layers) == 0:
            graph_indices = np.random.choice(batch.num_graphs, size=self.sample_size, replace=False)
            graph_indices = torch.tensor(graph_indices, dtype=torch.long)
            small_batch = Batch.from_data_list(batch.index_select(graph_indices))
            node_mask = torch.isin(batch.batch, graph_indices)
            node_indices = node_mask.nonzero(as_tuple=False).view(-1)
            adj = to_scipy_sparse_matrix(small_batch.edge_index, num_nodes=small_batch.x.shape[0]).tobsr()

            layer = _fit_one_inner_layer(
                self.layer_depth, depth_indices, x, value,
                node_indices, adj, node_sample_weight, self.min_leaf_inner
            )
            feature_block = layer.predict(x, adj_full)
            stat = _candidate_layer_score(
                layer,
                feature_block,
                [],
                r2_threshold=self.adapt_r2_threshold,
                diversity_threshold=self.adapt_diversity_threshold,
            )
            selected_layers = [layer]
            selected_feature_blocks = [feature_block]
            selected_stats = [stat]
            total_candidates += 1
            best_score = self._temporary_final_score(
                np.concatenate([x, feature_block], axis=1), batch.batch.cpu(), y_graph
            )

        x_new = np.concatenate(selected_feature_blocks, axis=1)
        mean_r2 = float(np.mean([s["r2"] for s in selected_stats])) if selected_stats else 0.0
        mean_div = float(np.mean([s["diversity"] for s in selected_stats])) if selected_stats else 0.0
        min_support = int(min([getattr(l, "min_leaf_support_", 0) for l in selected_layers])) if selected_layers else 0

        history = {
            "layer_id": int(layer_id),
            "effective_width": int(len(selected_layers)),
            "candidate_budget": int(max_width),
            "total_candidates_trained": int(total_candidates),
            "base_target_score": float(base_score),
            "selected_target_score": float(best_score),
            "mean_r2": mean_r2,
            "mean_diversity": mean_div,
            "min_inner_leaf_support": min_support,
        }

        return selected_layers, x_new, history

    def _forward_inner_layers(self, batch):
        batch = batch.to("cpu")
        x = batch.x.cpu().numpy()
        adj = to_scipy_sparse_matrix(batch.edge_index.cpu(), num_nodes=x.shape[0]).tobsr()

        cursor = 0
        layer_widths = self.layer_widths if getattr(self, "layer_widths", None) else [
            self.width] * (len(self.layer) // max(1, self.width))
        for w in layer_widths:
            x_new = np.concatenate(
                [self.layer[j].predict(x, adj) for j in range(cursor, cursor + w)],
                axis=1
            )
            x = np.concatenate([x, x_new], axis=1)
            cursor += w
        return x, batch.batch.cpu()

    def predict(self, batch):
        x, batch_index = self._forward_inner_layers(batch)
        return self.out_layer.predict(x, batch_index)

    def predict_proba(self, batch):
        x, batch_index = self._forward_inner_layers(batch)
        X_graph = _pool(x, batch_index)
        return self.out_layer.dt.predict_proba(X_graph)

    def compute_loss_per_sample(self, batch, eps=1e-12):
        P = self.predict_proba(batch)
        y = batch.y.cpu().numpy().astype(int)
        classes = self.out_layer.dt.classes_
        class_to_col = {c: i for i, c in enumerate(classes)}
        losses = np.empty(len(y), dtype=float)
        for i, yi in enumerate(y):
            j = class_to_col.get(yi, None)
            if j is None:
                losses[i] = -np.log(P[i].max() + eps)
            else:
                losses[i] = -np.log(P[i, j] + eps)
        return losses

    def compute_loss(self, batch, eps=1e-12):
        return self.compute_loss_per_sample(batch, eps=eps)

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

    def _plot_layers(self, show=False, path=None):
        import matplotlib.pyplot as plt

        layer_widths = getattr(self, "layer_widths", None)
        if not layer_widths:
            layer_widths = [self.width] * (len(self.layer) // max(1, self.width))

        ncols = max(1, max(layer_widths) if layer_widths else self.width)
        nrows = len(layer_widths) + 1

        fig, axs = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(4 * ncols, 4 * nrows),
            squeeze=False,
        )

        offset = 0
        for row, row_width in enumerate(layer_widths):
            for col in range(ncols):
                if col < row_width and offset + col < len(self.layer):
                    self.layer[offset + col].plot(axs[row, col], offset + col)
                else:
                    plt.delaxes(axs[row, col])
            offset += row_width

        self.out_layer.plot(axs[-1, 0], len(self.layer))
        for col in range(1, ncols):
            plt.delaxes(axs[-1, col])

        if path is not None:
            fig.savefig(path)
            plt.close(fig)
        elif show:
            plt.show()

    def save_image(self, path):
        self._plot_layers(path=path)

    def plot(self):
        self._plot_layers(show=True)

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
    def __init__(self, max_depth, depth_indices, min_leaf=1):
        self.max_depth = max_depth
        self.depth_indices = [index for index in depth_indices]
        self.n_features_in = sum(depth_indices)
        self.min_leaf = min_leaf
        self.dt = None
        self.leaf_indices = None
        self.leaf_values = None
        self.leaf_formulas = None
        self.fit_mse_ = None
        self.fit_r2_ = None
        self.min_leaf_support_ = None

    def fit(self, x, y, adj, sample_weight=None):
        if self.n_features_in == 0:
            x = np.ones((x.shape[0], 1))
        x_neigh = adj @ x
        deg = adj.sum(axis=1)
        x = np.asarray(np.concatenate([
            x, x_neigh, x_neigh / deg.clip(1e-6, None)
        ], axis=1))
        self.dt = DecisionTreeRegressor(
            max_depth=self.max_depth,
            splitter='random',
            min_samples_leaf=self.min_leaf,
            min_samples_split=max(2, 2 * self.min_leaf),
        )
        self.dt.fit(x, y, sample_weight=sample_weight)

        pred_y = self.dt.predict(x)
        y_arr = np.asarray(y, dtype=np.float64)
        pred_arr = np.asarray(pred_y, dtype=np.float64)
        self.fit_mse_ = float(np.mean((y_arr - pred_arr) ** 2))
        denom = float(np.mean((y_arr - y_arr.mean(axis=0, keepdims=True)) ** 2)) + 1e-12
        self.fit_r2_ = float(1.0 - self.fit_mse_ / denom)

        leaves = _leaves(self.dt.tree_)
        self.min_leaf_support_ = int(np.min(self.dt.tree_.n_node_samples[leaves])) if len(leaves) > 0 else 0
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
    def __init__(self, max_depth, ccp_alpha, depth_indices, min_leaf=1):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.depth_indices = [index for index in depth_indices]
        self.n_features_in = sum(depth_indices)
        self.min_leaf = min_leaf
        self.dt = None
        self.leaf_indices = None
        self.leaf_values = None

    def fit(self, x, batch, y, sample_weight=None):
        x = _pool(x, batch)
        self.dt = DecisionTreeClassifier(
            max_depth=self.max_depth,
            ccp_alpha=self.ccp_alpha,
            min_samples_leaf=self.min_leaf,
            min_samples_split=max(2, 2 * self.min_leaf),
        )
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
        model(batch.to(device))
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

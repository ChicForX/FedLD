import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from model.gnn import GNN
from model.idt import IDT, get_activations


class GNNClient:
    def __init__(self, client_id, data_bundle, args, device=0):
        self.client_id = client_id
        self.args = args
        if isinstance(device, int):
            self.device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.num_features, self.num_classes = data_bundle[0], data_bundle[1]
        self.train_loader, self.val_loader = data_bundle[2], data_bundle[3]
        self.train_val_batch, self.test_batch = data_bundle[4], data_bundle[5]

        # 计算类别权重（与之前一致）：len(y)/(C * bincount)
        bincount = torch.bincount(self.train_val_batch.y, minlength=self.num_classes)
        self.class_weight = (len(self.train_val_batch) / (self.num_classes * bincount.float())).to(self.device)

        self.model = None
        self.idt = None

    @torch.no_grad()
    def _eval_loader(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weight)
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * batch.y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return avg_loss, acc

    def train(self, conv="GCN"):
        # 初始化模型
        self.model = GNN(
            num_features=self.num_features,
            num_classes=self.num_classes,
            layers=self.args.layers,
            dim=self.args.dim,
            activation=self.args.activation,
            conv=conv,
            pool=self.args.pooling,
            lr=self.args.lr,
            weight=self.class_weight,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weight)

        # 早停配置
        patience = 10
        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0

        max_epochs = getattr(self.args, "max_steps", 1000)

        for epoch in range(1, max_epochs + 1):
            # ---- Train ----
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch)
                loss = criterion(logits, batch.y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.y.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

            train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)

            # ---- Validation ----
            val_loss, val_acc = self._eval_loader(self.val_loader)

            # 打印（每个 epoch 打印一次；如需稀疏打印可加条件）
            if (epoch + 1) % 100 == 0:
                print(f"[Client {self.client_id}] Epoch {epoch}/{max_epochs} "
                      f"- Train loss: {train_loss:.4f}, acc: {train_acc:.4f} "
                      f"| Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # 恢复最佳权重并设为 eval
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    @torch.no_grad()
    def _adaptive_capacity(self, idt_batch):
        import numpy as np

        a = self.args
        w_min = getattr(a, "adapt_w_min", 8)
        w_max = getattr(a, "adapt_w_max", 32)
        k_ratio_min = getattr(a, "adapt_k_ratio_min", 0.001)
        k_ratio_max = getattr(a, "adapt_k_ratio_max", 0.005)

        self.model.eval()
        dev = next(self.model.parameters()).device
        b = idt_batch.to(dev)

        H = self.model.get_graph_representation(b).detach().cpu().numpy().astype(np.float64)
        H = H - H.mean(axis=0, keepdims=True)
        if H.shape[0] < 2 or np.allclose(H, 0.0):
            srank = 1
        else:
            sv = np.linalg.svd(H, compute_uv=False)
            sv = sv[sv > 1e-12]
            if len(sv) == 0:
                srank = 1
            else:
                energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
                srank = int(np.searchsorted(energy, 0.99) + 1)

        try:
            n_graphs = int(idt_batch.num_graphs)
        except Exception:
            n_graphs = int(b.y.size(0)) if hasattr(b, "y") else 100

        k_min = max(2, int(round(k_ratio_min * n_graphs)))
        k_max = max(k_min + 1, int(round(k_ratio_max * n_graphs)))

        logits = self.model.out(self.model.get_graph_representation(b))
        z = logits.detach().cpu().numpy().astype(np.float64)
        if z.ndim == 2 and z.shape[1] >= 2:
            sz = np.sort(z, axis=1)
            margins = sz[:, -1] - sz[:, -2]
            m0 = float(margins.mean())
            sd = float(margins.std())
            if sd < 1e-8:
                frac = 0.5
            else:
                frac = 1.0 / (1.0 + np.exp(-(m0 / (sd + 1e-8) - 1.0)))
        else:
            frac = 0.5

        k_final = int(round(k_min + (k_max - k_min) * frac))
        k_final = max(k_min, min(k_max, k_final))
        k_inner = max(1, k_final // 2)

        support_cap = max(w_min, n_graphs // max(1, 2 * k_final))
        w_max_local = int(max(w_min, min(w_max, support_cap)))

        print(f"[Client {self.client_id}] Adaptive capacity budget: "
              f"srank={srank} (diagnostic only), n_graphs={n_graphs}, "
              f"candidate_width=[{w_min},{w_max_local}], "
              f"k_range=[{k_min},{k_max}], frac={frac:.3f} -> "
              f"k_inner={k_inner}, k_final={k_final}")

        return w_max_local, k_inner, k_final

    def distill_to_idt(self, use_pred=False):
        print(f"IDT Parameters: width={self.args.width}, sample_size={self.args.sample_size}, "
              f"layer_depth={self.args.layer_depth}, max_depth={self.args.max_depth}, "
              f"ccp_alpha={self.args.ccp_alpha}")
        idt_batch = self.train_val_batch
        with torch.no_grad():
            values = get_activations(idt_batch, self.model)
            if use_pred:
                y = idt_batch.y.detach().cpu().numpy()
            else:
                dev = next(self.model.parameters()).device
                y = self.model(idt_batch.to(dev)).argmax(dim=-1).detach().cpu().numpy()
                idt_batch = idt_batch.to("cpu")

        _width = self.args.width
        _min_leaf_inner = getattr(self.args, "min_leaf_inner", 1)
        _min_leaf_final = getattr(self.args, "min_leaf_final", 1)
        if getattr(self.args, "adaptive_capacity", False):
            _width, _min_leaf_inner, _min_leaf_final = self._adaptive_capacity(idt_batch)

        use_adaptive_width = bool(getattr(self.args, "adaptive_capacity", False))

        self.idt = IDT(
            width=_width,
            sample_size=self.args.sample_size,
            layer_depth=self.args.layer_depth,
            max_depth=self.args.max_depth,
            ccp_alpha=self.args.ccp_alpha,
            min_leaf_inner=_min_leaf_inner,
            min_leaf_final=_min_leaf_final,
            adaptive_width=use_adaptive_width,
            width_min=getattr(self.args, "adapt_w_min", 8),
            width_step=getattr(self.args, "adapt_width_step", 4),
            width_patience=getattr(self.args, "adapt_width_patience", 2),
            adapt_r2_threshold=getattr(self.args, "adapt_r2_threshold", 0.01),
            adapt_diversity_threshold=getattr(self.args, "adapt_diversity_threshold", 0.05),
            adapt_target_gain_threshold=getattr(self.args, "adapt_target_gain_threshold", 0.002),
            verbose_adaptive_width=getattr(self.args, "adapt_width_verbose", True),
        ).fit(idt_batch, values, y=y)

        if use_adaptive_width and hasattr(self.idt, "adaptive_width_history"):
            eff_widths = [h.get("effective_width", 0) for h in self.idt.adaptive_width_history]
            print(f"[Client {self.client_id}] Distillation-selected effective widths: {eff_widths}")

        return self.idt

    def evaluate(self):
        result = {}
        if self.model:
            with torch.no_grad():
                pred = self.model(self.test_batch).argmax(dim=-1)
                result['model_acc'] = (pred == self.test_batch.y).float().mean().item()
        if self.idt:
            pred = self.idt.predict(self.test_batch)
            result['idt_acc'] = (pred == self.test_batch.y.numpy()).mean()
            result['fidelity'] = self.idt.fidelity(self.test_batch, self.model)
            result['idt_f1_macro'] = self.idt.f1_score(self.test_batch)
        return result

    def get_model_state_dict(self):
        """返回模型的状态字典"""
        if self.model is not None:
            return self.model.state_dict()
        else:
            return None

    def set_model_state_dict(self, state_dict):
        """设置模型的状态字典"""
        if self.model is not None and state_dict is not None:
            self.model.load_state_dict(state_dict)

    def get_model_parameters(self):
        """返回模型参数的列表"""
        if self.model is not None:
            return [param.data for param in self.model.parameters()]
        else:
            return []

    def set_model_parameters(self, parameters):
        """设置模型参数"""
        if self.model is not None and parameters:
            for param, new_param in zip(self.model.parameters(), parameters):
                param.data.copy_(new_param)

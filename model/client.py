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

    def distill_to_idt(self, use_pred=False):
        print(f"IDT Parameters: width={self.args.width}, sample_size={self.args.sample_size}, "
              f"layer_depth={self.args.layer_depth}, max_depth={self.args.max_depth}, "
              f"ccp_alpha={self.args.ccp_alpha}")
        with torch.no_grad():
            values = get_activations(self.train_val_batch, self.model)
            if use_pred:
                y = self.train_val_batch.y.detach().cpu().numpy()
            else:
                y = self.model(self.train_val_batch).argmax(dim=-1).detach().cpu().numpy()

        self.idt = IDT(
            width=self.args.width,
            sample_size=self.args.sample_size,
            layer_depth=self.args.layer_depth,
            max_depth=self.args.max_depth,
            ccp_alpha=self.args.ccp_alpha
        ).fit(self.train_val_batch, values, y=y)

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
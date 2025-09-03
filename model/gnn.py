import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Tanh, Sigmoid, Sequential
from torch_geometric.nn import GCNConv, GINConv, GraphNorm, global_mean_pool, global_add_pool
from lightning import LightningModule
from sklearn.metrics import f1_score


def gin_conv(in_channels, out_channels):
    return GINConv(
        Sequential(
            Linear(in_channels, 2 * in_channels),
            GraphNorm(2 * in_channels),
            ReLU(),
            Linear(2 * in_channels, out_channels)
        )
    )


class GNN(LightningModule):
    def __init__(self, num_features, num_classes, layers=3, dim=64,
                 conv="GCN", activation="ReLU", pool="mean", lr=1e-4, weight=None):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = Linear(num_features, dim)

        if activation == "ReLU":
            self.act = ReLU()
        elif activation == "Tanh":
            self.act = Tanh()
        elif activation == "Sigmoid":
            self.act = Sigmoid()
        else:
            raise ValueError(f"Unknown activation {activation}")

        if conv == "GCN":
            self.conv_fn = GCNConv
        elif conv == "GIN":
            self.conv_fn = gin_conv
        else:
            raise ValueError(f"Unknown convolution type {conv}")

        self.conv_layers = torch.nn.ModuleList([
            self.conv_fn(dim, dim) for _ in range(layers)
        ])
        self.norms = torch.nn.ModuleList([
            GraphNorm(dim) for _ in range(layers)
        ])

        self.out = Sequential(
            Linear(dim, dim),
            self.act,
            Linear(dim, num_classes)
        )

        self.loss = torch.nn.NLLLoss(weight=weight)

    def forward_layers(self, data):
        """
        返回每一层的节点表示列表：X^{(0)}, X^{(1)}, ..., X^{(L)}
        不含 graph-level pooling 与最后分类层。
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        out_per_layer = [x]  # X^{(0)}

        for conv, norm in zip(self.conv_layers, self.norms):
            x = self.act(x + norm(conv(x, edge_index)))
            out_per_layer.append(x)  # X^{(1)}, X^{(2)}, ...

        return out_per_layer  # list of tensor [X^{(0)}, X^{(1)}, ..., X^{(L)}]

    def forward_layers_with_pool(self, data):
        """
        返回每一层的节点表示 + graph-level pooling 表示。
        用于完整支持 X^{(0)} ~ X^{(L+1)}
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        out_per_layer = [x]  # X^{(0)}

        for conv, norm in zip(self.conv_layers, self.norms):
            x = self.act(x + norm(conv(x, edge_index)))
            out_per_layer.append(x)

        if self.hparams.pool == "mean":
            pooled = global_mean_pool(x, batch)
        elif self.hparams.pool == "add":
            pooled = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method {self.hparams.pool}")

        out_per_layer.append(pooled)  # X^{(L+1)}
        return out_per_layer

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        for conv, norm in zip(self.conv_layers, self.norms):
            x = x + norm(conv(x, edge_index))
            x = self.act(x)

        if self.hparams.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.hparams.pool == "add":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method {self.hparams.pool}")

        x = self.out(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean()
        f1 = f1_score(batch.y.cpu(), out.argmax(dim=1).cpu(), average='macro')
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_f1_macro", f1)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(out, batch.y)
        acc = (out.argmax(dim=1) == batch.y).float().mean()
        f1 = f1_score(batch.y.cpu(), out.argmax(dim=1).cpu(), average='macro')
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1_macro", f1, prog_bar=True)
        return loss

    # for semigraphfl
    def get_graph_representation(self, data):
        """
        获取图级表示，用于SemiGraphFL中的相似度计算
        返回pooling后但未经过最终分类层的特征表示
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 嵌入层
        x = self.embedding(x)

        # 图卷积层
        for conv, norm in zip(self.conv_layers, self.norms):
            x = x + norm(conv(x, edge_index))
            x = self.act(x)

        # 图级池化
        if self.hparams.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.hparams.pool == "add":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method {self.hparams.pool}")

        # 返回池化后的图表示（不经过最终的分类层）
        return x

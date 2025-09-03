import copy
import torch
import numpy as np
from model.idt import IDT


class Server:
    def __init__(self, clients):
        self.clients = clients
        self.global_model = clients[0].get_model_state_dict()  # 初始化为第一个 client 的参数
        self.client_idts = []  # 初始化为空

    def aggregate_parameters(self):
        """
        联邦平均客户端 GNN 参数
        """
        state_dicts = [client.get_model_state_dict() for client in self.clients]
        avg_state_dict = {}

        for key in state_dicts[0].keys():
            avg_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)

        self.global_model = avg_state_dict
        return avg_state_dict

    def distribute_parameters(self, state_dict):
        for client in self.clients:
            client.set_model_state_dict(state_dict)

    def predict(self, batch):
        """
        使用客户端 IDT 的预测进行 majority voting。
        """
        if not self.client_idts:
            raise ValueError("No client IDTs available. Have you called aggregate()?")

        preds = []
        for idt in self.client_idts:
            pred = idt.predict(batch)
            preds.append(pred)

        preds = np.stack(preds, axis=0)  # shape: [num_clients, num_samples]
        voted = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
        return voted

    def evaluate_global(self, test_batches):
        """
        在每个客户端的测试集上评估投票后的 IDT。
        """
        results = []
        for batch in test_batches:
            pred = self.predict(batch)
            acc = (pred == batch.y.numpy()).mean()
            results.append(acc)

        return {
            'avg_global_idt_acc': np.mean(results),
            'client_wise_acc': results
        }

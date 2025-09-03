import torch
import numpy as np
from typing import List, Dict, Optional, Callable
import pickle
import os
from collections import Counter
import json
from model.idt import IDT, get_activations
from torch_geometric.data import Batch
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter


class SimpleVotingIDTAggregator:
    """简化的基于投票的IDT聚合器"""

    def __init__(self, save_dir: str = "./voting_results"):
        """初始化投票聚合器"""
        self.save_dir = save_dir
        self.client_idts = []
        self.client_weights = []
        self.client_info = []
        self.global_idt = None

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        print(f"[VotingAggregator] Initialized, results will be saved to: {save_dir}")

    def add_client_idt(self, idt, weight: float = 1.0, client_info: Dict = None):
        """添加客户端IDT"""
        self.client_idts.append(idt)
        self.client_weights.append(weight)

        if client_info is None:
            client_info = {'client_id': len(self.client_idts) - 1, 'weight': weight}

        self.client_info.append(client_info)
        print(f"[VotingAggregator] Added client {client_info.get('client_id', len(self.client_idts) - 1)} "
              f"with weight {weight:.3f}")

    def predict(self, batch):
        """
        加权投票预测标签。
        """
        if not self.client_idts:
            raise ValueError("No client IDTs available.")

        preds = []
        for idt in self.client_idts:
            pred = idt.predict(batch)
            preds.append(pred)

        preds = np.stack(preds, axis=0)  # [num_clients, num_samples]
        num_clients, num_samples = preds.shape
        weights = np.array(self.client_weights).reshape(-1, 1)  # [num_clients, 1]

        # 计算加权投票结果
        num_classes = preds.max() + 1  # 假设标签是从0开始的连续整数
        weighted_votes = np.zeros((num_samples, num_classes))

        for c in range(num_clients):
            for i in range(num_samples):
                label = preds[c, i]
                weighted_votes[i, label] += weights[c, 0]

        voted = weighted_votes.argmax(axis=1)
        return voted

    def create_global_idt(self,
                          train_batches: List,
                          values_generator: Callable,
                          idt_params: Dict = None):
        """
        基于投票结果创建全局IDT

        Args:
            train_batches: 训练数据批次列表
            values_generator: 生成IDT训练values的函数
            idt_params: IDT参数字典
        """
        if not self.client_idts:
            raise ValueError("No client IDTs available")

        print(f"[VotingAggregator] Creating global IDT from voting results...")

        # 权重归一化（总和为1）
        total_weight = sum(self.client_weights)
        if total_weight == 0:
            print("[VotingAggregator] Warning: total client weight is 0, fallback to uniform weights")
            num_clients = len(self.client_weights)
            self.client_weights = [1.0 / num_clients] * num_clients
        else:
            self.client_weights = [w / total_weight for w in self.client_weights]
        print(f"[VotingAggregator] Normalized client weights: {self.client_weights}")

        # 默认IDT参数
        default_params = {
            'width': 8,
            'sample_size': 1000,
            'layer_depth': 2,
            'max_depth': None,
            'ccp_alpha': 1e-3
        }

        if idt_params:
            default_params.update(idt_params)

        # 收集所有训练数据和投票标签
        all_batches_data = []
        all_true_labels = []
        all_voted_labels = []

        print(f"[VotingAggregator] Processing {len(train_batches)} training batches...")

        for i, batch in enumerate(train_batches):
            # 获取真实标签
            true_labels = batch.y.numpy()
            all_true_labels.extend(true_labels)
            all_batches_data.append(batch)

            # 获取投票结果
            voted_labels = self.predict(batch)
            all_voted_labels.extend(voted_labels)

            print(f"  Batch {i + 1}: {len(true_labels)} samples processed")

        # 分析投票效果
        self._analyze_voting_effect(all_true_labels, all_voted_labels)

        # 合并所有批次
        print(f"[VotingAggregator] Combining all batches...")
        combined_batch = self._combine_batches(all_batches_data)

        # 生成IDT训练所需的values
        print(f"[VotingAggregator] Generating values for IDT training...")
        values = values_generator(combined_batch)

        # 创建并训练新的IDT
        print(f"[VotingAggregator] Training new IDT with voted labels...")

        self.global_idt = IDT(
            width=default_params['width'],
            sample_size=default_params['sample_size'],
            layer_depth=default_params['layer_depth'],
            max_depth=default_params['max_depth'],
            ccp_alpha=default_params['ccp_alpha']
        )

        # 使用投票结果作为监督信号训练IDT
        self.global_idt.fit(combined_batch, values, np.array(all_voted_labels))

        print(f"[VotingAggregator] Global IDT training completed!")
        print(f"  IDT structure: {len(self.global_idt.layer)} layers + output layer")

        return self.global_idt

    def _analyze_voting_effect(self, true_labels, voted_labels):
        """分析投票效果"""
        true_labels = np.array(true_labels)
        voted_labels = np.array(voted_labels)

        # 计算投票准确率
        voting_accuracy = (true_labels == voted_labels).mean()

        # 分析标签分布
        true_dist = Counter(true_labels)
        voted_dist = Counter(voted_labels)

        print(f"\n[VotingAggregator] Voting Analysis:")
        print(f"  Total samples: {len(true_labels)}")
        print(f"  Voting accuracy vs true labels: {voting_accuracy:.4f}")
        print(f"  True label distribution: {dict(true_dist)}")
        print(f"  Voted label distribution: {dict(voted_dist)}")

        # 保存分析结果
        self.voting_analysis = {
            'voting_accuracy': voting_accuracy,
            'true_distribution': dict(true_dist),
            'voted_distribution': dict(voted_dist),
            'total_samples': len(true_labels)
        }

    def _combine_batches(self, batches):
        """合并多个批次为一个大批次"""
        all_data_list = []
        for batch in batches:
            if hasattr(batch, 'to_data_list'):
                all_data_list.extend(batch.to_data_list())
            else:
                all_data_list.append(batch)
        return Batch.from_data_list(all_data_list)

    def evaluate_global_idt(self, test_batches: List):
        """评估全局IDT性能"""
        if self.global_idt is None:
            raise ValueError("Global IDT not created yet. Call create_global_idt() first")

        all_idt_predictions = []
        all_voting_predictions = []
        all_labels = []

        for batch in test_batches:
            idt_pred = self.global_idt.predict(batch)
            voting_pred = self.predict(batch)
            true_labels = batch.y.detach().cpu().numpy()

            all_idt_predictions.extend(idt_pred)
            all_voting_predictions.extend(voting_pred)
            all_labels.extend(true_labels)

        # Accuracy
        idt_accuracy = sum(p == l for p, l in zip(all_idt_predictions, all_labels)) / len(all_labels)
        voting_accuracy = sum(p == l for p, l in zip(all_voting_predictions, all_labels)) / len(all_labels)

        # Precision, Recall, F1 (macro)
        f1 = f1_score(all_labels, all_idt_predictions, average='macro')
        precision = precision_score(all_labels, all_idt_predictions, average='macro')
        recall = recall_score(all_labels, all_idt_predictions, average='macro')

        # Return metrics (no print)
        return {
            'global_idt_accuracy': idt_accuracy,
            'global_idt_f1': f1,
            'global_idt_precision': precision,
            'global_idt_recall': recall,
            'direct_voting_accuracy': voting_accuracy,
            'improvement': idt_accuracy - voting_accuracy,
            'total_samples': len(all_labels),
            'idt_pred_distribution': dict(Counter(all_idt_predictions)),
            'voting_pred_distribution': dict(Counter(all_voting_predictions)),
            'true_label_distribution': dict(Counter(all_labels))
        }

    def save_results(self, evaluation_results: Dict = None):
        """保存聚合结果"""
        print(f"\n[VotingAggregator] Saving results to {self.save_dir}...")

        try:
            # 保存全局IDT
            if self.global_idt is not None:
                idt_path = os.path.join(self.save_dir, "global_idt.pkl")
                with open(idt_path, 'wb') as f:
                    pickle.dump(self.global_idt, f)
                print(f"  Global IDT saved to {idt_path}")

                # 保存IDT可视化
                try:
                    self.global_idt.prune()
                    viz_path = os.path.join(self.save_dir, "global_idt_structure.png")
                    self.global_idt.save_image(viz_path)
                    print(f"  IDT structure saved to {viz_path}")
                except Exception as e:
                    print(f"  IDT structure save failed: {e}")

                # 保存输出层可视化
                try:
                    output_viz_path = os.path.join(self.save_dir, "global_idt_output_layer.png")
                    self.global_idt.save_output_layer_spacious(output_viz_path)
                    print(f"  Output layer saved to {output_viz_path}")
                except Exception as e:
                    print(f"  Output layer save failed: {e}")

            # 保存聚合信息
            aggregation_info = {
                'num_clients': len(self.client_idts),
                'client_weights': self.client_weights,
                'client_info': self.client_info,
                'voting_analysis': getattr(self, 'voting_analysis', {}),
                'method': 'simple_voting_aggregation'
            }

            info_path = os.path.join(self.save_dir, "aggregation_info.json")
            with open(info_path, 'w') as f:
                json_safe_info = self._convert_to_json_serializable(aggregation_info)
                json.dump(json_safe_info, f, indent=2)
            print(f"  Aggregation info saved to {info_path}")

            # 保存评估结果
            if evaluation_results:
                eval_path = os.path.join(self.save_dir, "evaluation_results.json")
                with open(eval_path, 'w') as f:
                    json_safe_results = self._convert_to_json_serializable(evaluation_results)
                    json.dump(json_safe_results, f, indent=2)
                print(f"  Evaluation results saved to {eval_path}")

            print(f"[VotingAggregator] All results saved successfully!")

        except Exception as e:
            print(f"[VotingAggregator] Error saving results: {e}")

    def _convert_to_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, (np.integer, np.floating)) else k:
                        self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def get_summary(self):
        """获取聚合摘要"""
        summary = {
            'num_clients': len(self.client_idts),
            'client_weights': self.client_weights,
            'has_global_idt': self.global_idt is not None,
            'method': 'simple_voting_aggregation'
        }

        if hasattr(self, 'voting_analysis'):
            summary['voting_analysis'] = self.voting_analysis

        if self.global_idt:
            summary['global_idt_info'] = {
                'num_layers': len(self.global_idt.layer),
                'width': self.global_idt.width,
                'has_output_layer': self.global_idt.out_layer is not None
            }

        return summary
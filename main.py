import torch
import time
from argparse import ArgumentParser
from data.dataset import data_label_skew
from model.client import GNNClient
from model.aggregate.idt_voting import SimpleVotingIDTAggregator
from model.idt import get_activations
from utils.loss_mia import MIAttacker
from utils.sia import SourceInferenceAttackerIDT
from utils.shadow_mia import ShadowMIAttacker, default_shadow_factory
from torch_geometric.data import Batch
import os
import pickle
import numpy as np


def run_federated_voting(args):
    # 设置标签分布
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

    # 初始化客户端
    clients = []
    for cid in range(args.kfold):
        bundle = (num_features, num_classes,
                  train_loaders[cid], val_loaders[cid],
                  train_val_batches[cid], test_batches[cid])
        client = GNNClient(cid, bundle, args, device=cid % args.devices)
        clients.append(client)

    # 本地训练 + 蒸馏为 IDT
    print("\n=== Client Training & IDT Distillation ===")
    for client in clients:
        print(f"[Client {client.client_id}] Training GNN...")
        client.train(conv="GCN")
        print(f"[Client {client.client_id}] Distilling to IDT...")
        client.distill_to_idt(use_pred=True)

        # 检查 IDT 蒸馏结果
        test_batch = client.test_batch
        GCN = client.model
        GCN.eval()
        with torch.no_grad():
            dev = next(GCN.parameters()).device
            print(dev)
            tb = client.test_batch.to(dev)
            logits = GCN(tb)
        acc = (logits.argmax(-1) == tb.y).float().mean().item()

        print(f"[Client {client.client_id}] IDT Distillation Results:")
        print(f"  GCN test accuracy: {acc:.4f}")
        print(f"  IDT test accuracy: {client.idt.accuracy(test_batch):.4f}")
        print(f"  IDT F1 score:      {client.idt.f1_score(test_batch):.4f}")
        print(f"  Fidelity:          {client.idt.fidelity(test_batch, GCN):.4f}")

        # 保存每个客户端的IDT到本地文件
        save_dir = getattr(args, 'save_dir', './saved_client_idts')
        os.makedirs(save_dir, exist_ok=True)

        idt_save_path = os.path.join(save_dir, f'client_{client.client_id}_idt.pkl')
        with open(idt_save_path, 'wb') as f:
            pickle.dump(client.idt, f)
        print(f"[Client {client.client_id}] IDT saved to {idt_save_path}")

    # ================== 投票聚合 ==================
    print("\n=== Starting Voting-based Aggregation ===")

    # 创建投票聚合器
    aggregator = SimpleVotingIDTAggregator(
        save_dir=f"./voting_results/{args.dataset}_{args.weight_method}"
    )

    # 添加客户端IDT到聚合器
    print("[VotingAggregator] Adding client IDTs...")
    for client in clients:
        if hasattr(client, 'idt') and client.idt is not None:
            # 计算客户端权重
            client_weight = calculate_client_weight(client, args.weight_method)

            client_info = {
                'client_id': client.client_id,
                'data_size': len(client.train_loader.dataset) if hasattr(client, 'train_loader') else 0,
                'accuracy': getattr(client, 'best_accuracy', 0.0),
                'test_accuracy': client.idt.accuracy(client.test_batch),
                'weight_method': args.weight_method
            }

            aggregator.add_client_idt(client.idt, client_weight, client_info)
            print(f"  Client {client.client_id}: weight={client_weight:.3f}, "
                  f"test_acc={client_info['test_accuracy']:.3f}")

    # 定义values生成器
    def values_generator(batch):
        """生成IDT训练所需的values"""
        # 使用第一个客户端的模型生成values（可以改进为ensemble）
        reference_model = clients[0].model
        return get_activations(batch, reference_model)

    # 基于投票创建全局IDT
    print("\n[VotingAggregator] Creating global IDT from voting...")
    start_time = time.time()

    global_idt = aggregator.create_global_idt(
        train_val_batches,  # 使用训练+验证数据
        values_generator,
        idt_params={
            'width': args.width,
            'sample_size': args.sample_size,
            'layer_depth': args.layer_depth,
            'max_depth': args.max_depth,
            'ccp_alpha': args.ccp_alpha
        }
    )

    aggregation_time = time.time() - start_time
    print(f"[VotingAggregator] Global IDT creation completed in {aggregation_time:.2f} seconds")

    # ================== 评估 ==================
    print("\n=== Evaluation ===")

    # 评估全局IDT性能
    evaluation_results = aggregator.evaluate_global_idt(test_batches)

    print(f"\n[VotingAggregator] Final Results:")
    print(f"  Global IDT accuracy:   {evaluation_results['global_idt_accuracy']:.4f}")
    print(f"  Global IDT F1 score:   {evaluation_results['global_idt_f1']:.4f}")
    print(f"  Global IDT Precision:  {evaluation_results['global_idt_precision']:.4f}")
    print(f"  Global IDT Recall:     {evaluation_results['global_idt_recall']:.4f}")
    print(f"  Direct voting accuracy:{evaluation_results['direct_voting_accuracy']:.4f}")
    print(f"  Accuracy improvement:  {evaluation_results['improvement']:+.4f}")

    # ================== 可视化 ==================
    print("\n=== Visualization ===")

    # 保存客户端IDT可视化
    for client in clients:
        try:
            # 保存输出层
            output_path = f"./voting_results/{args.dataset}_{args.weight_method}/client_{client.client_id}_output.png"
            client.idt.save_output_layer_spacious(output_path, figsize=(12, 8))
            print(f"[Client {client.client_id}] Output layer saved to {output_path}")

            # 保存完整IDT结构（可选）
            if hasattr(client.idt, 'save_image'):
                full_path = f"./voting_results/{args.dataset}_{args.weight_method}/client_{client.client_id}_full.png"
                client.idt.prune()
                client.idt.save_image(full_path)
                print(f"[Client {client.client_id}] Full IDT structure saved to {full_path}")

        except Exception as e:
            print(f"[Client {client.client_id}] Visualization failed: {e}")

    # 保存全局IDT可视化
    # ================== 分析与保存 ==================
    print("\n=== Saving Results ===")

    # 保存所有结果
    try:
        aggregator.save_results(evaluation_results)
        print("[VotingAggregator] All results saved successfully")
    except Exception as e:
        print(f"[VotingAggregator] Save results failed: {e}")

    # 打印最终摘要
    print("\n=== Final Summary ===")
    try:
        summary = aggregator.get_summary()
        print(f"Aggregation method: Voting-based")
        print(f"Number of clients: {summary['num_clients']}")
        print(f"Client weights: {[f'{w:.3f}' for w in summary['client_weights']]}")
        print(f"Global IDT layers: {summary.get('global_idt_info', {}).get('num_layers', 'Unknown')}")
        print(
            f"Voting accuracy improvement: {summary.get('voting_analysis', {}).get('voted_balance_score', 0) - summary.get('voting_analysis', {}).get('true_balance_score', 0):.3f}")
    except Exception as e:
        print(f"Summary generation failed: {e}")

    print("\n=== Federated Voting Aggregation Completed ===")

    # =================== MIA Attack部分 ===================
    # 收集有效的客户端IDT
    valid_clients = []
    for client in clients:
        if (hasattr(client, 'idt') and client.idt is not None and
                hasattr(client.idt, 'out_layer') and client.idt.out_layer is not None):
            valid_clients.append(client)

    print("\n=== Running Membership Inference Attack ===")
    try:
        attacker = MIAttacker()
        for client in valid_clients:
            print(f"[Client {client.client_id}] loss-based MIA Attack")
            try:
                # 获取训练数据作为成员数据（1/20采样）
                train_data = next(iter(train_loaders[client.client_id]))
                train_data_list = train_data.to_data_list()
                num_members = max(1, len(train_data_list) // 20)
                member_data = train_data_list[:num_members]
                member_batch = Batch.from_data_list(member_data)

                # 获取测试数据作为非成员数据（数量 = num_members）
                test_data_list = test_batches[client.client_id].to_data_list()
                nonmember_data = test_data_list[:num_members]
                nonmember_batch = Batch.from_data_list(nonmember_data)

                attacker.train_attack_model(client.idt, member_batch, nonmember_batch)
                print(f"[Client {client.client_id}] MIA attack completed")
            except Exception as e:
                print(f"[Client {client.client_id}] loss-based MIA attack failed: {str(e)}")
    except Exception as e:
        print(f"MIA setup failed: {str(e)}")

    print("\n=== Running Shadow-Model MIA (per-client) ===")

    shadow_attacker = ShadowMIAttacker(
        shadow_factory=default_shadow_factory,
        include_loss_feature=True,
        random_state=args.seed,
        max_samples_per_shadow=2000,  # 可选：限制每个shadow收集样本数
    )

    for client in valid_clients:
        cid = client.client_id
        print(f"\n[Client {cid}] Shadow-MIA")

        try:
            # member: 来自该client训练集（1/20采样）
            train_data = next(iter(train_loaders[cid]))
            train_list = train_data.to_data_list()
            k = max(1, len(train_list) // 20)
            member_batch = Batch.from_data_list(train_list[:k])

            # non-member: 来自该client测试集（数量=k）
            test_list = test_batches[cid].to_data_list()
            nonmember_batch = Batch.from_data_list(test_list[:k])

            # 训练影子攻击并打印 AUC
            stats = shadow_attacker.train_attack_model(
                target_model=client.idt,  # 如果baseline是GNN则换成 client.model
                member_batch=member_batch,
                nonmember_batch=nonmember_batch,
                num_shadows=5,  # 影子模型数量
                seed=args.seed + cid,  # 每个client不同随机种子
                shadow_epochs=5,  # GNN shadow训练轮数（IDT会忽略）
                shadow_lr=1e-3,
            )

            print(
                f"[Client {cid}] "
                f"AUC_target_transfer={stats['auc_target_transfer']:.4f} | "
                f"AUC_shadow_train={stats['auc_shadow_train']:.4f} | "
                f"AUC_shadow_eval={stats.get('auc_shadow_eval', float('nan')):.4f}"
            )

        except Exception as e:
            print(f"[Client {cid}] Shadow-MIA failed: {e}")

    print("\n=== Running Source Inference Attack (SIA) ===")

    # 1) 所有客户端的被攻击模型（IDT）
    idts_by_cid = {c.client_id: c.idt for c in valid_clients}

    # 2) 每个客户端取一小批 member graphs 作为“已知成员”目标
    targets_by_true_cid = {}
    for client in valid_clients:
        cid = client.client_id
        train_data = next(iter(train_loaders[cid]))
        train_list = train_data.to_data_list()
        k = max(1, len(train_list) // 20)
        targets_by_true_cid[cid] = Batch.from_data_list(train_list[:k]).to("cpu")

    sia = SourceInferenceAttackerIDT(
        loss_mode="cross_entropy",
        temperature=1.0,
        use_probs=True,
        print_ties=True,
    )
    metrics, sia_results = sia.eval_asr(idts_by_cid, targets_by_true_cid)

    print(metrics)

    print("\n=== Federated Learning Completed ===")

    return clients, global_idt


def calculate_client_weight(client, weight_method):
    """计算客户端权重"""
    if weight_method == 'uniform':
        return 1.0

    elif weight_method == 'data_size':
        data_size = len(client.train_loader.dataset) if hasattr(client, 'train_loader') else 1
        return float(data_size)

    elif weight_method == 'idt_quality':
        # 基于IDT结构的权重计算
        if not hasattr(client, 'idt') or client.idt is None:
            return 0.1  # 如果没有IDT，给予最低权重

        if client.idt.out_layer is None or client.idt.out_layer.dt is None:
            return 0.1  # 如果没有输出层决策树，给予最低权重

        # 使用你的结构权重计算方法
        final_weight = calculate_client_weight_by_structure(client)
        return float(max(0.01, final_weight))  # 确保最小权重为0.01

    else:
        return 1.0


def calculate_client_weight_by_structure(client):
    """
    根据客户端IDT输出层的结构性信息分配权重：
    - 根节点 impurity 越高，叶节点 impurity 越低 → 模型更可信。

    返回值范围在 [0, 1]，可作为投票时的权重。
    """
    try:
        dt = client.idt.out_layer.dt
        tree = dt.tree_

        # 根节点 impurity
        root_gini = tree.impurity[0]

        # 找出叶节点 (修正：使用 -1，这是sklearn中TREE_LEAF的值)
        leaf_mask = tree.children_left == -1
        leaf_ginis = tree.impurity[leaf_mask]
        leaf_weights = tree.n_node_samples[leaf_mask]

        # 检查是否有叶节点
        if len(leaf_ginis) == 0:
            print(f"[Client {client.client_id}] No leaf nodes found")
            return 0.0

        # 计算加权平均叶节点 Gini
        avg_leaf_gini = np.average(leaf_ginis, weights=leaf_weights)

        # 纯化得分 = 根 impurity - 平均叶 impurity
        purification_score = root_gini - avg_leaf_gini

        # 归一化到 [0, 1]
        normalized_score = purification_score / (root_gini + 1e-8)
        normalized_score = max(0.0, min(1.0, normalized_score))

        print(f"[Client {client.client_id}] structure weight = {normalized_score:.4f} "
              f"(root_gini={root_gini:.4f}, avg_leaf_gini={avg_leaf_gini:.4f}, "
              f"purification={purification_score:.4f})")

        return float(normalized_score)

    except Exception as e:
        print(f"[Client {client.client_id}] Failed to compute structure weight: {e}")
        return 0.0


if __name__ == '__main__':
    parser = ArgumentParser()

    # 数据和模型
    parser.add_argument('--dataset', type=str, default='EMLC0')
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--layer_depth', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--ccp_alpha', type=float, default=1e-3)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--samples_per_label', type=int, default=1000)

    # 聚合参数 - 修改为适用于投票方法
    parser.add_argument('--weight_method', type=str, default='idt_quality',
                        choices=['uniform', 'data_size', 'idt_quality'],
                        help='Method for calculating client weights in voting')

    parser.add_argument('--aggregation_method', type=str, default='voting',
                        help='Now using voting-based aggregation')
    parser.add_argument('--similarity_threshold', type=float, default=0.8,
                        help='Not used in voting method but kept for compatibility')

    args = parser.parse_args()

    print("=== Federated IDT with Voting Aggregation ===")
    print(f"Dataset: {args.dataset} | Weight Method: {args.weight_method}")
    print(f"IDT Params: width={args.width}, depth={args.layer_depth}, max_depth={args.max_depth}")
    print("=" * 60)

    run_federated_voting(args)

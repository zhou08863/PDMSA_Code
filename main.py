import os
import warnings
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from transformers import logging as hf_logging
from Config import config
from utils.data_pipeline import data_format, read_from_file, train_val_split, Processor
from Trainer import Trainer
from Models import get_model_class, model_requires_img_features
from collections import Counter
from utils.advanced_visualization import generate_large_scale_visualizations
from utils.visualization_utils import (ensure_dir, save_metrics_to_csv,
                                 create_advanced_training_visualization,
                                 visualize_channel_attention, visualize_spatial_attention,
                                 create_confusion_matrix_visualization)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

if os.name == 'nt':
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
else:
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
matplotlib.rcParams['axes.unicode_minus'] = False

train_loader = None
val_loader = None
test_loader = None
model = None

parser = argparse.ArgumentParser()
parser.add_argument('--text_pretrained_model', default=config.bert_name,
                    help='文本分析模型', type=str)
parser.add_argument('--fuse_model_type', default=config.fuse_model_type, help='融合模型类别', type=str)
parser.add_argument('--lr', default=config.learning_rate, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=config.weight_decay, help='设置权重衰减', type=float)
parser.add_argument('--epoch', default=config.epoch, help='设置训练轮数', type=int)
parser.add_argument('--accumulation_steps', default=config.accumulation_steps, help='梯度累积步数', type=int)
parser.add_argument('--modality', default=config.modality, choices=['both', 'text', 'image'],
                    help='选择训练与测试使用的模态，可选both、text、image')
parser.add_argument('--patience', default=config.early_stopping_patience, help='早停耐心值', type=int)
parser.add_argument('--min_delta', default=config.early_stopping_min_delta, help='最小改善阈值', type=float)
parser.add_argument('--early_stopping_metric', default=config.early_stopping_metric, choices=['acc', 'loss', 'f1', 'val_macro_f1'], help='早停监控指标',
                    type=str)
args = parser.parse_args()


def apply_full_un_runtime_config(cfg):
    if cfg.fuse_model_type != 'UN':
        return cfg
    cfg.use_cbam = True
    cfg.use_bi_attention = True
    cfg.credibility_eval = True
    cfg.use_credibility = True
    cfg.weighting_strategy = 'text_confidence_softmax'
    return cfg


def infer_runtime_class_weights(train_data, cfg):
    label_order = ['negative', 'positive']
    label_counts = Counter(item[3] for item in train_data)
    total_samples = sum(label_counts.get(label, 0) for label in label_order)
    if total_samples <= 0:
        return list(cfg.loss_weight)

    raw_weights = []
    for label in label_order:
        count = max(label_counts.get(label, 0), 1)
        raw_weights.append(total_samples / (count * len(label_order)))

    raw_sum = sum(raw_weights)
    if raw_sum <= 0:
        return list(cfg.loss_weight)
    return [float(weight / raw_sum) for weight in raw_weights]


config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.epoch = args.epoch
config.bert_name = args.text_pretrained_model
config.fuse_model_type = args.fuse_model_type
config.accumulation_steps = args.accumulation_steps
config.modality = args.modality
config.early_stopping_metric = args.early_stopping_metric
config.early_stopping_patience = args.patience
config.early_stopping_min_delta = args.min_delta
config = apply_full_un_runtime_config(config)
print(f'训练模态: {config.modality}')

print('TextModel: {}, ImageModel: {}, FuseModel: {}'.format(config.bert_name, 'ResNet50', config.fuse_model_type))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config.device = device
processor = Processor(config)
model = get_model_class(config.fuse_model_type)(config)
trainer = Trainer(config, processor, model, device)

train_loader = None
val_loader = None
test_loader = None

visualizations_dir = os.path.join(config.output_path, f'visualizations_{config.modality}')
ensure_dir(visualizations_dir)

metrics_dir = os.path.join(visualizations_dir, 'metrics')
attention_maps_dir = os.path.join(visualizations_dir, 'attention_maps')
confusion_matrices_dir = os.path.join(visualizations_dir, 'confusion_matrices')
csv_dir = os.path.join(visualizations_dir, 'csv_data')

for directory in [metrics_dir, attention_maps_dir, confusion_matrices_dir, csv_dir]:
    ensure_dir(directory)

train_losses, val_losses, test_losses = [], [], []
train_accuracies, val_accuracies, test_accuracies = [], [], []
train_f1s, val_f1s, test_f1s = [], [], []
learning_rates = []


def train():
    global train_loader, val_loader, test_loader, model
    needs_img_features = model_requires_img_features(config.fuse_model_type)
    
    data_format(os.path.join(config.root_path, 'data', 'train.txt'),
                config.data_dir, config.train_data_path)
    data = read_from_file(config.train_data_path, config.data_dir, config.only)

    guids = [item[0] for item in data]
    if len(set(guids)) != len(guids):
        print("Duplicates found in data. Removing duplicates.")
        unique_data = {}
        for item in data:
            guid = item[0]
            if guid not in unique_data:
                unique_data[guid] = item
        data = list(unique_data.values())
    else:
        print("No duplicates found in data.")

    labels = [item[3] for item in data]
    print("总数据集类别分布:", Counter(labels))

    train_data, val_data, test_data = train_val_split(data)
    train_guids = set([item[0] for item in train_data])
    val_guids = set([item[0] for item in val_data])
    overlap = train_guids.intersection(val_guids)
    if overlap:
        print(f"Error: There are {len(overlap)} overlapping GUIDs between train and validation sets.")
    else:
        print("No overlapping GUIDs between train and validation sets.")

    test_guids = set([item[0] for item in test_data])
    overlap_train_test = train_guids.intersection(test_guids)
    overlap_val_test = val_guids.intersection(test_guids)
    if overlap_train_test or overlap_val_test:
        print(f"Error: There are overlapping GUIDs between train/val and test sets.")
    else:
        print("No overlapping GUIDs between train/val and test sets.")

    train_labels = [item[3] for item in train_data]
    val_labels = [item[3] for item in val_data]
    test_labels = [item[3] for item in test_data]
    print("训练集类别分布:", Counter(train_labels))
    print("验证集类别分布:", Counter(val_labels))
    print("测试集类别分布:", Counter(test_labels))
    if getattr(config, 'auto_loss_weight', True):
        config.loss_weight = infer_runtime_class_weights(train_data, config)
        print("根据训练集自动计算 loss_weight:", config.loss_weight)

    train_loader = processor(train_data, config.train_params, is_training=True)
    val_loader = processor(val_data, config.val_params, is_training=False)
    test_loader = processor(test_data, config.test_params, is_training=False)

    best_metric = float('-inf')
    if args.early_stopping_metric == 'loss':
        best_metric = float('inf')
    best_epoch = 0
    no_improvement_count = 0

    def is_better(current, best, metric_type):
        if metric_type == 'loss':
            return current < best - args.min_delta
        else:
            return current > best + args.min_delta

    for e in range(config.epoch):
        print('-' * 20 + ' ' + 'Epoch ' + str(e + 1) + ' ' + '-' * 20)
        train_loss, train_acc, train_f1, _ = trainer.train(train_loader, needs_img_features)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1s.append(train_f1)
        learning_rates.append(trainer.optimizer.param_groups[0]['lr'])

        print('Train Loss: {}'.format(train_loss))
        print('Train Acc: {}'.format(train_acc))
        print('Train Weighted F1: {}'.format(train_f1))

        vloss, vacc, vf1, vmacro_f1, _ = trainer.valid(val_loader, needs_img_features)
        print('Valid Loss: {}'.format(vloss))
        print('Valid Acc: {}'.format(vacc))
        print('Valid Weighted F1: {}'.format(vf1))
        print('Valid Macro F1: {}'.format(vmacro_f1))

        val_losses.append(vloss)
        val_accuracies.append(vacc)
        val_f1s.append(vf1)

        tloss_test, tacc_test, tf1_test, tmacro_f1_test, _ = trainer.valid(test_loader, needs_img_features)
        print('Test Loss: {}'.format(tloss_test))
        print('Test Acc: {}'.format(tacc_test))
        print('Test Weighted F1: {}'.format(tf1_test))
        print('Test Macro F1: {}'.format(tmacro_f1_test))

        test_losses.append(tloss_test)
        test_accuracies.append(tacc_test)
        test_f1s.append(tf1_test)

        current_metric = {
            'acc': vacc,
            'loss': vloss,
            'f1': vf1,
            'val_macro_f1': vmacro_f1
        }[args.early_stopping_metric]

        trainer.scheduler.step(current_metric)

        if is_better(current_metric, best_metric, args.early_stopping_metric):
            best_metric = current_metric
            best_epoch = e
            no_improvement_count = 0

            best_model_save_path = os.path.join(config.output_path, f'best_model_{config.modality}.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print(f'Update best model! Best {args.early_stopping_metric}: {best_metric:.4f}')
            print(f'Model saved to {best_model_save_path}')

            y_true_val, y_pred_val = trainer.get_predictions(val_loader, needs_img_features)
            if y_true_val and y_pred_val:
                cm_val_path = os.path.join(confusion_matrices_dir, f'{config.modality}_confusion_matrix_val_best.png')
                create_confusion_matrix_visualization(
                    y_true_val, y_pred_val, cm_val_path,
                    title=f'{config.modality.capitalize()} Validation Confusion Matrix'
                )
                print(f'Validation confusion matrix saved to {cm_val_path}')

            y_true_test, y_pred_test = trainer.get_predictions(test_loader, needs_img_features)
            if y_true_test and y_pred_test:
                cm_test_path = os.path.join(confusion_matrices_dir, f'{config.modality}_confusion_matrix_test_best.png')
                create_confusion_matrix_visualization(
                    y_true_test, y_pred_test, cm_test_path,
                    title=f'{config.modality.capitalize()} Test Confusion Matrix'
                )
                print(f'Test confusion matrix saved to {cm_test_path}')

        else:
            no_improvement_count += 1
            print(f'No improvement in {args.early_stopping_metric}. '
                  f'Epochs without improvement: {no_improvement_count}/{args.patience}')

        if no_improvement_count >= args.patience:
            print(f'Early stopping triggered. Best {args.early_stopping_metric}: '
                  f'{best_metric:.4f} at epoch {best_epoch + 1}')
            break

    metrics_data = {
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'test_loss': test_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies,
        'test_acc': test_accuracies,
        'train_f1': train_f1s,
        'val_f1': val_f1s,
        'test_f1': test_f1s,
        'learning_rate': learning_rates
    }

    csv_path = os.path.join(csv_dir, f'{config.modality}_training_metrics.csv')
    save_metrics_to_csv(metrics_data, csv_path)

    plot_metrics(train_losses, val_losses, test_losses,
                 train_accuracies, val_accuracies, test_accuracies,
                 train_f1s, val_f1s, test_f1s)

    if config.modality in ['image', 'both']:
        create_model_visualizations(model, val_loader)


def plot_metrics(train_losses, val_losses, test_losses,
                 train_accuracies, val_accuracies, test_accuracies,
                 train_f1s, val_f1s, test_f1s):
    epochs = list(range(1, len(train_losses) + 1))

    loss_plot_path = os.path.join(metrics_dir, f'{config.modality}_loss_curve.png')
    create_advanced_training_visualization(
        epochs, train_losses, val_losses, test_losses,
        '损失值', '损失值', loss_plot_path
    )

    acc_plot_path = os.path.join(metrics_dir, f'{config.modality}_accuracy_curve.png')
    create_advanced_training_visualization(
        epochs, train_accuracies, val_accuracies, test_accuracies,
        '准确率', '准确率', acc_plot_path
    )

    f1_plot_path = os.path.join(metrics_dir, f'{config.modality}_f1_score_curve.png')
    create_advanced_training_visualization(
        epochs, train_f1s, val_f1s, test_f1s,
        'F1分数', 'F1分数', f1_plot_path
    )

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, 'purple', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title(f'{config.modality.capitalize()} Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.7)
    lr_plot_path = os.path.join(metrics_dir, f'{config.modality}_learning_rate.png')
    plt.savefig(lr_plot_path, dpi=300)
    plt.close()


def create_model_visualizations(model, val_loader):
    for batch in val_loader:
        guids, texts, texts_mask, imgs, labels, img_features = batch[:6]
        for i in range(min(5, len(imgs))):
            img_tensor = imgs[i]
            img_feature = img_features[i]

            img_np = img_tensor.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            from PIL import Image
            original_img = Image.fromarray((img_np * 255).astype(np.uint8))

            channel_attn_path = os.path.join(attention_maps_dir, f'{config.modality}_channel_attention_sample_{i}.png')
            visualize_channel_attention(model, img_tensor.to(device), img_feature.to(device), channel_attn_path)

            spatial_attn_path = os.path.join(attention_maps_dir, f'{config.modality}_spatial_attention_sample_{i}.png')
            visualize_spatial_attention(model, img_tensor.to(device), img_feature.to(device), original_img,
                                        spatial_attn_path)

        break


if __name__ == '__main__':
    train()
    print("开始生成高级可视化分析...")
    visualization_dir = os.path.join(config.output_path, f'visualizations_{config.modality}_enhanced')
    ensure_dir(visualization_dir)

    generate_large_scale_visualizations(
        model=model,
        data_loader=val_loader,
        output_dir=visualization_dir,
        sample_count=1
    )

import os
import torch
import argparse
from Config import config
from utils.data_pipeline import data_format, read_from_file, Processor
from Models import get_model_class, model_requires_img_features
from Trainer import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--text_pretrained_model', default=config.bert_name,
                    help='文本分析模型', type=str)
parser.add_argument('--fuse_model_type', default='UN', help='融合模型类别', type=str)
parser.add_argument('--load_model_path', default=os.path.join(config.output_path, 'best_model.pth'),
                    help='已经训练好的模型路径', type=str)
parser.add_argument('--text_only', action='store_true', help='仅用文本预测')
parser.add_argument('--img_only', action='store_true', help='仅用图像预测')
parser.add_argument('--lr', default=3e-5, help='设置学习率', type=float)
parser.add_argument('--weight_decay', default=1e-4, help='设置权重衰减', type=float)
parser.add_argument('--accumulation_steps', default=8, help='梯度累积步数', type=int)
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

config.bert_name = args.text_pretrained_model
config.fuse_model_type = args.fuse_model_type
config.load_model_path = args.load_model_path
config.learning_rate = args.lr
config.weight_decay = args.weight_decay
config.accumulation_steps = args.accumulation_steps

if args.text_only:
    config.modality = 'text'
elif args.img_only:
    config.modality = 'image'
else:
    config.modality = 'both'

config = apply_full_un_runtime_config(config)

print('TextModel: {}, ImageModel: {}, FuseModel: {}'.format(config.bert_name, 'ResNet50', config.fuse_model_type))
print(f'当前测试模态: {config.modality}')

processor = Processor(config)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
config.device = device
needs_img_features = model_requires_img_features(config.fuse_model_type)

model = get_model_class(config.fuse_model_type)(config).to(device)
if config.load_model_path is not None:
    model.load_state_dict(torch.load(config.load_model_path, map_location=device), strict=False)

def test():
    
    try:
        data_format(os.path.join(config.root_path, 'data', 'test.txt'),
                    config.data_dir, config.test_data_path)
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        return

    try:
        test_data = read_from_file(config.test_data_path, config.data_dir, config.only)
    except UnicodeDecodeError as e:
        print(f"Error reading file {config.test_data_path}: {e}")
        return

    test_loader = processor(test_data, config.test_params)
    trainer = Trainer(config, processor, model, device)

    print("\n开始测试集推理...")
    true_labels, pred_labels = trainer.get_predictions(test_loader, needs_img_features)

    print("\n计算测试集详细评估指标...")
    tloss, tacc, tf1, tmacro_f1, _ = trainer.valid(test_loader, needs_img_features)

    print(f"\n[Test 最终结果] Loss: {tloss:.4f}, Acc: {tacc:.4f}, Weighted F1: {tf1:.4f}, Macro F1: {tmacro_f1:.4f}")

if __name__ == "__main__":
    if args.load_model_path is None:
        print('Please provide the path to the trained model using --load_model_path')
    else:
        test()

import os
import json
import random

import chardet
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torchvision import transforms
import jieba
import jieba.posseg as pseg
import cv2


def _log_numeric_progress(stage_name, current_step, total_steps):
    if total_steps <= 0:
        return
    half_step = (total_steps + 1) // 2

    if current_step == total_steps:
        print(f"\r{stage_name}进度: 100%/100%", end="\n", flush=True)
    elif current_step == half_step and half_step != total_steps:
        print(f"\r{stage_name}进度: 50%/100%", end="", flush=True)


# 将文本和标签格式化成一个json
def data_format(input_path, data_dir, output_path):
    data = []
    with open(input_path, encoding='utf-8') as f:
        lines = f.readlines()
        total_lines = len(lines)
        for idx, line in enumerate(lines, start=1):
            try:
                # 尝试分割数据，如果分割失败将抛出 ValueError
                guid, label = line.replace('\n', '').split(',')[:2]  # 确保分割为两部分
            except ValueError as e:
                print(f"处理行时出错: {line.strip()}")
                print(f"错误: {e}")
                continue  # 跳过有问题的行

            text_path = os.path.join(data_dir, (guid + '.txt'))
            if guid == 'guid':  # 如果是标题行，则跳过
                continue

            # 尝试读取文本文件并解码
            try:
                with open(text_path, 'rb') as textf:
                    text_byte = textf.read()
                    encode = chardet.detect(text_byte)
                    # 如果 chardet 没有检测到编码或者编码不是字符串，使用 'utf-8' 作为默认编码
                    encoding = encode.get('encoding') or 'utf-8'
                    text = text_byte.decode(encoding)
            except UnicodeDecodeError as e:
                print(f"解码文本文件时出错，GUID {guid}: {e}")
                # 可以选择跳过该文件，或者使用其他方式尝试解码
                continue
            except Exception as e:
                print(f"读取文本文件时出错，GUID {guid}: {e}")
                continue

            # 清理文本数据
            text = text.strip()
            data.append({
                'guid': guid,
                'label': label,
                'text': text
            })
            _log_numeric_progress('Formating', idx, total_lines)

        # 写入格式化后的数据到输出文件
        with open(output_path, 'w', encoding='utf-8') as wf:  # 确保以 utf-8 编码写入
            json.dump(data, wf, indent=4)


# 读取数据，返回[(guid, text, img, label)]元组列表
def read_from_file(path, data_dir, only=None):
    data = []
    guids_set = set()
    with open(path, 'r', encoding='utf-8') as f:  # 明确指定 UTF-8 编码
        json_file = json.load(f)
        total_items = len(json_file)
        for idx, d in enumerate(json_file, start=1):
            guid, label, text = d['guid'], d['label'], d['text']
            if guid == 'guid':
                continue
            if guid in guids_set:
                print(f"重复的GUID: {guid}，跳过。")
                continue
            guids_set.add(guid)

            if only == 'text':
                img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                # 尝试直接从data_dir目录加载图像
                img_path = os.path.join(data_dir, f"{guid}.jpg")

                # 检查文件是否存在
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img_obj:
                            img_obj.load()
                            img_copy = img_obj.copy()
                        img = img_copy
                    except Exception as e:
                        print(f"加载图像 {img_path} 时出错: {e}")
                        img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
                else:
                    # 尝试查找PNG格式
                    img_path_png = os.path.join(data_dir, f"{guid}.png")
                    if os.path.exists(img_path_png):
                        try:
                            with Image.open(img_path_png) as img_obj:
                                img_obj.load()
                                img_copy = img_obj.copy()
                            img = img_copy
                        except Exception as e:
                            print(f"加载图像 {img_path_png} 时出错: {e}")
                            img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
                    else:
                        print(f"找不到图像文件: {guid}，使用黑色图像替代。")
                        img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))

            if only == 'img':
                text = ''

            data.append((guid, text, img, label))
            _log_numeric_progress('Loading', idx, total_items)

    return data


# 分离训练集、验证集和测试集
def train_val_split(data, val_size=0.2, test_size=0.2, random_state=42):
    labels = [item[3] for item in data]  # 获取标签列表
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size / (1 - test_size),
        stratify=[item[3] for item in train_val_data],
        random_state=random_state,
    )
    return train_data, val_data, test_data


# 写入数据
def write_to_file(path, outputs):
    outputs = list(outputs)
    total_items = len(outputs)
    with open(path, 'w', encoding='utf-8') as f:
        for idx, line in enumerate(outputs, start=1):
            f.write(line)
            f.write('\n')
            _log_numeric_progress('Writing', idx, total_items)


# 保存模型
def save_model(output_path, model_type, model):
    output_model_dir = os.path.join(output_path, model_type)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)  # 没有文件夹则创建
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


class APIDataset(Dataset):
    def __init__(self, guids, texts, imgs, labels, img_features, raw_texts, raw_imgs) -> None:
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels
        self.img_features = img_features  # 新增：图像的额外特征
        self.raw_texts = raw_texts
        self.raw_imgs = raw_imgs

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, index):
        return self.guids[index], self.texts[index], \
            self.imgs[index], self.labels[index], self.img_features[index], \
            self.raw_texts[index], self.raw_imgs[index]

    def collate_fn(self, batch):
        guids = [b[0] for b in batch]
        texts = [torch.LongTensor(b[1]) for b in batch]
        imgs = torch.stack([b[2] for b in batch])
        labels = torch.LongTensor([b[3] for b in batch])
        img_features = torch.stack([b[4] for b in batch])  # 新增：图像的额外特征
        raw_texts = [b[5] for b in batch]
        raw_imgs = [b[6] for b in batch]

        # 处理文本 统一长度 增加mask tensor
        texts_mask = [torch.ones_like(text) for text in texts]

        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)

        return guids, paded_texts, paded_texts_mask, imgs, labels, img_features, raw_texts, raw_imgs


def api_encode(data, labelvocab, config, is_training=False):
    # labelvocab.add_label('neutral')屏蔽掉中性标签，用于二分类
    labelvocab.add_label('negative', id=0)
    labelvocab.add_label('positive', id=1)
    labelvocab.add_label('null')  # 空标签

    # 文本处理 BERT的tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)
    bert_model = AutoModelForMaskedLM.from_pretrained(config.bert_name)
    bert_model.eval()

    # 文本增强函数
    def augment_text(text, max_replacements=2):
        # 限制文本长度，避免超过BERT最大限制
        max_text_length = 450  # 留出一些空间给特殊标记和额外字符
        if len(text) > max_text_length:
            text = text[:max_text_length]

        # 使用jieba进行分词并标注词性
        words = pseg.cut(text)
        word_list = [(word, flag) for word, flag in words]

        # 选择可以替换的词（名词、动词、副词）
        replaceable_pos = {'n', 'v', 'd'}  # 名词、动词、副词
        candidate_indices = [i for i, (word, flag) in enumerate(word_list) if flag in replaceable_pos]

        if not candidate_indices:
            return text  # 如果没有可替换的词，直接返回原文

        # 确定替换的词数，最多为max_replacements，且不超过候选词的数量
        num_replacements = min(max_replacements, max(1, int(len(word_list) * 0.1)), len(candidate_indices))

        # 随机选择要替换的词的位置
        replace_indices = random.sample(candidate_indices, num_replacements)

        # 构建掩盖后的文本
        augmented_tokens = []
        for i, (word, flag) in enumerate(word_list):
            if i in replace_indices:
                augmented_tokens.append('[MASK]')
            else:
                augmented_tokens.append(word)

        masked_text = ''.join(augmented_tokens)

        # 使用BERT tokenizer对掩盖后的文本进行编码，并获取[MASK]的位置
        encoding = tokenizer(masked_text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encoding['input_ids']

        # 找到[MASK]的索引位置
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            return text  # 如果没有掩码位置，返回原文

        with torch.no_grad():
            outputs = bert_model(input_ids)
            predictions = outputs.logits

        # 替换被掩盖的词
        for pos in mask_positions:
            predicted_logits = predictions[0, pos]
            predicted_ids = torch.topk(predicted_logits, k=10).indices.tolist()
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

            # 过滤特殊符号和子词标记
            predicted_tokens = [token for token in predicted_tokens if token not in tokenizer.all_special_tokens]
            predicted_tokens = [token.replace('##', '') for token in predicted_tokens]

            if not predicted_tokens:
                continue  # 如果没有合适的替换词，跳过

            # 选择一个替换词
            new_token = random.choice(predicted_tokens)

            # 更新 input_ids 中的对应位置
            input_ids[0, pos] = tokenizer.convert_tokens_to_ids(new_token)

        # 解码得到增强后的文本
        augmented_text = tokenizer.decode(input_ids[0], skip_special_tokens=True).replace(' ', '')

        return augmented_text

    # 图像处理 torchvision的transforms
    def get_resize(image_size):
        for i in range(20):
            if 2 ** i >= image_size:
                return 2 ** i
        return image_size

    # 根据是否为训练数据，选择不同的图像增强策略
    if is_training:
        img_transform = transforms.Compose([
            transforms.Resize(get_resize(config.image_size)),
            transforms.CenterCrop(config.image_size),
            transforms.RandomHorizontalFlip(0.7),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize(get_resize(config.image_size)),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # 额外的图像特征处理
    def compute_image_features(img, config):
        """简化后的图像特征提取函数 - 不再计算手工特征"""
        # 返回空特征向量
        return torch.zeros(config.img_feature_dim, dtype=torch.float32)

    # 根据模态优化数据处理
    guids, encoded_texts, encoded_imgs, encoded_labels, img_features = [], [], [], [], []
    raw_texts, raw_imgs = [], []
    total_items = len(data)
    for idx, line in enumerate(data, start=1):
        guid, text, img, label = line
        guids.append(guid)
        raw_texts.append(text if text else '无内容')
        raw_imgs.append(img.copy() if hasattr(img, 'copy') else img)

        # 根据模态选择性处理
        if config.modality in ['text', 'both']:
            # 文本处理
            text = text.replace('#', '').strip()
            if not text:
                text = '无内容'
            if is_training:
                text = augment_text(text)
            tokens = tokenizer.tokenize('[CLS] ' + text + ' [SEP]')
            # 确保token长度不超过BERT模型的最大限制
            if len(tokens) > 512:
                tokens = tokens[:512]
            encoded_ids = tokenizer.convert_tokens_to_ids(tokens)
            if not encoded_ids:
                encoded_ids = tokenizer.convert_tokens_to_ids(['[CLS]', '无内容', '[SEP]'])
            encoded_texts.append(encoded_ids)
        else:
            # 如果是仅图像模态，使用空文本
            encoded_texts.append(tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]']))

        if config.modality in ['image', 'both']:
            # 图像处理
            encoded_imgs.append(img_transform(img))
            img_feature = compute_image_features(img, config)
            img_features.append(img_feature)
        else:
            # 如果是仅文本模态，使用空图像
            encoded_imgs.append(torch.zeros((3, config.image_size, config.image_size)))
            img_features.append(torch.zeros(config.img_feature_dim))

        # 标签处理
        label_id = labelvocab.label_to_id(label)
        if label_id == -1:
            label_id = labelvocab.label_to_id('null')
        encoded_labels.append(label_id)
        _log_numeric_progress('Encoding', idx, total_items)

    return guids, encoded_texts, encoded_imgs, encoded_labels, img_features, raw_texts, raw_imgs


def api_decode(outputs, labelvocab):
    formated_outputs = ['guid,tag']
    outputs = list(outputs)
    total_items = len(outputs)
    for idx, (guid, label) in enumerate(outputs, start=1):
        # 只处理积极和消极标签，忽略或映射 'null' 标签
        if labelvocab.id_to_label(label) in ['positive', 'negative']:
            formated_outputs.append((str(guid) + ',' + labelvocab.id_to_label(label)))
        else:
            formated_outputs.append((str(guid) + ',null'))  # 或者根据需求进行其他处理
        _log_numeric_progress('Decoding', idx, total_items)
    return formated_outputs


def api_metric(true_labels, pred_labels):
    # 打印详细的分类报告
    print(classification_report(true_labels, pred_labels))

    # 计算各项指标
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return accuracy


class LabelVocab:
    UNK = 'UNK'

    def __init__(self) -> None:
        self.label2id = {}
        self.id2label = {}

    def __len__(self):
        return len(self.label2id)

    def add_label(self, label, id=None):
        if label not in self.label2id:
            if id is None:
                self.label2id.update({label: len(self.label2id)})
                self.id2label.update({len(self.id2label): label})
            else:
                self.label2id.update({label: id})
                self.id2label.update({id: label})

    def label_to_id(self, label):
        return self.label2id.get(label, self.label2id.get(self.UNK, -1))  # 返回 -1 如果没有找到

    def id_to_label(self, id):
        return self.id2label.get(id, self.UNK)


class Processor:
    def __init__(self, config) -> None:
        self.config = config
        self.labelvocab = LabelVocab()

    def __call__(self, data, params, is_training=False):
        return self.to_loader(data, params, is_training)

    def encode(self, data, is_training=False):
        return api_encode(data, self.labelvocab, self.config, is_training=is_training)

    def decode(self, outputs):
        return api_decode(outputs, self.labelvocab)

    def metric(self, inputs, outputs):
        return api_metric(inputs, outputs)

    def to_dataset(self, data, is_training=False):
        dataset_inputs = self.encode(data, is_training=is_training)
        return APIDataset(*dataset_inputs)

    def to_loader(self, data, params, is_training=False):
        dataset = self.to_dataset(data, is_training=is_training)
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn)

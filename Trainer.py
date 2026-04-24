import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score


def initialize_weights(model):
    excluded_prefixes = (
        'text_model.bert',
        'img_model.full_resnet',
        'img_model.resnet_h',
        'img_model.resnet_p',
    )
    for name, module in model.named_modules():
        if not name or name.startswith(excluded_prefixes):
            continue
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class Trainer():
    def __init__(self, config, processor, model,
                 device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                 initialize_model_weights=True):
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device

        if initialize_model_weights:
            initialize_weights(self.model)

        no_decay = ('bias', 'LayerNorm.weight')
        bert_prefixes = ('text_model.bert',)
        resnet_prefixes = ('img_model.full_resnet', 'img_model.resnet_h', 'img_model.resnet_p')
        named_parameters = list(self.model.named_parameters())

        optimizer_grouped_parameters = []
        group_settings = [
            ('bert', bert_prefixes, self.config.bert_learning_rate),
            ('resnet', resnet_prefixes, self.config.resnet_learning_rate),
            ('default', (), self.config.learning_rate),
        ]

        for group_name, prefixes, learning_rate in group_settings:
            for use_no_decay in (False, True):
                group_params = []
                for param_name, param in named_parameters:
                    if not param.requires_grad:
                        continue

                    is_bert_param = param_name.startswith(bert_prefixes)
                    is_resnet_param = param_name.startswith(resnet_prefixes)
                    if group_name == 'bert' and not is_bert_param:
                        continue
                    if group_name == 'resnet' and not is_resnet_param:
                        continue
                    if group_name == 'default' and (is_bert_param or is_resnet_param):
                        continue

                    has_no_decay = any(nd in param_name for nd in no_decay)
                    if has_no_decay != use_no_decay:
                        continue
                    group_params.append(param)

                if group_params:
                    optimizer_grouped_parameters.append({
                        'params': group_params,
                        'lr': learning_rate,
                        'weight_decay': 0.0 if use_no_decay else self.config.weight_decay
                    })

        self.optimizer = Adam(optimizer_grouped_parameters, betas=(0.9, 0.999))

        scheduler_mode = 'min' if getattr(config, 'early_stopping_metric', 'acc') == 'loss' else 'max'

        if config.modality == 'text':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_mode,
                factor=config.scheduler_factor_text,
                patience=config.scheduler_patience_text,
                verbose=True,
                min_lr=config.scheduler_min_lr_text
            )
        elif config.modality == 'image':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_mode,
                factor=config.scheduler_factor_image,
                patience=config.scheduler_patience_image,
                verbose=True,
                min_lr=config.scheduler_min_lr_image
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_mode,
                factor=config.scheduler_factor_multimodal,
                patience=config.scheduler_patience_multimodal,
                verbose=True,
                min_lr=config.scheduler_min_lr_multimodal
            )
        self.class_loss_tracker = defaultdict(list)
        self.loss_func = nn.CrossEntropyLoss(
            weight=torch.tensor(self.config.loss_weight, dtype=torch.float32).to(self.device))
        if hasattr(self.model, 'loss_func'):
            self.model.loss_func = self.loss_func
        self.current_epoch = 0
        self.last_train_loss_components = {'main_loss': 0.0, 'aux_loss': 0.0, 'total_loss': 0.0}
        self.last_valid_loss_components = {'main_loss': 0.0, 'aux_loss': 0.0, 'total_loss': 0.0}

    @staticmethod
    def _unpack_batch(batch):
        guids, texts, texts_mask, imgs, labels, img_features = batch[:6]
        raw_texts = batch[6] if len(batch) > 6 else None
        raw_imgs = batch[7] if len(batch) > 7 else None
        return guids, texts, texts_mask, imgs, labels, img_features, raw_texts, raw_imgs

    @staticmethod
    def _log_numeric_progress(stage_name, current_step, total_steps, interval=100):
        if total_steps <= 0:
            return
        half_step = (total_steps + 1) // 2

        if current_step == total_steps:
            print(f"\r{stage_name}进度: 100%/100%", end="\n", flush=True)
        elif current_step == half_step and half_step != total_steps:
            print(f"\r{stage_name}进度: 50%/100%", end="", flush=True)

    def _read_model_loss_breakdown(self, fallback_total_loss):
        default_breakdown = {
            'main_loss': float(fallback_total_loss),
            'aux_loss': 0.0,
            'total_loss': float(fallback_total_loss),
        }
        model_breakdown = getattr(self.model, 'last_loss_breakdown', None)
        if not isinstance(model_breakdown, dict):
            return default_breakdown

        parsed = {}
        for key, default_value in default_breakdown.items():
            raw_value = model_breakdown.get(key, default_value)
            try:
                parsed[key] = float(raw_value)
            except (TypeError, ValueError):
                parsed[key] = default_value
        return parsed

    def _call_model(self, texts, texts_mask, imgs, labels, img_features, needs_img_features, include_labels,
                    raw_texts=None, raw_imgs=None):
        model_kwargs = {}

        if self.config.modality == 'text':
            model_kwargs.update({
                'texts': texts,
                'texts_mask': texts_mask,
                'imgs': None,
            })
            if needs_img_features:
                model_kwargs['img_features'] = None
        elif self.config.modality == 'image':
            model_kwargs.update({
                'texts': None,
                'texts_mask': None,
                'imgs': imgs,
            })
            if needs_img_features:
                model_kwargs['img_features'] = img_features
        else:
            model_kwargs.update({
                'texts': texts,
                'texts_mask': texts_mask,
                'imgs': imgs,
            })
            if needs_img_features:
                model_kwargs['img_features'] = img_features

        if include_labels:
            model_kwargs['labels'] = labels

        if getattr(self.model, 'supports_raw_inputs', False):
            model_kwargs['raw_texts'] = raw_texts
            model_kwargs['raw_imgs'] = raw_imgs

        return self.model(**model_kwargs)

    def train(self, train_loader, needs_img_features=True, class_weights=None):
        self.model.train()
        loss_list = []
        true_labels, pred_labels = [], []
        component_sums = {'main_loss': 0.0, 'aux_loss': 0.0, 'total_loss': 0.0}
        component_steps = 0

        if class_weights is not None:
            self.loss_func = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.loss_func = nn.CrossEntropyLoss(
                weight=torch.tensor(self.config.loss_weight, dtype=torch.float32).to(self.device))
        if hasattr(self.model, 'loss_func'):
            self.model.loss_func = self.loss_func

        self.class_loss_tracker.clear()

        accumulation_steps = self.config.accumulation_steps
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            guids, texts, texts_mask, imgs, labels, img_features, raw_texts, raw_imgs = self._unpack_batch(batch)
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(
                self.device), labels.to(self.device)
            labels = labels.long()

            if needs_img_features:
                img_features = img_features.to(self.device)

            pred, loss = self._call_model(
                texts, texts_mask, imgs, labels, img_features, needs_img_features, include_labels=True,
                raw_texts=raw_texts, raw_imgs=raw_imgs)

            if torch.isnan(loss):
                print("Loss is NaN")
                continue

            if torch.isnan(pred).any():
                print("Predictions contain NaN")
                continue

            loss_list.append(loss.item())
            current_breakdown = self._read_model_loss_breakdown(loss.item())
            for key in component_sums:
                component_sums[key] += current_breakdown[key]
            component_steps += 1
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.argmax(dim=1).tolist())

            for i, label in enumerate(labels):
                self.class_loss_tracker[label.item()].append(loss.item())

            (loss / accumulation_steps).backward()

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_loss = round(sum(loss_list) / len(loss_list), 5) if loss_list else float('nan')
        train_acc = accuracy_score(true_labels, pred_labels) if true_labels else 0.0
        train_f1 = f1_score(true_labels, pred_labels, average='weighted') if true_labels else 0.0
        if component_steps > 0:
            self.last_train_loss_components = {
                key: component_sums[key] / component_steps for key in component_sums
            }
        else:
            self.last_train_loss_components = {'main_loss': 0.0, 'aux_loss': 0.0, 'total_loss': 0.0}

        return train_loss, train_acc, train_f1, loss_list

    def valid(self, val_loader, needs_img_features=True):
        print("开始验证阶段，批次数量:", len(val_loader))
        self.model.eval()
        val_loss = 0
        true_labels, pred_labels = [], []
        component_sums = {'main_loss': 0.0, 'aux_loss': 0.0, 'total_loss': 0.0}
        component_steps = 0

        self.class_loss_tracker.clear()

        total_batches = len(val_loader)
        for batch_count, batch in enumerate(val_loader, start=1):
            guids, texts, texts_mask, imgs, labels, img_features, raw_texts, raw_imgs = self._unpack_batch(batch)
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(
                self.device), labels.to(self.device)
            labels = labels.long()

            if needs_img_features:
                img_features = img_features.to(self.device)

            self._log_numeric_progress("验证", batch_count, total_batches)
            if batch_count % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                pred, loss = self._call_model(
                    texts, texts_mask, imgs, labels, img_features, needs_img_features, include_labels=True,
                    raw_texts=raw_texts, raw_imgs=raw_imgs)

            if torch.isnan(loss):
                print("Validation Loss is NaN")
                continue

            if torch.isnan(pred).any():
                print("Validation Predictions contain NaN")
                continue

            val_loss += loss.item()
            current_breakdown = self._read_model_loss_breakdown(loss.item())
            for key in component_sums:
                component_sums[key] += current_breakdown[key]
            component_steps += 1
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.argmax(dim=1).tolist())

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1

        if not true_labels or not pred_labels:
            print("警告：验证阶段没有获取到标签或预测结果！")
            return 0.0, 0.0, 0.0, 0.0, {
                'weighted_precision': 0.0,
                'weighted_recall': 0.0,
                'macro_precision': 0.0,
                'macro_recall': 0.0,
                'confusion_matrix': None,
            }

        val_acc = accuracy_score(true_labels, pred_labels) if true_labels else 0.0
        val_f1 = f1_score(true_labels, pred_labels, average='weighted') if true_labels else 0.0
        val_macro_f1 = f1_score(true_labels, pred_labels, average='macro') if true_labels else 0.0
        if component_steps > 0:
            self.last_valid_loss_components = {
                key: component_sums[key] / component_steps for key in component_sums
            }
        else:
            self.last_valid_loss_components = {'main_loss': 0.0, 'aux_loss': 0.0, 'total_loss': 0.0}
        weighted_precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        weighted_recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        macro_precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
        macro_recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(self.config.num_labels)))

        print(
            f"验证完成 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Weighted F1: {val_f1:.4f}, "
            f"Macro F1: {val_macro_f1:.4f}, Weighted Precision: {weighted_precision:.4f}, "
            f"Weighted Recall: {weighted_recall:.4f}, Macro Precision: {macro_precision:.4f}, "
            f"Macro Recall: {macro_recall:.4f}")

        return val_loss, val_acc, val_f1, val_macro_f1, {
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'confusion_matrix': cm,
        }

    def update_class_weights(self, min_weight=1.0, max_weight=10.0):
        return torch.ones(self.config.num_labels, dtype=torch.float).to(self.device)

    def predict(self, test_loader, needs_img_features=True):
        self.model.eval()
        pred_guids, pred_labels = [], []

        total_batches = len(test_loader)
        for batch_count, batch in enumerate(test_loader, start=1):
            guids, texts, texts_mask, imgs, labels, img_features, raw_texts, raw_imgs = self._unpack_batch(batch)
            texts, texts_mask, imgs = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device)
            if needs_img_features:
                img_features = img_features.to(self.device)
            with torch.no_grad():
                pred = self._call_model(
                    texts, texts_mask, imgs, labels=None, img_features=img_features,
                    needs_img_features=needs_img_features, include_labels=False,
                    raw_texts=raw_texts, raw_imgs=raw_imgs)
            pred_guids.extend(guids)
            pred_labels.extend(pred.argmax(dim=1).tolist())
            self._log_numeric_progress("预测", batch_count, total_batches)

        return [(guid, label) for guid, label in zip(pred_guids, pred_labels)]

    def collect_predictions(self, data_loader, needs_img_features=True):
        self.model.eval()
        true_labels, pred_labels, guids_all = [], [], []
        raw_texts_all = []

        start_time = time.time()
        num_samples = 0
        total_batches = len(data_loader)

        with torch.no_grad():
            for batch_count, batch in enumerate(data_loader, start=1):
                guids, texts, texts_mask, imgs, labels, img_features, raw_texts, raw_imgs = self._unpack_batch(batch)
                texts = texts.to(self.device)
                texts_mask = texts_mask.to(self.device)
                imgs = imgs.to(self.device)
                num_samples += len(guids)

                if needs_img_features:
                    img_features = img_features.to(self.device)

                prob_vec = self._call_model(
                    texts, texts_mask, imgs, labels=None, img_features=img_features,
                    needs_img_features=needs_img_features, include_labels=False,
                    raw_texts=raw_texts, raw_imgs=raw_imgs)

                pred_labels.extend(prob_vec.argmax(dim=1).cpu().tolist())
                true_labels.extend(labels.tolist())
                guids_all.extend(guids)
                raw_texts_all.extend(raw_texts if raw_texts is not None else [None] * len(guids))
                self._log_numeric_progress("收集预测", batch_count, total_batches)

        end_time = time.time()
        inference_time = end_time - start_time
        avg_ms_per_sample = (inference_time / num_samples) * 1000 if num_samples > 0 else 0.0
        if num_samples > 0:
            print(f"推理完成。总耗时: {inference_time:.2f}秒，平均耗时: {avg_ms_per_sample:.2f}毫秒/样本")

        return {
            'guids': guids_all,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'raw_texts': raw_texts_all,
            'inference_time': inference_time,
            'avg_ms_per_sample': avg_ms_per_sample,
            'num_samples': num_samples,
        }

    def get_predictions(self, data_loader, needs_img_features=True, return_metadata=False):
        results = self.collect_predictions(data_loader, needs_img_features)
        if return_metadata:
            return results['true_labels'], results['pred_labels'], results
        return results['true_labels'], results['pred_labels']

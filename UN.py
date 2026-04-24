import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision.models import resnet50

class TextModel(nn.Module):
    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.dropout = nn.Dropout(config.bert_dropout)
        self.trans = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

        freeze_text_backbone = getattr(config, 'fixed_text_model_params', False)
        for param in self.bert.parameters():
            param.requires_grad = not freeze_text_backbone

        self.credibility_fc = nn.Sequential(
            nn.Linear(config.middle_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, bert_inputs, masks, token_type_ids=None, compute_credibility=True):
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        hidden_state = bert_out['last_hidden_state']
        pooler_out = bert_out['pooler_output']

        text_feature = self.trans(self.dropout(pooler_out))

        credibility_score = None
        if compute_credibility:
            credibility_score = self.credibility_fc(text_feature)

        return self.trans(hidden_state), text_feature, credibility_score


class ImageModel(nn.Module):
    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.full_resnet = resnet50(pretrained=True)
        self.resnet_h = nn.Sequential(
            *(list(self.full_resnet.children())[:-2])
        )
        self.resnet_p = nn.Sequential(
            list(self.full_resnet.children())[-2],
            nn.Flatten()
        )

        if config.use_cbam:
            self.cbam = CBAM(
                channels=2048,
                reduction_ratio=config.cbam_reduction_ratio,
                kernel_size=config.cbam_kernel_size
            )
        else:
            self.cbam = IdentityModule()

        self.hidden_trans = nn.Sequential(
            nn.Conv2d(2048, config.img_hidden_seq, 1),
            nn.BatchNorm2d(config.img_hidden_seq),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=2),
            nn.Dropout(config.resnet_dropout),
            nn.Linear(7 * 7, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

        self.trans = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(2048, config.middle_hidden_size),
            nn.BatchNorm1d(config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

        freeze_image_backbone = getattr(
            config,
            'fixed_img_model_params',
            getattr(config, 'fixed_image_model_params', False)
        )
        for param in self.full_resnet.parameters():
            param.requires_grad = not freeze_image_backbone

    def forward(self, imgs, img_features=None):
        hidden_state = self.resnet_h(imgs)

        enhanced_features = self.cbam(hidden_state)

        feature = self.resnet_p(enhanced_features)
        hidden_state_transformed = self.hidden_trans(enhanced_features)

        return hidden_state_transformed, self.trans(feature)

class IdentityModule(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction_ratio=reduction_ratio
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        channel_out = self.sigmoid(avg_out + max_out)
        return x * channel_out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.spatial_feature_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 基础空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_feature_conv(spatial)

        # 简化：只使用CNN特征的空间注意力
        spatial_out = self.sigmoid(spatial)
        return x * spatial_out


class BiAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(BiAttention, self).__init__()
        self.text2img = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.img2text = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.img_norm = nn.LayerNorm(embed_dim)

    def forward(self, text_feats, img_feats, text_mask):
        text_feats = text_feats.permute(1, 0, 2)
        img_feats = img_feats.permute(1, 0, 2)
        valid_text_mask = text_mask.bool()
        key_padding_mask = ~valid_text_mask

        attn_output_t2i, _ = self.text2img(text_feats, img_feats, img_feats)
        attn_output_i2t, _ = self.img2text(img_feats, text_feats, text_feats, key_padding_mask=key_padding_mask)

        attn_output_t2i = self.text_norm(attn_output_t2i + text_feats).permute(1, 0, 2)
        attn_output_i2t = self.img_norm(attn_output_i2t + img_feats).permute(1, 0, 2)

        text_lengths = valid_text_mask.sum(dim=1, keepdim=True).clamp_min(1).to(attn_output_t2i.dtype)
        text_attended = (attn_output_t2i * valid_text_mask.unsqueeze(-1)).sum(dim=1) / text_lengths
        img_attended = torch.mean(attn_output_i2t, dim=1)

        return attn_output_t2i, attn_output_i2t, text_attended, img_attended


class SharedEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(SharedEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, features, padding_mask=None):
        features = features.permute(1, 0, 2)
        output = self.transformer_encoder(features, src_key_padding_mask=padding_mask)
        output = output.permute(1, 0, 2)
        if padding_mask is None:
            return torch.mean(output, dim=1)

        valid_tokens = (~padding_mask).unsqueeze(-1).to(output.dtype)
        denom = valid_tokens.sum(dim=1).clamp_min(1.0)
        return (output * valid_tokens).sum(dim=1) / denom


# Model类修改
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.text_model = TextModel(config)
        self.img_model = ImageModel(config)

        self.bi_attention = BiAttention(config.middle_hidden_size, config.attention_nhead, config.attention_dropout)
        self.shared_encoder = SharedEncoder(config.middle_hidden_size, config.attention_nhead, config.attention_dropout)

        self.gate = nn.Sequential(
            nn.Linear(config.middle_hidden_size * 2, config.middle_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(config.middle_hidden_size, 2),
            nn.Softmax(dim=1)
        )
        self.static_weight_logits = nn.Parameter(torch.zeros(2))
        self.adaptive_weight_predictor = nn.Sequential(
            nn.Linear(config.middle_hidden_size * 3 + 2, config.middle_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size, 2)
        )
        self.fusion_norm = nn.LayerNorm(config.middle_hidden_size)

        self.classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        self.entropy_probe = nn.Linear(config.middle_hidden_size, config.num_labels)

        self.loss_func = nn.CrossEntropyLoss(weight=torch.tensor(config.loss_weight).to(config.device))
        self.aux_loss_weight = float(getattr(config, 'aux_loss_weight', 0.2))
        self.image_correction_scale = float(getattr(config, 'image_correction_scale', 0.35))
        self.gate_temperature = float(getattr(config, 'gate_temperature', 8.0))
        self.logit_blend_ratio = float(getattr(config, 'logit_blend_ratio', 0.5))
        self.last_loss_breakdown = None

    def _compute_weighted_fusion(self, text_feature, img_feature, credibility_score,
                                 text_attended=None, img_attended=None, joint_representation=None, strategy=None):
        strategy = strategy or getattr(self.config, 'weighting_strategy', 'text_confidence_softmax')

        if strategy == 'fixed_50_50':
            weight_values = text_feature.new_tensor([[0.5, 0.5]]).expand(text_feature.size(0), -1)
        elif strategy == 'learned_static':
            weight_values = torch.softmax(self.static_weight_logits, dim=0).unsqueeze(0).expand(text_feature.size(0), -1)
        elif strategy == 'text_confidence_softmax':
            if credibility_score is None:
                raise ValueError('text_confidence_softmax 策略需要可用的文本置信度门控分数')
            confidence_prior = torch.cat([credibility_score, 1 - credibility_score], dim=1)
            if joint_representation is None:
                joint_representation = 0.5 * (text_feature + img_feature)
            adaptive_inputs = torch.cat(
                [
                    text_feature * credibility_score,
                    img_feature * (1 - credibility_score),
                    joint_representation,
                    confidence_prior,
                ],
                dim=1
            )
            adaptive_logits = self.adaptive_weight_predictor(adaptive_inputs)
            weight_values = torch.softmax(adaptive_logits + confidence_prior, dim=1)
        elif strategy == 'attention_magnitude':
            text_source = text_attended if text_attended is not None else text_feature
            img_source = img_attended if img_attended is not None else img_feature
            text_strength = text_source.norm(dim=1, keepdim=True)
            img_strength = img_source.norm(dim=1, keepdim=True)
            weight_values = torch.cat([text_strength, img_strength], dim=1)
            weight_values = weight_values / weight_values.sum(dim=1, keepdim=True).clamp_min(1e-8)
        elif strategy == 'entropy_confidence':
            text_logits = self.entropy_probe(text_feature.detach())
            img_logits = self.entropy_probe(img_feature.detach())
            text_prob = torch.softmax(text_logits, dim=-1)
            img_prob = torch.softmax(img_logits, dim=-1)
            max_entropy = torch.log(torch.tensor(float(self.config.num_labels), device=text_feature.device))
            text_conf = 1.0 - (-(text_prob * torch.log(text_prob.clamp_min(1e-8))).sum(dim=1, keepdim=True) / max_entropy)
            img_conf = 1.0 - (-(img_prob * torch.log(img_prob.clamp_min(1e-8))).sum(dim=1, keepdim=True) / max_entropy)
            weight_values = torch.cat([text_conf, img_conf], dim=1)
            weight_values = weight_values / weight_values.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            if credibility_score is None:
                raise ValueError('text_confidence 策略需要可用的文本置信度门控分数')
            weight_values = torch.cat([credibility_score, 1 - credibility_score], dim=1)

        fused_feature = weight_values[:, 0].unsqueeze(1) * text_feature + \
                        weight_values[:, 1].unsqueeze(1) * img_feature
        return fused_feature, weight_values

    def _compute_text_primary_logits(self, text_logits, image_logits, credibility_score):
        text_weight = credibility_score.squeeze(1).clamp(0.0, 1.0)
        image_correction = self.image_correction_scale * (1.0 - text_weight).unsqueeze(1) * (image_logits - text_logits)
        return text_logits + image_correction

    def forward(self, texts, texts_mask, imgs, img_features=None, labels=None):
        self.last_loss_breakdown = None
        # 检查输入确定实际使用的模态
        if texts is None and imgs is not None:
            temp_modality = 'image'
        elif texts is not None and imgs is None:
            temp_modality = 'text'
        else:
            temp_modality = self.config.modality

        if temp_modality == 'text':
            # 仅使用文本，不计算文本置信度门控分数
            text_hidden_state, text_feature, _ = self.text_model(texts, texts_mask, compute_credibility=False)
            final_feature = text_feature  # 直接使用文本特征

        elif temp_modality == 'image':
            # 仅使用图像
            img_hidden_state, img_feature = self.img_model(imgs)
            final_feature = img_feature  # 直接使用图像特征

        else:
            # both模式下需要计算文本置信度门控分数
            text_hidden_state, text_feature, credibility_score = self.text_model(texts, texts_mask, compute_credibility=True)
            img_hidden_state, img_feature = self.img_model(imgs)
            text_attended, img_attended = None, None

            # 根据config.use_bi_attention决定是否使用双向注意力机制
            if getattr(self.config, 'use_bi_attention', True):
                text_context_seq, img_context_seq, text_attended, img_attended = self.bi_attention(
                    text_hidden_state, img_hidden_state, texts_mask)
            else:
                text_context_seq = text_hidden_state
                img_context_seq = img_hidden_state
                valid_text_mask = texts_mask.bool()
                text_lengths = valid_text_mask.sum(dim=1, keepdim=True).clamp_min(1).to(text_hidden_state.dtype)
                text_attended = (text_hidden_state * valid_text_mask.unsqueeze(-1)).sum(dim=1) / text_lengths
                img_attended = torch.mean(img_hidden_state, dim=1)

            valid_text_mask = texts_mask.bool()
            text_padding_mask = ~valid_text_mask
            img_padding_mask = torch.zeros(
                (img_context_seq.size(0), img_context_seq.size(1)),
                dtype=torch.bool,
                device=img_context_seq.device
            )
            combined_features = torch.cat([text_context_seq, img_context_seq], dim=1)
            combined_padding_mask = torch.cat([text_padding_mask, img_padding_mask], dim=1)
            joint_representation = self.shared_encoder(combined_features, padding_mask=combined_padding_mask)

            use_credibility = getattr(self.config, 'use_credibility', True)
            credibility_eval = getattr(self.config, 'credibility_eval', use_credibility)
            weighting_strategy = getattr(self.config, 'weighting_strategy', 'text_confidence_softmax')
            if not (use_credibility and credibility_eval) and weighting_strategy in ('text_confidence', 'text_confidence_softmax'):
                weighting_strategy = 'fixed_50_50'

            fused_feature, _ = self._compute_weighted_fusion(
                text_feature, img_feature, credibility_score,
                text_attended=text_attended,
                img_attended=img_attended,
                joint_representation=joint_representation,
                strategy=weighting_strategy
            )

            gate_values = self.gate(torch.cat([fused_feature, joint_representation], dim=1))
            final_feature = (
                gate_values[:, 0].unsqueeze(1) * fused_feature +
                gate_values[:, 1].unsqueeze(1) * joint_representation
            )
            final_feature = self.fusion_norm(final_feature)

        # 分类
        if temp_modality == 'both':
            text_logits = self.classifier(text_feature)
            image_logits = self.classifier(img_feature)
            fused_logits = self.classifier(final_feature)
            text_primary_logits = self._compute_text_primary_logits(text_logits, image_logits, credibility_score)
            blend_ratio = min(max(self.logit_blend_ratio, 0.0), 1.0)
            logits = blend_ratio * text_primary_logits + (1.0 - blend_ratio) * fused_logits
        else:
            logits = self.classifier(final_feature)
        prob_vec = torch.softmax(logits, dim=-1)

        if labels is not None:
            main_loss = self.loss_func(logits, labels)
            aux_loss = logits.new_tensor(0.0)

            if temp_modality == 'both' and credibility_score is not None:
                with torch.no_grad():
                    text_only_logits = self.classifier(text_feature)
                    image_only_logits = self.classifier(img_feature)
                    text_label_prob = torch.softmax(text_only_logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
                    image_label_prob = torch.softmax(image_only_logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
                    target_credibility = torch.sigmoid(
                        self.gate_temperature * (text_label_prob - image_label_prob)
                    )

                aux_loss = F.binary_cross_entropy(
                    credibility_score.squeeze(1),
                    target_credibility.detach()
                )

            total_loss = main_loss + self.aux_loss_weight * aux_loss
            self.last_loss_breakdown = {
                'main_loss': float(main_loss.detach().item()),
                'aux_loss': float(aux_loss.detach().item()),
                'total_loss': float(total_loss.detach().item()),
            }
            return prob_vec, total_loss
        else:
            return prob_vec

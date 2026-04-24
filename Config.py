import os
import torch

class config:
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_path, 'data', 'data')
    train_data_path = os.path.join(root_path, 'data', 'train.json')
    test_data_path = os.path.join(root_path, 'data', 'test.json')
    output_path = os.path.join(root_path, 'output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None

    num_labels = 2
    loss_weight = [0.80, 0.20]
    modality = 'both'
    learning_rate = 3e-5
    weight_decay = 1e-4
    epoch = 50
    comparison_epochs = epoch
    ablation_epochs = epoch
    un_only_epochs = epoch
    accumulation_steps = 8
    early_stopping_metric = 'val_macro_f1'
    early_stopping_patience = 5
    early_stopping_min_delta = 0.002
    early_stopping_warmup = 8
    baseline_early_stopping_patience = 5
    baseline_early_stopping_min_delta = 0.0005
    auto_loss_weight = True
    aux_loss_weight = 0.0
    image_correction_scale = 0.0
    gate_temperature = 5.0
    logit_blend_ratio = 0.0

    credibility_eval = True
    use_bi_attention = True
    use_early_stopping = True
    use_credibility = credibility_eval  

    vocab_size = 30522
    embed_dim = 768
    lstm_hidden_size = 384
    lstm_layers = 2
    lstm_dropout = 0.2
    vgg_dropout = 0.3

    use_cbam = True

    fuse_model_type = 'UN'
    weighting_strategy = 'text_confidence_softmax'
    only = None
    middle_hidden_size = 128
    attention_nhead = 8
    attention_dropout = 0.2
    fuse_dropout = 0.50
    out_hidden_size = 64

    fixed_text_model_params = False
    bert_name = os.path.join(root_path, 'chinese_roberta_wwm_ext')
    bert_learning_rate = 5e-6
    bert_dropout = 0.10

    fixed_img_model_params = False
    image_size = 224
    fixed_image_model_params = fixed_img_model_params
    resnet_learning_rate = 3e-6
    resnet_dropout = 0.15
    img_hidden_seq = 64
    img_feature_dim = 0
    img_feature_norm = False

    cbam_reduction_ratio = 8
    cbam_kernel_size = 7
    cbam_dropout = 0.3

    scheduler_factor_text = 0.5
    scheduler_patience_text = 2
    scheduler_min_lr_text = 1e-7
    scheduler_factor_image = 0.5
    scheduler_patience_image = 3
    scheduler_min_lr_image = 5e-7
    scheduler_factor_multimodal = 0.5
    scheduler_patience_multimodal = 1
    scheduler_min_lr_multimodal = 1e-7

    checkout_params = {'batch_size': 16, 'shuffle': False}
    train_params = {'batch_size': 8, 'shuffle': True, 'num_workers': 0}
    val_params = {'batch_size': 8, 'shuffle': False, 'num_workers': 0}
    test_params = {'batch_size': 8, 'shuffle': False, 'num_workers': 0}

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    clip_model_name = os.path.join(root_path, 'CLIP')
    clip_tuning_mode = 'full'
    clip_freeze_backbone = False
    seeds = [42]

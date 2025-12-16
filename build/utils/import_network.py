import sys
try:
    import utils.networks as networks
except:
    import networks
import torch
import torch.nn as nn
import torch.nn.functional as F

networks_path = '//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/2025_XXX_XAI_STN_MERs/Scripts/utils/networks'
sys.path.append(networks_path)

def import_model(model_name: str, parameters=None):

    if model_name == 'CNN_PwMhAT':
        model = networks.CNN_PwMhAT(input_size=10000, patch_size=200,
                                    in_channels=1, out_channels=32,
                                    embed_dim=32, num_heads=2,
                                    output_dim=2, 
                                    dropout=0.1)
        lr = 0.0001
        bs = 32
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        # criterion = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        # num_epochs = 100
        # train_size = 60000
        # total_steps = (train_size // bs) * num_epochs
        # scheduler = cosine_warmup_scheduler(
        #         optimizer,
        #         warmup_steps = int(0.1 * total_steps),
        #         total_steps  = total_steps,
        #         min_lr = 0.01 # 1/100 * lr_0
        #     )

        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=10)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr/1000, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[10])

    if model_name == 'CNN_PwMhATv2':
        model = networks.CNN_PwMhATv2(input_size=10000, patch_size=200,
                                    in_channels=1,
                                    embed_dim=32, num_heads=2,
                                    output_dim=2)
        lr = 1e-3
        bs = 32
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-5)
        # criterion = torch.nn.BCEWithLogitsLoss()
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = FocalLoss(gamma=2.0)

    if model_name == 'CNN_PwMhATv3':
        model = networks.CNN_PwMhATv3()
        lr = 0.0001
        bs = 32
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

        # num_epochs = 100
        # train_size = 60000
        # total_steps = (train_size // bs) * num_epochs

        # scheduler = cosine_warmup_scheduler(
        #         optimizer,
        #         warmup_steps = int(0.1 * total_steps),
        #         total_steps  = total_steps,
        #         min_lr = 0.05
        #     )

        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=10)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr/1000, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[10])

    elif model_name == 'CNN_ReSpAT':
        model = networks.CNN_ReSpAT()
        lr = 0.0001
        bs = 20
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
        criterion = torch.nn.BCELoss()

        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=10)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr/1000, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[10])

    elif model_name == 'ResNet_AT':
        model = networks.ResNet_AT()
        lr = 1e-7 #1e-7
        bs = 128
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=lr, step_size_up=60000//bs, cycle_momentum=False)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=lr, step_size_up=10, cycle_momentum=False)
        criterion = torch.nn.BCEWithLogitsLoss()

    elif model_name == 'ML_STIM':
        model = networks.ML_STIM()
        lr = 0.001
        bs = 32
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=10)
        criterion = torch.nn.BCEWithLogitsLoss()

    elif model_name == 'ConvMixer1D':
        model = networks.ConvMixer1D(in_channels=1, 
                                     dim=128, depth=4, patch_size=200, kernel_size=5, 
                                     num_classes=2, 
                                     input_length=10000)
        lr = 0.01
        bs = 32
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=lr, step_size_up=10)
        criterion = torch.nn.BCEWithLogitsLoss()

    # num_epochs = 100
    # train_size = 60000
    # total_steps = (train_size // bs) * num_epochs

    # scheduler = cosine_warmup_scheduler(
    #         optimizer,
    #         warmup_steps = int(0.1 * total_steps),
    #         total_steps  = total_steps,
    #         min_lr = 0.05
    #     )

    return model, lr, bs, scheduler, optimizer, criterion

def save_model_architecture_image(model, output_path=None):
    from apply_transform import transform_data
    import numpy as np
    import torch
    from torchview import draw_graph

    model_name = model.__class__.__name__
    sample_input = np.random.randn(1, 10000)
    fsamp = 10000
    sample_input, _ = transform_data(sample_input, fsamp, model, verbose=False, features_extremes=None)
    sample_input = torch.tensor(sample_input, dtype=torch.float32)
    if len(sample_input.shape) == 1:
        sample_input = torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0)
    if output_path is None:
        output_path = f'{model_name}_architecture'

    model_graph = draw_graph(
        model,
        input_data=sample_input,
        expand_nested=True,
        depth=3,
        save_graph=True,
        filename=output_path,
        directory='.',
        device=sample_input.device
    )
    model_graph.visual_graph.render(output_path, format='png', cleanup=True)

def cosine_warmup_scheduler(
    optimizer,
    warmup_steps,
    total_steps,
    min_lr=0.0
):
    """
    Cosine annealing LR with warmup.
    - LR starts at 0.
    - Warms up to base LR linearly.
    - Then cosine decay down to min_lr.
    """

    import math
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            # linear warmup
            return step / warmup_steps
        
        # cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        
        # scale cosine curve so it ends at min_lr/base_lr
        return cosine * (1 - min_lr) + min_lr

    return LambdaLR(optimizer, lr_lambda)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean()

if __name__ == "__main__":
    import os
    # model_list = ['CNN_CBAM', 'ML_STIM', 'ResCNN_PwMhAT', 'CNN_PwMhAT', 'ResCNN_SpAT', 'ResNet_AT', 'ConvMixer1D']
    model_list = ['CNN_PwMhATv3']
    output_folder = 'utils/networks/architectures_png'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for model_name in model_list:
        model,_,_,_,_,_ = import_model(model_name)
        output_path = f'{output_folder}/{model_name}'
        save_model_architecture_image(model, output_path)
        print(f'Saved architecture for {model_name} at {output_path}')
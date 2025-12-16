import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Workaround for OpenMP conflict
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Use offscreen Qt platform

import torch
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of Qt
import matplotlib.pyplot as plt

epochs = 100
lr0 = 1e-7 #0.0005
bs = 32
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr0/10, last_epoch=-1)

# scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)
# scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr0/1000, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[10])
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-8, max_lr=lr0, step_size_up=10, cycle_momentum=False)

lr = []

for epoch in range(epochs):
    lr.append(optimizer.param_groups[0]['lr'])
    # Simulate a training step
    optimizer.zero_grad()
    loss = model(torch.randn(10, 10)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Plot the learning rate evolution
plt.plot(range(epochs), lr, linewidth=2)
plt.show()

print(f'Initial LR: {lr[0]:.6f}')
print(f'Final LR: {lr[-1]:.6f}')
print(f'LR reduction factor: {lr[-1]/lr[0]:.6f}')


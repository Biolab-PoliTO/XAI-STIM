import torch

class ML_STIM(torch.nn.Module):
    def __init__(self, input_size=9, output_size=2):
        super(ML_STIM, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 7)
        self.fc2 = torch.nn.Linear(7, 4)
        self.fc3 = torch.nn.Linear(4, 4)
        self.fc4 = torch.nn.Linear(4, 2)
        self.fc5 = torch.nn.Linear(2, output_size)

    def forward(self, x):
        
        # Pass through the neural network
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x, 0

def main():
    """
    ...
    """
    from torchinfo import summary
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model instance
    model = ML_STIM(input_size=9)
    model.to(device)

    model_summary = summary(
        model, 
        input_size=(1, 9),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=1
    )
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n2. PARAMETER COUNT:")
    print("-" * 40)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

if __name__ == "__main__":
    main()
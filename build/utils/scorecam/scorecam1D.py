import torch
import torch.nn.functional as F

class ScoreCAM1D:
    """
    ScoreCAM for 1D signals (time-series data).
    Adapted from the original 2D ScoreCAM implementation.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: The layer to compute activations from (e.g., patch_encoder)
        """
        self.model = model
        self.model.eval()
        
        self.activations = None
        self.target_layer = target_layer
        
        # Register forward hook to capture activations
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        self.target_layer.register_forward_hook(forward_hook)
    
    def forward(self, input, class_idx=None):
        """
        Generate ScoreCAM saliency map for 1D input.
        
        Args:
            input: Input tensor of shape (batch, length) or (batch, 1, length)
            class_idx: Target class index. If None, use predicted class.
            
        Returns:
            score_saliency_map: Normalized saliency map of shape (1, 1, length)
        """
        # Ensure input has channel dimension
        if input.dim() == 2:
            input = input.unsqueeze(1)  # (batch, 1, length)
        
        b, c, length = input.size()
        
        # Get prediction on raw input
        with torch.no_grad():
            output, _ = self.model(input)
            logit = output
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
        else:
            predicted_class = torch.LongTensor([class_idx]).to(input.device)
        
        # Get activations from target layer
        # activations shape: (b * num_patches, channels, temporal_dim)
        with torch.no_grad():
            _ = self.model(input)
            activations = self.activations
        
        # Handle the patch-based architecture
        # After patch_encoder: (B*num_patches, out_channels, 1)
        if activations.dim() == 3:
            activations = activations.squeeze(-1)  # (B*num_patches, out_channels)
        
        # Reshape to (B, num_patches, out_channels)
        num_patches = self.model.num_patches
        activations = activations.view(b, num_patches, -1)  # (B, num_patches, channels)
        
        k = activations.size(2)  # number of feature channels
        
        score_saliency_map = torch.zeros((1, 1, length)).to(input.device)
        
        with torch.no_grad():
            for i in range(k):
                # Get i-th channel activation across all patches
                # Shape: (B, num_patches)
                channel_activation = activations[:, :, i]
                
                # Upsample to match input length
                # Reshape to (B, 1, num_patches) for interpolation
                saliency_map = channel_activation.unsqueeze(1)  # (B, 1, num_patches)
                
                # Interpolate to input length
                saliency_map = F.interpolate(
                    saliency_map, 
                    size=length, 
                    mode='linear', 
                    align_corners=False
                )  # (B, 1, length)
                
                if saliency_map.max() == saliency_map.min():
                    continue
                
                # Normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                
                # Prediction on masked input
                masked_input = input * norm_saliency_map
                output, _ = self.model(masked_input)
                output = F.softmax(output, dim=1)
                score = output[0][predicted_class]
                
                score_saliency_map += score * saliency_map
        
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()
        
        if score_saliency_map_min == score_saliency_map_max:
            return None
        
        # Normalize to 0-1
        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (score_saliency_map_max - score_saliency_map_min)
        
        return score_saliency_map
    
    def __call__(self, input, class_idx=None):
        return self.forward(input, class_idx)
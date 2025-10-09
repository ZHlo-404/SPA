import clip
import torch
import torch.nn as nn


class CustomImageEncoder(torch.nn.Module):
    def __init__(self, clip_model: clip.model.CLIP, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        
        self.conv1 = clip_model.visual.conv1
        self.visual_transformer = clip_model.visual.transformer
        self.visual_positional_embedding = clip_model.visual.positional_embedding
        
        width = 768
        scale = width ** -0.5

        self.class_embedding = clip_model.visual.class_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

    def forward(self, images):
        """The forward function to compute representations for the images.

        Args:
            images (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The global feature and patch-wise features.
        """
        bs = images.shape[0]

        x = self.conv1(images)  # Convolutional layer
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten grid dimensions
        x = x.permute(0, 2, 1)  # Permute to [batch_size, num_patches, feature_dim]

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.visual_positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)

        x = self.visual_transformer(x)
        
        x = x.permute(1, 0, 2)

        img_features = self.ln_post(x)

        cls_feats = img_features[:, 0, :]

        if self.proj is not None:
            cls_feats = cls_feats @ self.proj

        return cls_feats, img_features[:, 1:, :]
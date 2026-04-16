import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from vector_quantize_pytorch import VectorQuantize


class ResidualBlock(nn.Module):
    """
    Bloco residual

    Estrutura:
    - Normalização (GroupNorm) + ativação (SiLU) + convolução 3x3.
    - Segunda sequência de normalização + ativação + dropout + convolução 3x3.
    - Conexão residual (skip connection) somando a entrada original à saída,
      com projeção 1x1 caso o número de canais mude.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)


        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)


        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)
    
class Downsample(nn.Module):
    #definindo downsampling com conv strided
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    #definindo upsampling com interpolação + conv
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, channels):
        self.channels = channels


class Encoder(nn.Module):
    def __init__(self, in_channels = 3, base_channels=32, latent_dim = 128):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size = 3, padding = 1)

        self.block1 = ResidualBlock(base_channels, base_channels)
        self.down1 = Downsample(base_channels)

        self.block2 = ResidualBlock(base_channels, base_channels*2)
        self.down2 = Downsample(base_channels*2)

        self.block3 = ResidualBlock(base_channels*2, base_channels*4)
        self.down3 = Downsample(base_channels*4)

        self.conv_out = nn.Conv2d(base_channels * 4, latent_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.down3(x)
        x = self.conv_out(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, base_channels=32, out_channels=3):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, base_channels*4, kernel_size = 3, padding = 1)

        self.block1 = ResidualBlock(base_channels*4, base_channels*4)
        self.up1 = Upsample(base_channels*4)

        self.block2 = ResidualBlock(base_channels*4, base_channels*2)
        self.up2 = Upsample(base_channels*2)

        self.block3 = ResidualBlock(base_channels*2, base_channels)
        self.up3 = Upsample(base_channels)

        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.block1(x)
        x = self.up1(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.block3(x)
        x = self.up3(x)
        x = self.conv_out(x)

        return torch.sigmoid(x)
    


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, num_embeddings=512):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim=latent_dim)
        self.vq = VectorQuantize(
            dim = latent_dim,
            codebook_size = num_embeddings,
            decay = 0.8,             
            commitment_weight = 0.25, 
            kmeans_init = False,      # <--- A MUDANÇA É APENAS AQUI!
            use_cosine_sim = True,
            accept_image_fmap = True 
        )
        self.decoder = Decoder(latent_dim, out_channels=in_channels)
    def forward(self, x):
        # 1. Passa pelo Encoder
        z = self.encoder(x)
        
        # 3. Passa pelo VQ (Nota: a ordem correta de retorno é quantizado, indices, loss)
        quantized, indices, vq_loss = self.vq(z)
        
        # 5. Reconstrói a imagem
        x_recon = self.decoder(quantized)
        
        return x_recon, vq_loss, indices

    def decode_indices(self, indices):
        """
        Converts discrete token indices back into a continuous image.
        
        Args:
            indices: Tensor of shape [batch_size, num_tokens] (e.g., [1, 64])
                     or [batch_size, height, width] (e.g., [1, 8, 8]).
        Returns:
            reconstructed_images: Tensor of shape [batch_size, channels, H, W]
        """
        import math
        
        # 1. Flatten indices if they come in as 2D spatial grids (e.g., B x 8 x 8 -> B x 64)
        if indices.dim() == 3:
            indices = indices.view(indices.size(0), -1)
            
        batch_size, seq_len = indices.shape
        
        # 2. Look up the continuous embeddings from the codebook
        # FIX APPLIED HERE: using self.vq instead of self.quantizer
        z_q = self.vq.embedding(indices) # Shape: [B, seq_len, embedding_dim]
        
        # 3. Reshape the 1D sequence back into a 2D spatial grid for the CNN decoder
        embedding_dim = z_q.shape[-1]
        grid_size = int(math.sqrt(seq_len)) # e.g., sqrt(64) = 8
        
        # Reshape to [B, H, W, C]
        z_q = z_q.view(batch_size, grid_size, grid_size, embedding_dim)
        
        # Permute to [B, C, H, W] because PyTorch convolutions expect channels first
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # 4. Pass through the decoder to get the final image pixels
        reconstructed_images = self.decoder(z_q)
        
        return reconstructed_images
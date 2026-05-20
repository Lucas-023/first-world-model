import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(channels):

    groups = min(32, channels)

    return nn.GroupNorm(
        num_groups=groups,
        num_channels=channels,
        eps=1e-6,
        affine=True
    )


class ResnetBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.0
    ):
        super().__init__()

        self.norm1 = Normalize(in_channels)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1
        )

        self.norm2 = Normalize(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                1
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttnBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.norm = Normalize(channels)

        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):

        h = self.norm(x)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        B, C, H, W = q.shape

        q = q.reshape(B, C, H * W)
        q = q.permute(0, 2, 1)

        k = k.reshape(B, C, H * W)

        attn = torch.bmm(q, k)

        attn = attn * (C ** -0.5)

        attn = torch.softmax(attn, dim=-1)

        v = v.reshape(B, C, H * W)

        attn = attn.permute(0, 2, 1)

        h = torch.bmm(v, attn)

        h = h.reshape(B, C, H, W)

        h = self.proj_out(h)

        return x + h


class Downsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="nearest"
        )

        return self.conv(x)


class Encoder(nn.Module):

    # 64x64 -> 4x4

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        latent_dim=256,
        dropout=0.0
    ):
        super().__init__()

        ch_mult = [1, 2, 4, 4]

        self.conv_in = nn.Conv2d(
            in_channels,
            base_channels,
            3,
            padding=1
        )

        blocks = []

        curr_res = 64
        in_ch = base_channels

        for mult in ch_mult:

            out_ch = base_channels * mult

            for _ in range(2):

                blocks.append(
                    ResnetBlock(
                        in_ch,
                        out_ch,
                        dropout
                    )
                )

                in_ch = out_ch

                if curr_res in [16, 8]:
                    blocks.append(
                        AttnBlock(in_ch)
                    )

            if curr_res != 4:

                blocks.append(
                    Downsample(in_ch)
                )

                curr_res //= 2

        self.blocks = nn.Sequential(*blocks)

        self.norm_out = Normalize(in_ch)

        self.conv_out = nn.Conv2d(
            in_ch,
            latent_dim,
            3,
            padding=1
        )

    def forward(self, x):

        x = self.conv_in(x)

        x = self.blocks(x)

        x = self.norm_out(x)

        x = nonlinearity(x)

        x = self.conv_out(x)

        return x


class Decoder(nn.Module):

    # 4x4 -> 64x64

    def __init__(
        self,
        out_channels=3,
        base_channels=64,
        latent_dim=256,
        dropout=0.0
    ):
        super().__init__()

        ch_mult = [4, 4, 2, 1]

        in_ch = base_channels * 4

        self.conv_in = nn.Conv2d(
            latent_dim,
            in_ch,
            3,
            padding=1
        )

        blocks = []

        curr_res = 4

        for mult in ch_mult:

            out_ch = base_channels * mult

            for _ in range(2):

                blocks.append(
                    ResnetBlock(
                        in_ch,
                        out_ch,
                        dropout
                    )
                )

                in_ch = out_ch

                if curr_res in [8, 16]:
                    blocks.append(
                        AttnBlock(in_ch)
                    )

            if curr_res != 64:

                blocks.append(
                    Upsample(in_ch)
                )

                curr_res *= 2

        self.blocks = nn.Sequential(*blocks)

        self.norm_out = Normalize(in_ch)

        self.conv_out = nn.Conv2d(
            in_ch,
            out_channels,
            3,
            padding=1
        )

    def forward(self, x):

        x = self.conv_in(x)

        x = self.blocks(x)

        x = self.norm_out(x)

        x = nonlinearity(x)

        x = self.conv_out(x)

        return x


class VQVAE(nn.Module):

    def __init__(
        self,
        in_channels=3,
        latent_dim=256,
        num_embeddings=512
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=64,
            latent_dim=latent_dim
        )

        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=num_embeddings,
            decay=0.99,
            commitment_weight=0.25,
            use_cosine_sim=True,
            accept_image_fmap=True
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=64,
            latent_dim=latent_dim
        )

    def forward(self, x):

        z = self.encoder(x)

        quantized, indices, vq_loss = self.vq(z)

        x_recon = self.decoder(quantized)

        return x_recon, vq_loss, indices

    @torch.no_grad()
    def encode_indices(self, x):

        z = self.encoder(x)

        _, indices, _ = self.vq(z)

        return indices

    @torch.no_grad()
    def decode_indices(self, indices):

        B = indices.shape[0]

        indices = indices.view(B, 4, 4)

        z_q = self.vq.get_codes_from_indices(indices)

        return self.decoder(z_q)
import torch
import torch.nn as nn
import pywt
import numpy as np


class DWTLayer(nn.Module):
    """
    Aplica la DWT de forma vectoritzada sobre un batch d’imatges grayscale.
    Concatena les subbandes LL, LH, HL, HH → (B, 4, H/2, W/2)
    """
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

    def forward(self, x):
        B, C, H, W = x.shape
        if C != 1:
            raise ValueError("DWTLayer només suporta imatges grayscale (1 canal).")

        x_np = x.squeeze(1).cpu().numpy()  # (B, H, W)
        dwt_out = []
        for img in x_np:
            LL, (LH, HL, HH) = pywt.dwt2(img, self.wavelet)
            stacked = np.stack([LL, LH, HL, HH], axis=0)
            dwt_out.append(stacked)

        dwt_np = np.stack(dwt_out, axis=0)  # (B, 4, H/2, W/2)
        dwt_tensor = torch.from_numpy(dwt_np).to(x.device).float()
        return dwt_tensor


class PatchEmbedding(nn.Module):
    """
    Converteix imatges en seqüències de vectors (patches → embeddings).
    """
    def __init__(self, in_channels, patch_size=4, emb_size=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)
        self.transpose = lambda x: x.transpose(1, 2)  # (B, C, N) → (B, N, C)

    def forward(self, x):
        x = self.proj(x)
        x = self.flatten(x)
        x = self.transpose(x)
        return x


class DWTFormer(nn.Module):
    """
    Arquitectura híbrida basada en DWT + Transformer per a classificació d’imatges mèdiques.
    """
    def __init__(self, num_classes=9, img_size=28, patch_size=4, emb_size=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.dwt = DWTLayer()
        self.dwt_norm = nn.LayerNorm([4, img_size // 2, img_size // 2])
        self.patch_embedding = PatchEmbedding(in_channels=4, patch_size=patch_size, emb_size=emb_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.dwt(x)                    # (B, 4, H/2, W/2)
        x = self.dwt_norm(x)              # normalització post-DWT
        x = self.patch_embedding(x)       # (B, N, emb_size)

        B, N, E = x.shape
        cls_tokens = self.cls_token.expand(B, 1, E)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer(x)
        x = self.norm(x[:, 0])            # token de classificació [CLS]
        return self.head(x)


# ✅ Exemple d’ús (test ràpid)
if __name__ == '__main__':
    with torch.no_grad():
        model = DWTFormer(num_classes=9)
        dummy_input = torch.randn(8, 1, 28, 28)
        output = model(dummy_input)
        print("Sortida:", output.shape)  # → (8, 9)

import torch,torchvision
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_channels : int, image_size : int = 48, patch_size : int = 16, embed_dim : int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (image_size // patch_size) ** 2  
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, embed_dim))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        x = self.proj(x)
        x = torch.flatten(x, 2)
        x = x.transpose(2, 1)

        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat([cls_token, x], dim = 1)
        x += self.pos_embed

        return x

# p = PatchEmbed(3)
# x = torch.rand(2, 3, 48, 48)
# s = p(x)
# print(s)
# print(s.shape)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_p = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(p = dropout_p)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

# m = MLP(768, 128, 7)
# t = torch.rand(2, 10, 768)
# t2 = m(t)
# print(t2.shape)

# attn = nn.MultiheadAttention(embed_dim = 768, num_heads = 3)
# attn_output, attn_weights = attn(t, t, t)
# print(attn_output)
# print(attn_output.shape) 

class Transformer(nn.Module):
    def __init__(
        self, embed_dim : int, num_heads : int, 
        mlp_ratio = 4.0, attn_dropout = 0., pos_dropout = 0., qkv_bias = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.attn =nn.MultiheadAttention(embed_dim = embed_dim, num_heads = num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x1 = self.norm1(x)
        attn_output, attn_weights = self.attn(x1, x1, x1)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))

        return x

# T = Transformer(768, 3)
# t2 = T(t)
# print(t2)
# print(t2.shape)
class vit(nn.Module):
    def __init__(
        self, in_channels : int = 3, image_size : int = 48, patch_size : int = 16, 
        num_classes : int = 7, embed_dim : int = 768, 
        num_heads : int = 12, layers : int = 12,
        mlp_ratio = 4.0, attn_dropout = 0., pos_dropout = 0., qkv_bias = True):

        super().__init__()

        self.input_embed = PatchEmbed(in_channels, image_size, patch_size, embed_dim)
        ls = [Transformer(embed_dim, num_heads, mlp_ratio, attn_dropout, pos_dropout, qkv_bias) for _ in range(layers)]
        self.transformer_encoder = nn.Sequential(*ls)

        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.input_embed(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        x = self.head(cls_token)

        return x

# v = vit()
# t = torch.rand(2, 3, 16, 16)
# res = v(t)
# print(res)
# print(res.shape)



# print(torch.__version__)
import torch 
from torch import nn 

class LinearScheduler:
    def __init__(self, beta_start, beta_end, steps, device):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.steps = steps 
        self.betas = torch.linspace(beta_start, beta_end, steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
    def add_noise(self, original, noise, t):
        original_shape = original.shape
        batch_size = original_shape[0] 

        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].reshape(batch_size, 1, 1, 1)

        return sqrt_alphas_cumprod * original + sqrt_one_minus_alphas_cumprod * noise

    def sample_prev_timestep(self, xt, noise_pred, t):

        x0 = ((xt - (self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alphas_cumprod.to(xt.device)[t]))
        x0 = torch.clamp(x0, 0, 1)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alphas_cumprod.to(xt.device)[t - 1]) / (1.0 - self.alphas_cumprod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            return mean + sigma * z, x0

def get_time_embedding(time_steps, temb_dim):

    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_heads=4, num_layers=1, down_sample=True):
        super().__init__()
        self.down_sample = down_sample
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.down_sample = down_sample
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.residual_first_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, stride=1)
            )
            for i in range(num_layers)
        ])

        self.t_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels) 
            )
            for _ in range(num_layers)
        ])        

        self.residual_last_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1) 
            )
            for _ in range(num_layers)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels)
            for _ in range(num_layers)
        ])

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        self.down_block = nn.Conv2d(out_channels, out_channels, kernel_size=4, padding=1, stride=2) if self.down_sample else nn.Identity()
            

    def forward(self, x, t_emb):
        out = x
        for i in range(self.num_layers):
            residual = out
            out = self.residual_last_conv[i](self.residual_first_conv[i](out) + self.t_layers[i](t_emb)[:, :, None, None])
            out = out + self.residual_input_conv[i](residual)

            b, c, h, w = out.shape 
            in_attn = out.view(b, c, h * w)
            in_attn = self.attention_norms[i](in_attn).transpose(1, 2)
            out = out + self.attention_layers[i](in_attn, in_attn, in_attn)[0].transpose(1, 2).reshape(b, c, h, w)

        out = self.down_block(out)
        return out 

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_heads=4, num_layers=1):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_heads = num_heads 
        self.num_layers = num_layers
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.residual_first_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, stride=1)
            )
            for i in range(num_layers + 1)
        ])

        self.t_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels) 
            )
            for _ in range(num_layers + 1)
        ])        

        self.residual_last_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1) 
            )
            for _ in range(num_layers + 1)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels)
            for _ in range(num_layers + 1)
        ])

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers + 1)
        ])

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )


    def forward(self, x, t_emb):
        out = x
        out = self.residual_last_conv[0](self.residual_first_conv[0](out) + self.t_layers[0](t_emb)[:, :, None, None])
        out = out + self.residual_input_conv[0](x)

        for i in range(self.num_layers):
            b, c, h, w = out.shape 
            in_attn = out.view(b, c, h * w)
            in_attn = self.attention_norms[i](in_attn).transpose(1, 2)
            out = out + self.attention_layers[i](in_attn, in_attn, in_attn)[0].transpose(1, 2).reshape(b, c, h, w)

            residual = out
            out = self.residual_last_conv[i + 1](self.residual_first_conv[i + 1](out) + self.t_layers[i + 1](t_emb)[:, :, None, None])
            out = out + self.residual_input_conv[i + 1](residual)

        return out 
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_heads=4, num_layers=1, up_sample=True):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_heads = num_heads 
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.residual_first_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1, stride=1)
            )
            for i in range(num_layers)
        ])

        self.t_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels) 
            )
            for _ in range(num_layers)
        ])        

        self.residual_last_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1) 
            )
            for _ in range(num_layers)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(8, out_channels)
            for _ in range(num_layers)
        ])

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        self.up_block = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1) if self.up_sample else nn.Identity()

    def forward(self, x, down_out, t_emb):
        out = self.up_block(x)
        out = torch.cat([out, down_out], dim=1) 
        for i in range(self.num_layers):
            residual = out
            out = self.residual_last_conv[i](self.residual_first_conv[i](out) + self.t_layers[i](t_emb)[:, :, None, None])
            out = out+ self.residual_input_conv[i](residual)

            b, c, h, w = out.shape 
            in_attn = out.view(b, c, h * w)
            in_attn = self.attention_norms[i](in_attn).transpose(1, 2)
            out = out + self.attention_layers[i](in_attn, in_attn, in_attn)[0].transpose(1, 2).reshape(b, c, h, w)

        return out 


class UNet(nn.Module):
    def __init__(self, t_emb_dim=128):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.down_blocks = nn.ModuleList([
            DownBlock(32, 64, t_emb_dim, 2, 4),
            DownBlock(64, 128, t_emb_dim, 2, 4),
            DownBlock(128, 256, t_emb_dim, 2, 4, False),
        ])
        self.mid_blocks = nn.ModuleList([
            MidBlock(256, 256, t_emb_dim, 2, 4),
            MidBlock(256, 128, t_emb_dim, 2, 4),
        ])
        self.up_blocks = nn.ModuleList([
            UpBlock(256, 64, t_emb_dim, 2, 4, False),
            UpBlock(128, 32, t_emb_dim, 2, 4),
            UpBlock(64, 16, t_emb_dim, 2, 4),
        ])
        self.conv_in = nn.Conv2d(3, 32, 3, 1, 1)
        
        self.t_proj_layer = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.norm_out = nn.GroupNorm(8, 16)
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(16, 2, 3, 1, 1)
        
    def forward(self, x, t):
        
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj_layer(t_emb)
        out = self.conv_in(x)
        
        down_outs = []
        
        for down_block in self.down_blocks:
            down_outs.append(out)
            out = down_block(out, t_emb)
                    
        for mid_block in self.mid_blocks:
            out = mid_block(out, t_emb)
            
        for up_block in self.up_blocks:
            down_out = down_outs.pop()
            out = up_block(out, down_out, t_emb)

        return self.conv_out(self.silu(self.norm_out(out)))
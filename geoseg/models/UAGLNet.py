import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import copy
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from huggingface_hub import PyTorchModelHubMixin


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None, 
                       attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups=self.dim//ca_num_heads

        if ca_attention == 1:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim//self.ca_num_heads, dim//self.ca_num_heads, kernel_size=(3+i*2), padding=(1+i), stride=1, groups=dim//self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
            self.proj0 = nn.Conv2d(dim, dim*expand_ratio, kernel_size=1, padding=0, stride=1, groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim*expand_ratio)
            self.proj1 = nn.Conv2d(dim*expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        else:
            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C//self.ca_num_heads).permute(3, 0, 4, 1, 2)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i= s[i]
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out,s_i],2)
            s_out = s_out.reshape(B, C, H, W)
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
            self.modulator = s_out
            s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v

        else:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                    use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                    drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1,expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention, 
            expand_ratio=expand_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        img_size = to_2tuple(img_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Head(nn.Module):
    def __init__(self, head_conv, dim):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, dim, head_conv, 2, padding=3 if head_conv==7 else 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True)]
        stem.append(nn.Conv2d(dim, dim, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class SMT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=512, in_chans=3, num_classes=2, embed_dims=[64, 128, 256, 512],
                 ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[4, 4, 4, 2], 
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 18, 2], ca_attentions=[1, 1, 1, 0], num_stages=4, head_conv=3, expand_ratio=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Head(in_chans, embed_dims[i])#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], ca_num_heads=ca_num_heads[i], sa_num_heads=sa_num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                ca_attention=0 if i==2 and j%2!=0 else ca_attentions[i], expand_ratio=expand_ratio)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward(self, x):
        fms = []
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            fms.append(x)

        return fms

class LRDU(nn.Module):
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class UpConv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UpConv,self).__init__()
        self.up = nn.Sequential(
            LRDU(ch_in,2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(ConvBlock,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.GELU()
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class DWconvBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(DWconvBlock,self).__init__()
        # self.pre = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # x = self.pre(x)
        x = x + self.conv(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, embed_dim=[64, 128, 256, 512], out_dim=[64, 128, 256, 512]):
        super(FeatureFusionModule,self).__init__()
        self.dwconv1 = DWconvBlock(embed_dim[0], out_dim[0])
        self.dwconv2 = DWconvBlock(embed_dim[1], out_dim[1])
        self.dwconv3 = DWconvBlock(embed_dim[2], out_dim[2])
        self.dwconv4 = DWconvBlock(embed_dim[3], out_dim[3])

        self.up_conv4 = UpConv(ch_in=out_dim[3], ch_out=out_dim[2])
        self.up_conv3 = UpConv(ch_in=out_dim[2], ch_out=out_dim[1])
        self.up_convx = UpConv(ch_in=out_dim[2], ch_out=out_dim[1])
        self.up_conv2 = UpConv(ch_in=out_dim[1], ch_out=out_dim[0])

        self.conv3 = ConvBlock(ch_in=out_dim[2]*2, ch_out=out_dim[2])
        self.convx = ConvBlock(ch_in=out_dim[1], ch_out=out_dim[1])
        self.conv2 = ConvBlock(ch_in=out_dim[1]*2, ch_out=out_dim[1])
        self.conv1 = ConvBlock(ch_in=out_dim[0]*2, ch_out=out_dim[0])
                
    def forward(self, fms):
        f1, f2, f3, f4 = fms
        f1 = self.dwconv1(f1)
        f2 = self.dwconv2(f2)
        f3 = self.dwconv3(f3)
        f4 = self.dwconv4(f4)

        f_high = torch.cat([f3, self.up_conv4(f4)], dim = 1)
        f_high = self.conv3(f_high)
        f_high = self.up_convx(f_high)
        f_high = self.convx(f_high)

        f_low = torch.cat([f2, self.up_conv3(f3)], dim = 1)
        f_low = self.conv2(f_low)
        f_low = torch.cat([f1, self.up_conv2(f_low)], dim = 1)
        f_low = self.conv1(f_low)

        return f_high, f_low

class UncertaintyModule(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(UncertaintyModule,self).__init__()
        self.input_proj = nn.Sequential(nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_dim), nn.GELU())
        self.mean_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=False)
        self.std_conv  = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=False)

    def reparameterize(self, mu, logvar, k=1):
        std = logvar.mul(0.5).exp_()
        B, _, H, W = std.shape
        eps = torch.randn((B, k, H, W), device=std.device) 
        sample_z = eps.mul(std).add_(mu)
        return sample_z

    def forward(self, x):
        x = self.input_proj(x)
        mean = self.mean_conv(x)
        std = self.std_conv(x)

        prob_x = self.reparameterize(mean, std, 1)
        prob_x = F.interpolate(prob_x, size=(512, 512), mode='bilinear', align_corners=True)
        prob_x = torch.sigmoid(prob_x)

        #uncertainty
        prob_out2 = self.reparameterize(mean, std, 50)
        prob_out2 = torch.sigmoid(prob_out2)   
        uncertainty = prob_out2.var(dim=1, keepdim=True)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())

        return uncertainty, prob_x

class UncertaintyAggregatedDecoder(nn.Module):
    def __init__(self, out_dim=[64, 128, 256, 512]):
        super(UncertaintyAggregatedDecoder,self).__init__()
        self.upconv = nn.Sequential(UpConv(ch_in=out_dim[1], ch_out=out_dim[0]),
                                    ConvBlock(out_dim[0], out_dim[0]))
        self.unc_low = UncertaintyModule(dim = out_dim[0], hidden_dim = out_dim[0])
        self.unc_high = UncertaintyModule(dim = out_dim[0], hidden_dim = out_dim[0])
        self.conv_head = nn.Sequential(
            ConvBlock(out_dim[0], out_dim[0]),
            ConvBlock(out_dim[0], out_dim[0]),
            LRDU(out_dim[0],4),
            nn.Conv2d(out_dim[0], out_dim[0], kernel_size=3, stride=1, padding=1,groups=out_dim[0]),
            nn.Conv2d(out_dim[0], out_dim[0], kernel_size=3, stride=1, padding=1,groups=out_dim[0]),
            nn.Conv2d(out_dim[0], out_dim[0], kernel_size=3, stride=1, padding=1,groups=out_dim[0]),
            nn.Conv2d(out_dim[0], 2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, f_high, f_low):
        f_high = self.upconv(f_high)

        uncertainty_high, prob_high = self.unc_high(f_high)
        uncertainty_low, prob_low = self.unc_low(f_low)
        fusion_f = (1 - uncertainty_low) * f_low + (1 - uncertainty_high) * f_high
        output = self.conv_head(fusion_f)

        if self.training:
            return output, prob_high, prob_low
        else:
            return output

def create_encoder(drop_path_rate = 0., pretrained_backbone = None):
    if pretrained_backbone is None:
        encoder = SMT(
            embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[4, 4, 4, 2], 
            qkv_bias=True, depths=[2, 2, 8, 1], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, drop_path_rate=drop_path_rate)
    else:
        encoder = SMT.from_pretrained(pretrained_backbone, drop_path_rate=drop_path_rate)
    return encoder

class UAGLNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, drop_path_rate=0., pretrained_backbone=None):
        super(UAGLNet,self).__init__()
        self.CoE = create_encoder(drop_path_rate=drop_path_rate, pretrained_backbone=pretrained_backbone)
        self.GLF = FeatureFusionModule(embed_dim=[64, 128, 256, 512], out_dim=[64, 128, 256, 512])
        self.UAD = UncertaintyAggregatedDecoder(out_dim=[64, 128, 256, 512])

    def forward(self, x):
        fms = self.CoE(x)
        f_high, f_low = self.GLF(fms)
        return self.UAD(f_high, f_low)

if __name__ == '__main__':
    model = UAGLNet()
    model.eval()
    input = torch.rand(1, 3, 512, 512)
    output= model(input)

    print(output.shape)

    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, input)

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("FLOPs: %.4f G" % (flops.total()/1e9))
        print("Params: %.4f M" % (total_paramters / 1e6)) 
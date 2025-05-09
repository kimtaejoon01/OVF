import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from safetensors import safe_open
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
import open_clip

MODEL_TAG="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class OVFNet(nn.Module):
    def __init__(self, use_clinical, lora_rank=16):
        super().__init__()

        vit_model = open_clip.create_model(MODEL_TAG).visual.trunk
        self.image_encoder = _LoRA_ViT_timm(vit_model=vit_model, rank=lora_rank)
        hidden_channel = 768

        self.metadata_encoder = None
        if use_clinical:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(),
            )
            hidden_channel += 64
 
        self.final_layer = nn.Linear(hidden_channel, 1)

    def forward(self, x, m=None):
        features = self.image_encoder(x)
        if self.metadata_encoder is not None and m is not None:
            features = torch.cat([features, self.metadata_encoder(m)], dim=1)
        logit = self.final_layer(features)
        return logit
    

class _LoRA_qkv_base(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        raise NotImplementedError
     
class _LoRA_qkv_timm(_LoRA_qkv_base):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_k: nn.Module,
        linear_b_k: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__(
            qkv,
            linear_a_q,
            linear_b_q,
            linear_a_k,
            linear_b_k,
            linear_a_v,
            linear_b_v,
        )

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_k = self.linear_b_k(self.linear_a_k(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, self.dim:-self.dim] += new_k
        qkv[:, :, -self.dim:] += new_v
        return qkv
 
class _LoRA_ViT_base(nn.Module):
    def __init__(self, vit_model, rank: int, lora_layer=None):
        super(_LoRA_ViT_base, self).__init__()

        assert rank > 0
        
        # Create for storage, then we can init them or load weights
        # These are linear layers
        self.w_As = []
        self.w_Bs = []

        # Lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # write your code here #
        #                      #
        ########################

        self.reset_parameters()
        self.lora_vit = vit_model

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError 

class _LoRA_ViT_timm(_LoRA_ViT_base):
    def __init__(self, vit_model: timm_ViT, rank: int, lora_layer=None):
        super(_LoRA_ViT_timm, self).__init__(vit_model=vit_model, rank=rank)

        if lora_layer is not None:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        # Create for storage, then we can init them or load weights
        # These are linear layers
        self.w_As = []
        self.w_Bs = []

        # Lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
            w_a_linear_k = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_k = nn.Linear(rank, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_v = nn.Linear(rank, self.dim, bias=False)
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            blk.attn.qkv = _LoRA_qkv_timm(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_k,
                w_b_linear_k,
                w_a_linear_v,
                w_b_linear_v,
            )
            # blk.attn.fused_attn = False
            # blk.attn.store_attention_map = store_attention_map.__get__(blk.attn, blk.attn.__class__)
            # blk.attn.forward = blk.attn.store_attention_map

        self.reset_parameters()
        self.lora_vit = vit_model

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit.forward_features(x)[:, 0, :]


def store_attention_map(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)

    # Store attention map before dropout
    self.attention_map = attn  

    attn = self.attn_drop(attn)
    x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x



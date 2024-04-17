import torch
import torch.nn as nn
import bitsandbytes as bnb

# Wrapping policy requires modules, base_layer has no grad params, lora_A, lora_B, dora_scale have grad params.
class DORALayer(nn.Module):
    "Same as LORA but also returnes weight norm. This will be wrapped as a single FSDP unit"
    def __init__(self, in_features, out_features, lora_rank, device, dtype, *args, **kwargs):
        super().__init__()
        # Init LoRA layers.
        std_dev = 1 / torch.sqrt(torch.tensor(lora_rank).float())
        lora_A_param = nn.Parameter(torch.randn(lora_rank, in_features).to(device=device, dtype=dtype)*std_dev)
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False, device=device, dtype=dtype)
        setattr(self.lora_A, "weight", lora_A_param)
        
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False, device=device, dtype=dtype)
        self.lora_B.weight.data.zero_()
    
    def forward(self, x, frozen_weight):
        output = self.lora_B(self.lora_A(x))
        # print("lora A shape:", self.lora_A.weight.shape)
        # print("lora B shape:", self.lora_B.weight.shape)
        # DoRA Section 4.3. Detach column norm to avoid backprop through it.
        column_norm = (frozen_weight + self.lora_B.weight @ self.lora_A.weight).norm(p=2, dim=1).detach()
        # print("column norm shape:", column_norm.shape, column_norm.shape)
        return output, column_norm
    
class MagnitudeLayer(nn.Module):
    "FSDP doesn't work with nn.ParameterDict hence this module: https://github.com/pytorch/pytorch/issues/79605"
    def __init__(self, vector_data, device, dtype):
        super().__init__()
        self.magnitude = nn.Parameter(vector_data.to(device=device, dtype=dtype))
        
    def forward(self, x):
        return x * self.magnitude.view(1,1,-1)
    
class HQQDORA(nn.Module):
    def __init__(self, base_layer, lora_rank, *args, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        dtype = getattr(base_layer, "compute_dtype", next(base_layer.parameters()).dtype)
        device = next(base_layer.parameters()).device
        
        # Init trainable magnitude parameter.
        self.magnitude_layer = MagnitudeLayer(self.base_layer.dora_scale.clone().to(dtype=dtype), device, dtype)
        self.base_layer.dora_scale = None
        torch.cuda.empty_cache()
        
        # Init DORA layers.
        self.dora_layer = DORALayer(base_layer.in_features, base_layer.out_features, lora_rank, device, dtype, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        # As per Tim Dettmers, for 4bit, we need to defensively clone here.
        # The reason is that in some cases, an error can occur that backprop
        # does not work on a manipulated view. This issue may be solved with
        # newer PyTorch versions but this would need extensive testing to be
        # sure.
        result = result.clone()

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = x.to(self.dora_layer.lora_A.weight.dtype)

        # m * (W + AB / ||W + AB||) @ X == m * ((W @ X + AB @ X) / ||W + AB||)
        output, column_norm = self.dora_layer(x, self.base_layer.dequantize_aten())
        if requires_conversion:
            output = output.to(expected_dtype)
        
        result += output        
        result = result / column_norm.view(1,1,-1) #unit vector result.
        result = self.magnitude_layer(result) #rescaled result.
        return result
    
class BNBDORA(nn.Module):
    def __init__(self, base_layer, lora_rank, *args, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        dtype = getattr(base_layer, "compute_dtype", next(base_layer.parameters()).dtype)
        device = next(base_layer.parameters()).device
        
        # Init trainable magnitude parameter.
        self.magnitude_layer = MagnitudeLayer(self.base_layer.dora_scale.clone().to(dtype=dtype), device, dtype)
        self.base_layer.dora_scale = None
        torch.cuda.empty_cache()
        
        # Init DORA layers.
        self.dora_layer = DORALayer(base_layer.in_features, base_layer.out_features, lora_rank, device, dtype, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        # As per Tim Dettmers, for 4bit, we need to defensively clone here.
        # The reason is that in some cases, an error can occur that backprop
        # does not work on a manipulated view. This issue may be solved with
        # newer PyTorch versions but this would need extensive testing to be
        # sure.
        result = result.clone()

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = x.to(self.dora_layer.lora_A.weight.dtype)

        # m * (W + AB / ||W + AB||) @ X == m * ((W @ X + AB @ X) / ||W + AB||)
        output, column_norm = self.dora_layer(x, bnb.functional.dequantize_4bit(self.base_layer.weight.data, 
                                                                                self.base_layer.weight.quant_state))
        if requires_conversion:
            output = output.to(expected_dtype)
        
        result += output        
        result = result / column_norm.view(1,1,-1) #unit vector result.
        result = self.magnitude_layer(result) #rescaled result.
        return result
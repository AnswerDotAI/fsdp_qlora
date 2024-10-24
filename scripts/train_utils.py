import functools

from torch import nn
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LLAMA_ATTENTION_CLASSES, LlamaMLP, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MISTRAL_ATTENTION_CLASSES, MistralMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, QWEN2_ATTENTION_CLASSES, Qwen2MLP
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, PHI3_ATTENTION_CLASSES, Phi3MLP



from .dora import DORALayer, MagnitudeLayer


# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)
def get_wrapping_policy(custom_policy:bool=False, vanilla_policy:bool=False):

    if custom_policy:
        def lambda_policy_fn(module):
            # LoRA and DoRA trainable layers.
            return ((isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module)) or 
                    (isinstance(module, (DORALayer, MagnitudeLayer))))
    else:
        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
    
    def self_attn_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, tuple((*LLAMA_ATTENTION_CLASSES.values(), 
                                         *MISTRAL_ATTENTION_CLASSES.values(),
                                         *QWEN2_ATTENTION_CLASSES.values(), 
                                         *PHI3_ATTENTION_CLASSES.values())))

    def mlp_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, (LlamaMLP, MistralMLP, Qwen2MLP, Phi3MLP))
    
    def layernorm_policy_fn(module):
         return isinstance(module, LlamaRMSNorm) and module.weight.requires_grad

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    self_attn_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    layernorm_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=layernorm_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer, Phi3DecoderLayer),
    )
    if vanilla_policy:
        return transformer_wrap_policy
    
    policies=[lambda_policy, transformer_wrap_policy]
    if not vanilla_policy:
        policies.extend([self_attn_policy, mlp_policy, layernorm_policy])
    return functools.partial(_or_policy, policies=policies)



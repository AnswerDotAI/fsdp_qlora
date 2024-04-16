import sys
sys.path.append('../scripts/')
import unittest, tempfile
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
import torch
import torch.nn as nn
from dora import HQQDORA, BNBDORA

import bitsandbytes
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit


# python -m unittest tests.test_quantize.TestQuantizer.test_quantizer
# hqq pinned: 72b2b641aadc44a7ded6b243915f90df3b3be385 (to_empty not compatible with FSDP after this commit)
class TestHQQDORA(unittest.TestCase):

    def setUp(self) -> None:
        # set seed
        torch.manual_seed(42)
        quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True,
                                  quant_scale=True, offload_meta=True, view_as_float=True)
        self.base_layer = HQQLinear(nn.Linear(128,256,bias=False), quant_config, compute_dtype=torch.float32)
        HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_hqq_dora(self):   
        """
        Test:  m * (W + AB / ||W + AB||) @ X == m * ((W @ X + AB @ X) / ||W + AB||)
        """
        frozen_weight = self.base_layer.dequantize_aten().clone().cuda()
        self.base_layer.dora_scale = frozen_weight.norm(p=2,dim=1)
        hqq_dora = HQQDORA(self.base_layer, 16)
        weight = (frozen_weight + hqq_dora.dora_layer.lora_B.weight @ hqq_dora.dora_layer.lora_A.weight)
        norm_adapted = weight / weight.norm(p=2, dim=1).view(-1,1)
        calc_weights = norm_adapted * hqq_dora.magnitude_layer.magnitude.view(-1,1)
        x = torch.randn(1, 16,128).cuda().to(torch.float32)
        closeness = torch.isclose(x @ calc_weights.t(), hqq_dora(x)).float().mean().item()
        self.assertTrue(closeness > 0.99)
        
class TestBNBDORA(unittest.TestCase):

    def setUp(self) -> None:
        # set seed
        torch.manual_seed(42)
        self.base_layer = Linear4bit(128, 32, bias=False, quant_type="nf4", 
                                     compute_dtype=torch.float32, quant_storage=torch.float32)
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_bnb_dora(self):   
        """
        Test:  m * (W + AB / ||W + AB||) @ X == m * ((W @ X + AB @ X) / ||W + AB||)
        """
        # quantize and dequantize to avoid numerical mismatch during test.
        self.base_layer.cuda()
        frozen_weight = bnb.functional.dequantize_4bit(self.base_layer.weight.data, 
                                                       self.base_layer.weight.quant_state).clone()
        self.base_layer.dora_scale = frozen_weight.norm(p=2,dim=1)
        bnb_dora = BNBDORA(self.base_layer, 16).cuda()
        
        weight = (frozen_weight + bnb_dora.dora_layer.lora_B.weight @ bnb_dora.dora_layer.lora_A.weight)
        norm_adapted = weight / weight.norm(p=2, dim=1).view(-1,1)
        calc_weights = norm_adapted * bnb_dora.magnitude_layer.magnitude.view(-1,1)
        x = torch.randn(1, 16,128).cuda().to(torch.float32)
        closeness = torch.isclose(x @ calc_weights.t(), bnb_dora(x)).float().mean().item() 
        self.assertTrue(closeness > 0.99)
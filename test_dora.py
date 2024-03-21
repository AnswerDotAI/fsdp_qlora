import unittest, tempfile
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
import torch
import torch.nn as nn
from dora import HQQDORA

# python -m unittest tests.test_quantize.TestQuantizer.test_quantizer
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
        Test  m * (W + AB / ||W + AB||) @ X == m * ((W @ X + AB @ X) / ||W + AB||)
        """
        frozen_weight = self.base_layer.dequantize_aten()
        self.base_layer.dora_scale = frozen_weight.norm(p=2,dim=1)
        hqq_dora = HQQDORA(self.base_layer, 16)
        weight = (frozen_weight + hqq_dora.dora_layer.lora_B.weight @ hqq_dora.dora_layer.lora_A.weight)
        norm_adapted = weight / weight.norm(p=2, dim=1).view(-1,1)
        calc_weights = norm_adapted * hqq_dora.magnitude_layer.magnitude.view(-1,1)
        x = torch.randn(1, 16,128).cuda().to(torch.float32)
        self.assertTrue(torch.isclose(x @ calc_weights.t(), hqq_dora(x)).float().mean().item() > 0.99)
import unittest, tempfile
import torch
import torch.nn as nn
import safetensors
from safetensors.torch import save_file
from pathlib import Path
from transformers.utils import hub, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from glob import glob 

# python -m unittest tests.test_quantize.TestQuantizer.test_quantizer
class TestBlockExpansion(unittest.TestCase):

    def setUp(self) -> None:
        # set seed        
        self.llama_pro_path = Path("/mnt/vol_b/models/meta-llama/Llama-2-7b-hf_blk_exp-32-35")
        self.filenames = glob(str(self.llama_pro_path/"*.safetensors"))
        num_original_layers, num_expanded_layers = self.llama_pro_path.name.split("blk_exp-")[1].split("-")
        self.num_original_layers, self.num_expanded_layers = int(num_original_layers), int(num_expanded_layers)
        self.split = int(self.num_original_layers / (self.num_expanded_layers - self.num_original_layers))

        
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_expanded_weights(self):   
        
        total_new_layers = self.num_expanded_layers - self.num_original_layers
        new_layer_ids = [self.split + (self.split + 1)*n for n in range(total_new_layers)]
        
        verify_weights = {}
        for filename in self.filenames:
            weights = safetensors.torch.load_file(str(filename))
            for k,v in iter(weights.items()):
                if any(((f"layers.{i}" in k) or (f"layers.{i-1}" in k) for i in new_layer_ids)):
                    verify_weights[k] = v
                    
        for k,v in verify_weights.items():
            if any(((f"layers.{i}" in k) for i in new_layer_ids)):
                if 'down_proj' in k or 'o_proj' in k:
                    assert torch.equal(v, torch.zeros_like(v))
                else:
                    lid = int(k.split("layers.")[1].split(".")[0])
                    assert torch.equal(verify_weights[k.replace(f"layers.{lid}", f"layers.{lid-1}")], v)
                
from typing import Sequence, Optional, List
from pathlib import Path
import json
import numpy as np
from ctypes import *

try:
    from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
except ImportError:
    LIB_LLAISYS = None
    DeviceType = type('DeviceType', (), {'CPU': 0, 'NVIDIA': 1})()
    DataType = type('DataType', (), {'F32': 0, 'F16': 1, 'BF16': 2, 'I64': 3})()


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),           
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", c_void_p),
        ("out_embed", c_void_p),
        ("out_norm_w", c_void_p),
        ("attn_norm_w", POINTER(c_void_p)),
        ("attn_q_w", POINTER(c_void_p)),
        ("attn_q_b", POINTER(c_void_p)),
        ("attn_k_w", POINTER(c_void_p)),
        ("attn_k_b", POINTER(c_void_p)),
        ("attn_v_w", POINTER(c_void_p)),
        ("attn_v_b", POINTER(c_void_p)),
        ("attn_o_w", POINTER(c_void_p)),
        ("mlp_norm_w", POINTER(c_void_p)),
        ("mlp_gate_w", POINTER(c_void_p)),
        ("mlp_up_w", POINTER(c_void_p)),
        ("mlp_down_w", POINTER(c_void_p)),
    ]


class Qwen2:

    def __init__(self, model_path: str, device_type: int = 0):
 
        if LIB_LLAISYS is None:
            raise RuntimeError("LIB_LLAISYS not available")
            
        self.model_path = Path(model_path)
        
        self._load_config()
        
        self._create_c_model(device_type)
        
        self._load_weights()
        
        print(f"Model loaded successfully from {model_path}")

    def _load_config(self):
        config_file = self.model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, "r") as f:
            config = json.load(f)
        
        self.vocab_size = config["vocab_size"]          
        self.hidden_size = config["hidden_size"]        
        self.num_layers = config["num_hidden_layers"]   
        self.num_heads = config["num_attention_heads"]  
        self.num_kv_heads = config["num_key_value_heads"]  
        self.head_dim = self.hidden_size // self.num_heads  
        self.intermediate_size = config["intermediate_size"]  
        self.max_seq_len = config.get("max_position_embeddings", 131072)  
        self.rms_norm_eps = config["rms_norm_eps"]      
        self.rope_theta = config.get("rope_theta", 10000.0)  
        self.eos_token_id = config["eos_token_id"]      
        
        self.dtype = DataType.BF16
        
        print(f"Config: layers={self.num_layers}, hidden={self.hidden_size}, "
              f"heads={self.num_heads}/{self.num_kv_heads}, dim={self.head_dim}")

    def _create_c_model(self, device_type: int):

        meta = LlaisysQwen2Meta()
        meta.dtype = self.dtype
        meta.nlayer = self.num_layers
        meta.hs = self.hidden_size
        meta.nh = self.num_heads
        meta.nkvh = self.num_kv_heads
        meta.dh = self.head_dim
        meta.di = self.intermediate_size
        meta.maxseq = self.max_seq_len
        meta.voc = self.vocab_size
        meta.epsilon = self.rms_norm_eps
        meta.theta = self.rope_theta
        meta.end_token = self.eos_token_id
        
        device_ids = (c_int * 1)(0)
        
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device_type,
            device_ids,
            1  # ndevice
        )
        
        if not self.model:
            raise RuntimeError("Failed to create C model")
        
        weights_c_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        if not weights_c_ptr:
            raise RuntimeError("Failed to get model weights")
        self.weights_ptr = cast(weights_c_ptr, POINTER(LlaisysQwen2Weights))

    def _load_weights(self):
        try:
            import safetensors
            import torch
        except ImportError:
            raise ImportError("safetensors and torch are required for loading weights")
        
        weight_files = sorted(self.model_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")
        
        for file in weight_files:
            print(f"Loading {file.name}...")
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    self._copy_weight(name, tensor)

    def _copy_weight(self, name: str, tensor):
        import torch
        
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        c_tensor = self._get_c_tensor(name)
        if c_tensor is None:
            return
        

        if tensor.dtype == torch.bfloat16:
            np_array = tensor.view(torch.uint16).numpy()
        elif tensor.dtype == torch.float16:
            np_array = tensor.view(torch.uint16).numpy()
        else:
            np_array = tensor.numpy()
        
        LIB_LLAISYS.tensorLoad(c_tensor, np_array.ctypes.data_as(c_void_p))

    def _get_c_tensor(self, name: str) -> Optional[c_void_p]:
    
        weights = self.weights_ptr[0]
        
        if name == "model.embed_tokens.weight":
            return weights.in_embed
        elif name == "lm_head.weight":
            return weights.out_embed
        elif name == "model.norm.weight":
            return weights.out_norm_w
        
        if not name.startswith("model.layers."):
            return None
        
        parts = name.split(".")
        if len(parts) < 4:
            return None
        
        layer_idx = int(parts[2])
        if layer_idx >= self.num_layers:
            return None
        
        suffix = ".".join(parts[3:])
        
        if suffix == "input_layernorm.weight":
            return weights.attn_norm_w[layer_idx]
        elif suffix == "self_attn.q_proj.weight":
            return weights.attn_q_w[layer_idx]
        elif suffix == "self_attn.q_proj.bias":
            return weights.attn_q_b[layer_idx]
        elif suffix == "self_attn.k_proj.weight":
            return weights.attn_k_w[layer_idx]
        elif suffix == "self_attn.k_proj.bias":
            return weights.attn_k_b[layer_idx]
        elif suffix == "self_attn.v_proj.weight":
            return weights.attn_v_w[layer_idx]
        elif suffix == "self_attn.v_proj.bias":
            return weights.attn_v_b[layer_idx]
        elif suffix == "self_attn.o_proj.weight":
            return weights.attn_o_w[layer_idx]
        elif suffix == "post_attention_layernorm.weight":
            return weights.mlp_norm_w[layer_idx]
        elif suffix == "mlp.gate_proj.weight":
            return weights.mlp_gate_w[layer_idx]
        elif suffix == "mlp.up_proj.weight":
            return weights.mlp_up_w[layer_idx]
        elif suffix == "mlp.down_proj.weight":
            return weights.mlp_down_w[layer_idx]
        
        return None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.8,
    ) -> List[int]:
        
        
        probs_buffer = (c_float * self.vocab_size)()
        output_tokens = list(inputs)  
        # prefill 阶段
        prompt_len = len(inputs)
        input_arr = (c_int64 * prompt_len)(*inputs)
        
        # result是状态值，0表示成功，真正返回的概率在probs_buffer中
        result = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            input_arr,
            prompt_len,
            probs_buffer,
            c_float(temperature)
        )
        
        if result != 0:
            raise RuntimeError(f"Prefill failed with code {result}")
        
        next_token = self._sample(probs_buffer, top_k, top_p)
        output_tokens.append(next_token)
        
        # decode 阶段
        generated_count = 1  
        while True:
            if max_new_tokens is not None and generated_count >= max_new_tokens:
                break
            if next_token == self.eos_token_id:
                break
            
            single_token = (c_int64 * 1)(next_token)
            
            result = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model,
                single_token,
                1,
                probs_buffer,
                c_float(temperature)
            )
            
            if result != 0:
                raise RuntimeError(f"Decode failed with code {result}")
            
            next_token = self._sample(probs_buffer, top_k, top_p)
            output_tokens.append(next_token)
            generated_count += 1
        
        return output_tokens

    def _sample(self, probs_buffer, top_k: int, top_p: float) -> int:
        probs = np.array(probs_buffer[:self.vocab_size], dtype=np.float32)
        
        if top_k > 1:
            kth_idx = np.argpartition(probs, -top_k)[-top_k]
            kth_val = probs[kth_idx]
            probs[probs < kth_val] = 0
            if probs.sum() > 0:
                probs = probs / probs.sum()
        
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            
            cutoff_idx = np.searchsorted(cumsum, top_p, side='right') + 1
            cutoff_idx = min(cutoff_idx, len(sorted_probs))
            cutoff_val = sorted_probs[cutoff_idx - 1]
            
            probs[probs < cutoff_val] = 0
            if probs.sum() > 0:
                probs = probs / probs.sum()
        
        if top_k == 1:
            return int(np.argmax(probs))
        else:
            return int(np.random.choice(self.vocab_size, p=probs))

    def __del__(self):
        """析构时释放模型: llaisysQwen2ModelDestroy"""
        if hasattr(self, 'model') and self.model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            self.model = None


# # 使用示例
# if __name__ == "__main__":
#     # 加载模型
#     model = Qwen2("/path/to/DeepSeek-R1-Distill-Qwen-1.5B", device_type=0)
    
#     # 生成（需要配合 tokenizer）
#     prompt_tokens = [1, 2, 3]  # 你的 tokenizer 编码结果
#     output = model.generate(prompt_tokens, max_new_tokens=100)
#     print(f"Generated: {output}")

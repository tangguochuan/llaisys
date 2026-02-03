from typing import Sequence, Optional, List
from pathlib import Path
import json
import numpy as np
from ctypes import *
import sys

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


class LlaisysQwen2Model(Structure):
    pass


# 设置函数参数和返回类型
def _setup_model_functions():
    if LIB_LLAISYS is None:
        return
    
    # llaisysQwen2ModelCreate
    LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),
        c_int,  # device
        POINTER(c_int),  # device_ids
        c_int   # ndevice
    ]
    LIB_LLAISYS.llaisysQwen2ModelCreate.restype = POINTER(LlaisysQwen2Model)
    
    # llaisysQwen2ModelDestroy
    LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [POINTER(LlaisysQwen2Model)]
    LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None
    
    # llaisysQwen2ModelWeights
    LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [POINTER(LlaisysQwen2Model)]
    LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)
    
    # llaisysQwen2ModelInfer
    LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
        POINTER(LlaisysQwen2Model),
        POINTER(c_int64),  # token_ids
        c_size_t,          # ntoken
        POINTER(c_float),  # probs_out
        c_float            # temperature
    ]
    LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int


_setup_model_functions()


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
        
        print(f"[DEBUG] Creating C model with dtype={meta.dtype}, nlayer={meta.nlayer}")
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device_type,
            device_ids,
            1  # ndevice
        )
        
        if not self.model:
            raise RuntimeError("Failed to create C model")
        
        print(f"[DEBUG] C model created: {self.model}")
        
        weights_c_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        if not weights_c_ptr:
            raise RuntimeError("Failed to get model weights")
        self.weights_ptr = cast(weights_c_ptr, POINTER(LlaisysQwen2Weights))
        print(f"[DEBUG] Weights pointer: {weights_c_ptr}")

    def _load_weights(self):
        try:
            import safetensors
            import torch
        except ImportError:
            raise ImportError("safetensors and torch are required for loading weights")
        
        weight_files = sorted(self.model_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found in {self.model_path}")
        
        print(f"[DEBUG] Found {len(weight_files)} weight files")
        loaded_count = 0
        skipped_count = 0
        
        for file in weight_files:
            print(f"[DEBUG] Loading {file.name}...")
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    result = self._copy_weight(name, tensor)
                    if result:
                        loaded_count += 1
                    else:
                        skipped_count += 1
        
        print(f"[DEBUG] Weight loading complete: {loaded_count} loaded, {skipped_count} skipped")

    def _copy_weight(self, name: str, tensor) -> bool:
        import torch
        
        print(f"[DEBUG] _copy_weight: name={name}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}, contiguous={tensor.is_contiguous()}")
        
        if not tensor.is_contiguous():
            print(f"[DEBUG] Making tensor contiguous...")
            tensor = tensor.contiguous()
        
        c_tensor = self._get_c_tensor(name)
        if c_tensor is None:
            print(f"[DEBUG] No C tensor mapping for {name}, skipping")
            return False
        
        print(f"[DEBUG] C tensor pointer: {c_tensor}")
        
        try:
            if tensor.dtype == torch.bfloat16:
                print(f"[DEBUG] Converting BF16 to uint16 view...")
                # Ensure contiguous before view
                tensor_u16 = tensor.view(torch.uint16)
                if not tensor_u16.is_contiguous():
                    print(f"[DEBUG] Making uint16 view contiguous...")
                    tensor_u16 = tensor_u16.contiguous()
                np_array = tensor_u16.numpy()
            elif tensor.dtype == torch.float16:
                print(f"[DEBUG] Converting F16 to uint16 view...")
                tensor_u16 = tensor.view(torch.uint16)
                if not tensor_u16.is_contiguous():
                    print(f"[DEBUG] Making uint16 view contiguous...")
                    tensor_u16 = tensor_u16.contiguous()
                np_array = tensor_u16.numpy()
            else:
                print(f"[DEBUG] Converting to numpy...")
                np_array = tensor.numpy()
            
            print(f"[DEBUG] np_array shape={np_array.shape}, dtype={np_array.dtype}, flags={np_array.flags}")
            
            data_ptr = np_array.ctypes.data_as(c_void_p)
            print(f"[DEBUG] Calling tensorLoad with data_ptr={data_ptr}")
            
            LIB_LLAISYS.tensorLoad(c_tensor, data_ptr)
            print(f"[DEBUG] tensorLoad successful")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to copy weight {name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False

    def _get_c_tensor(self, name: str) -> Optional[c_void_p]:
    
        weights = self.weights_ptr[0]
        
        if name == "model.embed_tokens.weight":
            ptr = weights.in_embed
        elif name == "lm_head.weight":
            ptr = weights.out_embed
        elif name == "model.norm.weight":
            ptr = weights.out_norm_w
        elif not name.startswith("model.layers."):
            return None
        else:
            parts = name.split(".")
            if len(parts) < 4:
                return None
            
            layer_idx = int(parts[2])
            if layer_idx >= self.num_layers:
                return None
            
            suffix = ".".join(parts[3:])
            
            if suffix == "input_layernorm.weight":
                ptr = weights.attn_norm_w[layer_idx]
            elif suffix == "self_attn.q_proj.weight":
                ptr = weights.attn_q_w[layer_idx]
            elif suffix == "self_attn.q_proj.bias":
                ptr = weights.attn_q_b[layer_idx]
            elif suffix == "self_attn.k_proj.weight":
                ptr = weights.attn_k_w[layer_idx]
            elif suffix == "self_attn.k_proj.bias":
                ptr = weights.attn_k_b[layer_idx]
            elif suffix == "self_attn.v_proj.weight":
                ptr = weights.attn_v_w[layer_idx]
            elif suffix == "self_attn.v_proj.bias":
                ptr = weights.attn_v_b[layer_idx]
            elif suffix == "self_attn.o_proj.weight":
                ptr = weights.attn_o_w[layer_idx]
            elif suffix == "post_attention_layernorm.weight":
                ptr = weights.mlp_norm_w[layer_idx]
            elif suffix == "mlp.gate_proj.weight":
                ptr = weights.mlp_gate_w[layer_idx]
            elif suffix == "mlp.up_proj.weight":
                ptr = weights.mlp_up_w[layer_idx]
            elif suffix == "mlp.down_proj.weight":
                ptr = weights.mlp_down_w[layer_idx]
            else:
                return None
        
        print(f"[DEBUG] _get_c_tensor: {name} -> ptr={ptr}")
        return ptr

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: Optional[int] = None,
        top_k: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.8,
    ) -> List[int]:
        
        print(f"[DEBUG] generate called: inputs={inputs[:20]}..., max_new_tokens={max_new_tokens}, top_k={top_k}, top_p={top_p}, temperature={temperature}")
        
        probs_buffer = (c_float * self.vocab_size)()
        output_tokens = list(inputs)  
        # prefill 阶段
        prompt_len = len(inputs)
        input_arr = (c_int64 * prompt_len)(*inputs)
        
        print(f"[DEBUG] Prefill: prompt_len={prompt_len}")
        
        # result是状态值，0表示成功，真正返回的概率在probs_buffer中
        print(f"[DEBUG] Calling llaisysQwen2ModelInfer for prefill...")
        result = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            input_arr,
            prompt_len,
            probs_buffer,
            c_float(temperature)
        )
        
        print(f"[DEBUG] Prefill result: {result}")
        
        if result != 0:
            raise RuntimeError(f"Prefill failed with code {result}")
        
        next_token = self._sample(probs_buffer, top_k, top_p)
        print(f"[DEBUG] Sampled next_token: {next_token}")
        output_tokens.append(next_token)
        
        # decode 阶段
        generated_count = 1  
        while True:
            if max_new_tokens is not None and generated_count >= max_new_tokens:
                print(f"[DEBUG] Reached max_new_tokens")
                break
            if next_token == self.eos_token_id:
                print(f"[DEBUG] Reached EOS token")
                break
            
            single_token = (c_int64 * 1)(next_token)
            
            print(f"[DEBUG] Decode step {generated_count}: token={next_token}")
            result = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model,
                single_token,
                1,
                probs_buffer,
                c_float(temperature)
            )
            
            print(f"[DEBUG] Decode result: {result}")
            
            if result != 0:
                raise RuntimeError(f"Decode failed with code {result}")
            
            next_token = self._sample(probs_buffer, top_k, top_p)
            print(f"[DEBUG] Sampled next_token: {next_token}")
            output_tokens.append(next_token)
            generated_count += 1
        
        print(f"[DEBUG] Generation complete: {len(output_tokens)} tokens")
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
            print(f"[DEBUG] Destroying model...")
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

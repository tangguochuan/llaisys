#pragma once
#include "llaisys/models/qwen2.h" 
#include "llaisys/tensor.h"
#include "../../tensor/tensor.hpp"
#include <vector>

namespace llaisys {
namespace models {

class Qwen2Model {
public:
    struct Config {
        size_t nlayer;
        size_t hidden_size;
        size_t num_query_heads;
        size_t num_kv_heads;
        size_t head_dim;
        size_t intermediate_size;
        size_t max_seq_Len;
        size_t vocab_size;
        float rms_norm_eps;
        float rope_theta;
        int64_t eos_token_id;
        llaisysDataType_t dtype;
        llaisysDeviceType_t device;
    };
    
    struct KVCache {
        // 每层一个 tensor: [max_seq_len, num_kv_heads, head_dim]
        std::vector<llaisys::tensor_t> k_cache;
        std::vector<llaisys::tensor_t> v_cache;
    };
    
    Qwen2Model(const LlaisysQwen2Meta* meta, 
               llaisysDeviceType_t device_type, 
               const std::vector<int>& device_ids);
    
    ~Qwen2Model();
    
    // 返回词表的 logits, 方便后面进行采样
    // logits shape: [1, vocab_size]
    llaisys::tensor_t infer(const std::vector<int64_t>& token_ids);
    
    // 重置 KV Cache（开始新对话时使用）
    void reset_cache();
    
    // 获取当前 cache 长度（用于调试）
    int get_cache_len() const { return cache_len_; }
    
    // 获取 weights（用于 Python 层加载权重）
    LlaisysQwen2Weights* weights() { return weights_; }

private:
    Config config_;
    LlaisysQwen2Weights* weights_;
    KVCache kv_cache_;
    int cache_len_;  // 当前已缓存的 token 数量
};

} // namespace models
} // namespace llaisys

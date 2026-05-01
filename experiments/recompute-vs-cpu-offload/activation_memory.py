MB = 1024 * 1024


def act_mem_total_per_layer(s, b, h, ffn_hidden_size, tp=1):
    return s * b * h * (18 + 4 * (ffn_hidden_size / h)) / tp


def act_mem_attn_norm(s, b, h, tp=1, bytes_per_elem=2):
    return (s/tp) * b * h * bytes_per_elem


def act_mem_qkv_linear(s, b, h, tp=1, bytes_per_elem=2):
    # X replicated, not shared
    return (s/tp) * b * h * bytes_per_elem


def act_mem_core_attn(s, b, h, num_heads, kv_heads, tp=1, bytes_per_elem=2):
    head_dim = h / num_heads
    q_out_size = s * b * (num_heads / tp) * head_dim  # q and out have same shape
    kv_size    = s * b * (kv_heads  / tp) * head_dim
    return (2 * q_out_size + 2 * kv_size) * bytes_per_elem


def act_mem_attn_proj(s, b, h, num_heads, tp=1, bytes_per_elem=2):
    head_dim = h / num_heads
    q_size = s * b * (num_heads / tp) * head_dim  # attn proj input = core attn output = q shape
    return q_size * bytes_per_elem


def act_mem_mlp_norm(s, b, h, tp=1, bytes_per_elem=2):
    return (s / tp) * b * h * bytes_per_elem

def act_fc1_output(s, b, ffn_hidden_size, tp=1, bytes_per_elem=2):
    return (s / tp) * b * ffn_hidden_size * bytes_per_elem 

if __name__ == "__main__":
    s          = 4096
    b          = 1
    h          = 6144
    ffn        = 20480
    num_heads  = 48
    kv_heads   = 8
    tp         = 4
    num_layers = 40

    total  = act_mem_total_per_layer(s, b, h, ffn, tp=tp)
    a_norm = act_mem_attn_norm      (s, b, h, tp=tp)
    qkv    = act_mem_qkv_linear     (s, b, h, tp=tp)
    c_attn = act_mem_core_attn      (s, b, h, num_heads, kv_heads, tp=tp)
    a_proj = act_mem_attn_proj      (s, b, h, num_heads, tp=tp)
    m_norm = act_mem_mlp_norm       (s, b, h, tp=tp)
    
    # these are not offloadable in fine grained offloading.
    a_fc1   = act_fc1_output         (s, b, ffn, tp=tp)
    gelu_output = a_fc1  # same shape as fc1 output 
    a_fc2 = (s / tp) * b * h * 2  
    bda_after_attn = (s / tp) * b * h 
    bda_after_mlp = (s / tp) * b * h
    
    offloadable   = a_norm + qkv + c_attn + a_proj + m_norm
    stuck = total - offloadable
    total_accounted = a_norm + qkv + c_attn + a_proj + m_norm + a_fc1 + gelu_output + a_fc2 + bda_after_attn + bda_after_mlp
    print(f"18b model  |  s={s}, b={b}, h={h}, ffn={ffn}, heads={num_heads}/{kv_heads} GQA, TP={tp}")
    print(f"{'─'*65}")
    print(f"  Total activation memory / layer:      {total      / MB:7.1f} MB  = sbh(18 + 4·ffn/h)/tp bytes")
    print(f"{'─'*65}")
    print(f"  [attn_norm]  input_layernorm input:  {a_norm     / MB:7.1f} MB  ({100*a_norm /total:5.1f}%)  = 2sbh/tp bytes")
    print(f"  [qkv_linear] QKV proj input:         {qkv        / MB:7.1f} MB  ({100*qkv   /total:5.1f}%)  = 2sbh/tp bytes")
    print(f"  [core_attn]  Q + K + V + out:        {c_attn     / MB:7.1f} MB  ({100*c_attn/total:5.1f}%)  = 2sb(2h + 2·h·kv/heads)/tp bytes")
    print(f"  [attn_proj]  attn output proj input: {a_proj     / MB:7.1f} MB  ({100*a_proj/total:5.1f}%)  = 2sb·heads·d_head/tp bytes")
    print(f"  [mlp_norm]   pre_mlp_layernorm input:{m_norm     / MB:7.1f} MB  ({100*m_norm/total:5.1f}%)  = 2sbh/tp bytes")
    print("----")
    print(f"  [mlp_fc1]    mlp fc1 output:         {a_fc1      / MB:7.1f} MB  ({100*a_fc1 /total:5.1f}%)  = 2sb·ffn/tp bytes")
    print(f"  [gelu_output] gelu output :          {gelu_output / MB:7.1f} MB  ({100*gelu_output /total:5.1f}%)  = 2sb·ffn/tp bytes")
    print(f"  [mlp_fc2]    mlp fc2 input :         {a_fc2 / MB:7.1f} MB  ({100*a_fc2 /total:5.1f}%)  = 2sb·h/tp bytes") 
    print(f"  [bda_attn]   bda after attn:         {bda_after_attn / MB:7.1f} MB  ({100*bda_after_attn /total:5.1f}%)  = 2sb·h/tp bytes")
    print(f"  [bda_mlp]    bda after mlp:          {bda_after_mlp / MB:7.1f} MB  ({100*bda_after_mlp /total:5.1f}%)  = 2sb·h/tp bytes")
    print(f"{'─'*65}")
    print(f"  Total accounted:                     {total_accounted / MB:7.1f} MB  ({100*total_accounted/total:5.1f}% of total)")
    print(f"  Offloadable subtotal (dense):        {offloadable   / MB:7.1f} MB  ({100*offloadable/total:.1f}% of layer)")
    print(f"  Stuck on GPU:  {stuck / MB:7.1f} MB")
    print(f"{'─'*65}")
    print("- attn_norm fused with qkv_linear\n")
    
    print(f"Some modules are replicated, not sharded. The following is their size in CPU")
    print(f"activation memory (not divided by TP):")
    print(f"  attn_norm: {act_mem_attn_norm(s, b, h, tp=1) * (num_layers-1)/ MB:7.1f} MB")
    print(f"  qkv_linear: {act_mem_qkv_linear(s, b, h, tp=1)* (num_layers-1) / MB:7.1f} MB")
    
    print(f"  Total activation memory ({num_layers} layers): {total * num_layers / MB:7.1f} MB")

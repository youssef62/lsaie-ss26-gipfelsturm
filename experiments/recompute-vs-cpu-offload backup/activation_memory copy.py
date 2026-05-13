MB    = 1024 * 1024
GFLOP = 1e9


# FLOPs formulas (per GPU, forward pass only).
# Column-parallel linears do an all-gather first, so the full-s tensor is
# contracted; row-parallel linears reduce-scatter after, same total work.
# Both simplify to  2 * s * b * in_features * out_features / tp.
# LayerNorm runs on the local [s/tp, b, h] shard (~5 ops/element).

def flop_layernorm(s, b, h, tp=1):
    return 5 * (s / tp) * b * h

def flop_qkv_linear(s, b, h, num_heads, kv_heads, tp=1):
    head_dim     = h / num_heads
    out_features = (num_heads + 2 * kv_heads) * head_dim  # Q + K + V (GQA)
    return 2 * s * b * h * out_features / tp

def flop_core_attn(s, b, h, tp=1):
    # QK^T + scores@V; same total as MHA (GQA saves memory, not FLOPs)
    return 4 * s**2 * b * h / tp

def flop_attn_proj(s, b, h, tp=1):
    return 2 * s * b * h * h / tp

def flop_mlp_fc(s, b, h, ffn_hidden_size, tp=1):
    return 2 * s * b * h * ffn_hidden_size / tp

def flop_gelu(s, b, ffn_hidden_size, tp=1):
    return 8 * (s / tp) * b * ffn_hidden_size  # ~8 ops/element


def act_mem_total_per_layer(s, b, h, ffn_hidden_size, tp=1):
    # Megatron formula: 10*sbh (replicated, not TP-sharded) + 24*sbh/tp (TP-sharded)
    return s * b * h * (10 + 24 / tp)


def act_mem_attn_norm(s, b, h, tp=1, bytes_per_elem=2):
    # Residual stream is replicated across TP ranks (no sequence parallelism)
    return s * b * h * bytes_per_elem


def act_mem_qkv_linear(s, b, h, tp=1, bytes_per_elem=2):
    # Input is the replicated residual stream (same as attn_norm input if fused)
    return s * b * h * bytes_per_elem


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
    # Residual stream is replicated across TP ranks (no sequence parallelism)
    return s * b * h * bytes_per_elem

def act_fc1_output(s, b, ffn_hidden_size, tp=1, bytes_per_elem=2):
    return (s / tp) * b * ffn_hidden_size * bytes_per_elem 

if __name__ == "__main__":
    s          = 4096
    b          = 1
    h          = 7168
    ffn        = 20480
    num_heads  = 56
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
    # fc1 input: saved by ColumnParallelLinear save_for_backward(input, weight)
    # input to column-parallel linear is the replicated residual stream (not TP-sharded)
    a_fc1_input  = s * b * h * 2
    # fc1 output: saved by BiasSwiGLUFunction as the activation's input for backward
    a_fc1_output = act_fc1_output(s, b, ffn, tp=tp)
    # fc2 input (= activation output): saved by RowParallelLinear save_for_backward(input, weight)
    a_fc2_input  = s * b * (ffn / tp) * 2
    # dropout masks applied to replicated (post-all-reduce) tensors → not TP-sharded
    bda_after_attn = s * b * h
    bda_after_mlp  = s * b * h

    offloadable     = a_norm + qkv + c_attn + a_proj + m_norm
    stuck           = total - offloadable
    total_accounted = a_norm + qkv + c_attn + a_proj + m_norm + a_fc1_input + a_fc1_output + a_fc2_input + bda_after_attn + bda_after_mlp

    # FLOPs per module (forward pass, per GPU)
    f_a_norm = flop_layernorm  (s, b, h,              tp=tp)
    f_qkv    = flop_qkv_linear (s, b, h, num_heads, kv_heads, tp=tp)
    f_c_attn = flop_core_attn  (s, b, h,              tp=tp)
    f_a_proj = flop_attn_proj  (s, b, h,              tp=tp)
    f_m_norm = flop_layernorm  (s, b, h,              tp=tp)
    f_fc1    = flop_mlp_fc     (s, b, h, ffn,         tp=tp)
    f_gelu   = flop_gelu       (s, b,    ffn,         tp=tp)
    f_fc2    = flop_mlp_fc     (s, b, h, ffn,         tp=tp)
    f_bda    = 0.0  # elementwise, negligible

    W = 110
    def row(tag, desc, mem, pct, flops, mem_formula, flop_formula):
        flop_str  = f"{flops/GFLOP:6.1f} GFLOP" if flops > 0 else "      —     "
        ratio_str = f"{flops/mem:5.1f} FLOP/B" if (flops > 0 and mem > 0) else "    —      "
        print(f"  {tag:<14} {desc:<26} {mem/MB:7.1f} MB ({pct:5.1f}%)  = {mem_formula:<22}  "
              f"{flop_str}  = {flop_formula:<26}  {ratio_str}")

    print(f"22b model  |  s={s}, b={b}, h={h}, ffn={ffn}, heads={num_heads}/{kv_heads} GQA, TP={tp}")
    print(f"{'─'*W}")
    print(f"  Total activation memory / layer:   {total/MB:7.1f} MB  = sbh(10 + 24/tp) bytes  [Megatron formula]")
    print(f"{'─'*W}")
    row("[attn_norm]",  "input_layernorm input:", a_norm, 100*a_norm /total, f_a_norm,
        "2sbh B  (replicated)",   "5·(s/tp)·b·h")
    row("[qkv_linear]", "QKV proj input:",        qkv,    100*qkv   /total, f_qkv,
        "2sbh B  (replicated)",   "2sb·h·(h+2·kv·d)/tp")
    row("[core_attn]",  "Q + K + V + out:",       c_attn, 100*c_attn/total, f_c_attn,
        "2sb(2h+2h·kv/heads)/tp B","4s²·b·h/tp")
    row("[attn_proj]",  "attn proj input:",        a_proj, 100*a_proj/total, f_a_proj,
        "2sb·heads·d_head/tp B",  "2s·b·h²/tp")
    row("[mlp_norm]",   "pre-MLP layernorm input:",m_norm, 100*m_norm/total, f_m_norm,
        "2sbh B  (replicated)",   "5·(s/tp)·b·h")
    print(f"  {'─'*30}  (not offloadable below)  {'─'*30}")
    row("[mlp_fc1_in]", "mlp fc1 input:",           a_fc1_input,  100*a_fc1_input /total, f_fc1,
        "2sbh B  (replicated)",   "2s·b·h·ffn/tp")
    row("[mlp_act_in]", "activation input:",        a_fc1_output, 100*a_fc1_output/total, f_gelu,
        "2sb·ffn/tp B",           "8·(s/tp)·b·ffn")
    row("[mlp_fc2_in]", "mlp fc2 input:",           a_fc2_input,  100*a_fc2_input /total, f_fc2,
        "2sb·ffn/tp B",           "2s·b·h·ffn/tp")
    row("[bda_attn]",   "bda after attn:",     bda_after_attn, 100*bda_after_attn/total, f_bda,
        "sb·h B   (replicated)",  "—")
    row("[bda_mlp]",    "bda after mlp:",      bda_after_mlp,  100*bda_after_mlp /total, f_bda,
        "sb·h B   (replicated)",  "—")
    print(f"{'─'*W}")
    print(f"  Total accounted:                  {total_accounted / MB:7.1f} MB  ({100*total_accounted/total:5.1f}% of total)")
    print(f"  Offloadable subtotal (dense):     {offloadable   / MB:7.1f} MB  ({100*offloadable/total:.1f}% of layer)")
    print(f"  Stuck on GPU:                     {stuck / MB:7.1f} MB")
    print(f"  FLOPs: forward-pass per GPU (2·s·b·in·out/tp for linears, 4·s²·b·h/tp for attn, 5·(s/tp)·b·h for LN)")
    print(f"{'─'*65}")
    print("- attn_norm fused with qkv_linear\n")
    
    print(f"Some modules are replicated, not sharded. The following is their size in CPU")
    print(f"activation memory (not divided by TP):")
    print(f"  attn_norm: {act_mem_attn_norm(s, b, h, tp=1) * (num_layers-1)/ MB:7.1f} MB")
    print(f"  qkv_linear: {act_mem_qkv_linear(s, b, h, tp=1)* (num_layers-1) / MB:7.1f} MB")
    
    print(f"  Total activation memory ({num_layers} layers): {total * num_layers / MB:7.1f} MB")

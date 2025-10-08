import pytest
import torch

import flashinfer
from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    nvfp4_quantize,
    mxfp4_quantize,
)
import math
import torch.nn.functional as F
from flashinfer.utils import get_compute_capability, LibraryError

# Layer: Context Attention
# Operation: CUTLASS FMHA --> not available, use FA2 backend, only support bf16 input, bf16, fp8 output
# FMHA Paged KV Cache (non varlen)
@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("page_size", [1, 5])
@pytest.mark.parametrize("seq_len", [1, 7, 127, 257])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_batch_paged_prefill_packed_input(
    batch_size,
    page_size,
    seq_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")

    nnz = batch_size * seq_len
    num_pages_per_req = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_req
    last_page_len = (seq_len - 1) % page_size + 1
    k_cache = torch.randn(
        size=(num_pages, page_size, num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_cache = (k_cache, v_cache)
    workspace_buffer = torch.empty(
        (256 * 1024 * 1024,), dtype=torch.uint8, device="cuda:0"
    )
    qo_indptr = torch.tensor(
        [i * seq_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    paged_kv_indptr = torch.tensor(
        [i * num_pages_per_req for i in range(batch_size + 1)],
        dtype=torch.int32,
        device="cuda:0",
    )
    paged_kv_indices = torch.tensor(
        list(range(num_pages)), dtype=torch.int32, device="cuda:0"
    )
    paged_kv_last_page_len = torch.tensor(
        [last_page_len for _ in range(batch_size)], dtype=torch.int32, device="cuda:0"
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, backend="fa2")
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=causal,
        q_data_type=torch.bfloat16,
    )

    qkv_packed = torch.randn(
        size=(nnz, (num_qo_heads + 2 * num_kv_heads) * head_dim),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    qkv_split_idx = (
        num_qo_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    q, _, _ = qkv_packed.split(qkv_split_idx, dim=-1)
    # pretend that we have already appended k/v to paged_kv table
    q = q.view(-1, num_qo_heads, head_dim)
    o_packed = wrapper.run(q, paged_kv_cache)
    o_contiguous = wrapper.run(q.contiguous(), paged_kv_cache)
    torch.testing.assert_close(o_packed, o_contiguous, rtol=1e-3, atol=2e-3)

# Layer: Context Attention
# Operation: Trtllm FMHA --> Not supported, use FA2 backend, only support bf16 input
# FMHA Ragged KV Cache (varlen)
@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("seq_len", [1, 7, 127, 257])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_batch_ragged_prefill_packed_input(
    batch_size, seq_len, num_kv_heads, num_qo_heads, head_dim, causal
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")
    nnz = batch_size * seq_len
    qkv_packed = torch.randn(
        nnz,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(nnz, num_kv_heads, head_dim)
    v = qkv_packed[:, (num_qo_heads + num_kv_heads) * head_dim :].reshape(
        nnz, num_kv_heads, head_dim
    )
    qo_indptr = torch.tensor(
        [i * seq_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    kv_indptr = qo_indptr

    workspace_buffer = torch.empty(
        (256 * 1024 * 1024,), dtype=torch.uint8, device="cuda:0"
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, backend="fa2")
    wrapper.plan(
        qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim, causal=causal, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16
    )
    o_packed = wrapper.run(q, k, v)
    o_contiguous = wrapper.run(q.contiguous(), k.contiguous(), v.contiguous())

    torch.testing.assert_close(o_packed, o_contiguous, rtol=1e-3, atol=1e-3)

# Deprecated
# Single non batched prefill, bf16 input only, bf16, fp8 output
@pytest.mark.parametrize("seq_len", [1, 7, 127, 999, 3579])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_single_prefill_packed_input(
    seq_len, num_kv_heads, num_qo_heads, head_dim, causal
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")
    qkv_packed = torch.randn(
        seq_len,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(
        seq_len, num_qo_heads, head_dim
    )
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(seq_len, num_kv_heads, head_dim)
    v = qkv_packed[:, (num_qo_heads + num_kv_heads) * head_dim :].reshape(
        seq_len, num_kv_heads, head_dim
    )

    o_packed = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal, o_dtype=torch.float8_e5m2)
    o_contiguous = flashinfer.single_prefill_with_kv_cache(
        q.contiguous(), k.contiguous(), v.contiguous(), causal=causal, o_dtype=torch.float8_e5m2
    )

    # torch.testing.assert_close(o_packed, o_contiguous, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(o_packed, o_contiguous, rtol=0.0, atol=0.0)

from tests.test_helpers.test_helpers import clear_cuda_cache
from tests.attention.test_deepseek_mla import generate_kv_from_cache, attention_ref
from flashinfer.utils import (
    is_sm90a_supported,
)
# Layer: Decode MLA -> Not supported for trtllm backend, use fa2
@pytest.mark.parametrize("batch_size", [1, 3, 5, 7, 157])
@pytest.mark.parametrize("kv_len", [1, 17, 33, 96, 97, 114, 514, 1024])
@pytest.mark.parametrize("qo_len", [1])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("backend", ["fa2"])
@pytest.mark.parametrize("use_cuda_graph", [False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_mla_page_attention(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    page_size,
    backend,
    use_cuda_graph,
    dtype,
):
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if causal and qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported for causal attention")
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        batch_size * qo_len, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        batch_size * qo_len, num_heads, head_dim_kpe, dtype=dtype, device=device
    )
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer,
        backend=backend,
        use_cuda_graph=True,
        qo_indptr=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        kv_indptr=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        kv_indices=torch.empty(1048576, dtype=torch.int32, device=device),
        kv_len_arr=torch.empty(batch_size, dtype=torch.int32, device=device),
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    )
    kv_indices = torch.arange(
        0, batch_size * pages_num, device=device, dtype=torch.int32
    )
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    if use_cuda_graph:
        kv_indptr_warmup = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        kv_indices_warmup = torch.arange(
            0, batch_size, device=device, dtype=torch.int32
        )
        kv_lens_warmup = torch.full((batch_size,), 0, dtype=torch.int32, device=device)
        wrapper.plan(
            q_indptr,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_lens_warmup,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            causal,
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    if use_cuda_graph:
        o.fill_(0)
        lse.fill_(0)
        g.replay()
    else:
        o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)

    q = torch.cat([q_nope, q_pe], dim=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(0, 1)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    if kv_len != 0:
        torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)

    # test with pre-allocated output
    o_buffer = torch.empty_like(o)
    lse_buffer = torch.empty_like(lse)
    wrapper.run(q_nope, q_pe, ckv, kpe, out=o_buffer, lse=lse_buffer)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(o, o_buffer, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)


# Layer: Dense GEMM (Attention)
# Layer: Dense GEMM (MoE)
# NVFP4, CUTLASS
@pytest.mark.parametrize("m", [1, 48, 128, 256, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cutlass"])
@pytest.mark.parametrize("use_128x4_sf_layout", [True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4"])
def test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    use_nvfp4 = fp4_type == "nvfp4"

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if backend == "trtllm":
        if res_dtype == torch.float16:
            pytest.skip("Skipping test for trtllm fp4 with float16")
        if compute_capability[0] in [11, 12]:
            pytest.skip("trtllm gemm does not support SM110/SM120/SM121 GPUs.")
    if not use_128x4_sf_layout and backend != "trtllm":
        pytest.skip("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
    if auto_tuning and backend == "cudnn":
        pytest.skip("Skipping test for cudnn fp4 with auto_tuning=True")
    if not use_nvfp4 and backend != "cudnn":
        pytest.skip("mx_fp4 is only supported for cudnn backend")

    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    # for trtllm, we need to shuffle mat2 because we swap A, B.
    do_shuffle_b = backend == "trtllm"

    block_size = 16 if use_nvfp4 else 32
    has_alpha = fp4_type == "mxfp4_alpha" or fp4_type == "nvfp4"

    if use_nvfp4:
        input_fp4, input_inv_s = nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=do_shuffle_b,
        )
    else:
        input_fp4, input_inv_s = mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = mxfp4_quantize(mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2) if has_alpha else None

    reference = torch.mm(input, mat2.T)

    res = torch.empty([m, n], device="cuda", dtype=res_dtype)

    try:
        with autotune(auto_tuning):
            mm_fp4(
                input_fp4,
                mat2_fp4.T,
                input_inv_s,
                mat2_inv_s.T,
                alpha,
                res_dtype,
                res,
                block_size=block_size,
                use_8x4_sf_layout=not use_128x4_sf_layout,
                backend=backend,
                use_nvfp4=use_nvfp4,
            )

        cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
        assert cos_sim > 0.97
    except LibraryError:
        # TODO: Remove this check once cuDNN backend version is updated to 9.14.0
        if (
            backend == "cudnn"
            and not use_nvfp4
            and (compute_capability[0] == 12 and compute_capability[1] == 0)
        ):
            pytest.xfail(
                "cudnn FP4 GEMM with mxfp4 quantization is not supported on SM120 with cuDNN backend version < 9.14.0."
            )
        else:
            pytest.fail("Unexpected LibraryError")






# Layer: Fused MoE (Trtllm)
# Not implemented for SM120 -> use cutlass? (are they functionally different?)

# Layer: Fused MoE (Cutlass)
import flashinfer.fused_moe as fused_moe
from flashinfer import (
    fp4_quantize,
    mxfp4_quantize,
)

from tests.moe.test_trtllm_cutlass_fused_moe import compute_routing, dequantize_nvfp4_to_dtype, torch_moe_nvfp4
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn

@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [2])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("intermediate_size", [128])
@pytest.mark.parametrize(
    "otype, wtype",
    [(torch.float16, torch.float8_e4m3fn), (torch.bfloat16, torch.float8_e4m3fn)],
)
@pytest.mark.parametrize("quantized_input", [False, True])
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10, 11, 12],
    reason="NVFP4 is only supported on SM100, SM110 and SM120",
)
def test_moe_nvfp4(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    wtype,
    quantized_input,
):
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    quant_blocksize = 16
    round_up = lambda x, y: (x + y - 1) // y * y
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype) / 10
    w1_cutlass = torch.cat((w1[:, n:, :], w1[:, :n, :]), dim=1).contiguous()

    sf_w1_2n = round_up(2 * n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_blockscale_cutlass = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )

    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_q = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w1_q_cutlass = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1).max().to(torch.float32)
        w2_amax = torch.abs(w2).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])

        w1_q_cutlass[expert], w1_blockscale_cutlass[expert] = fp4_quantize(
            w1_cutlass[expert], w1_gs[expert]
        )

        w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])

    x = torch.randn(m, k, dtype=otype).cuda()
    a1_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(x).max().to(
        torch.float32
    ).cuda()
    a1_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    # quant_scales format
    # auto const fc1_act_global = quant_scales.value()[0];
    # auto const fc1_weight_block = quant_scales.value()[1];
    # auto const fc1_global = quant_scales.value()[2];
    # auto const fc2_act_global = quant_scales.value()[3];
    # auto const fc2_weight_block = quant_scales.value()[4];
    # auto const fc2_global = quant_scales.value()[5];
    flash_output = torch.zeros_like(x)

    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]
    hidden_states = x
    input_sf = None
    if quantized_input:
        hidden_states, input_sf = fp4_quantize(x, a1_gs)
    _ = fused_moe.cutlass_fused_moe(
        hidden_states,
        selected_experts.to(torch.int),
        routing_weights,
        w1_q.contiguous().view(torch.long),
        w2_q.contiguous().view(torch.long),
        otype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        output=flash_output,
    )

    # Ref check
    a_fp4, a_scale_interleaved = fp4_quantize(x, a1_gs)
    _, m_k = a_fp4.shape
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a1_gs,
        dtype=otype,
        device=x.device,
        block_size=quant_blocksize,
    )

    w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=otype)
    w2_d = torch.empty((e, k, n), device="cuda", dtype=otype)

    for idx in range(0, e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx],
            w1_blockscale[idx],
            w1_gs[idx],
            dtype=w1.dtype,
            device=w1.device,
            block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx],
            w2_blockscale[idx],
            w2_gs[idx],
            dtype=w2.dtype,
            device=w2.device,
            block_size=quant_blocksize,
        )

    w1_q_cutlass = torch.cat((w1_q[:, n:, :], w1_q[:, :n, :]), dim=1).contiguous()
    w1_blockscale_cutlass = torch.cat(
        (w1_blockscale[:, n:, :], w1_blockscale[:, :n, :]), dim=1
    ).contiguous()
    ref_output = torch_moe_nvfp4(
        a_in_dtype, w1_d, w2_d, top_k, routing_weights, selected_experts
    )
    torch.testing.assert_close(ref_output, flash_output, rtol=2e-1, atol=2e-1)

# Layer: Masked Grouped GEMM (CuteDSL)
# Not implemented for NVFP4

# Layer: Communication (fused allreduce + rmsnorm) --> Not tested yet




if __name__ == "__main__":
    # test_batch_paged_prefill_packed_input(37, 5, 127, 4, 4, 64, True)
    # test_batch_ragged_prefill_packed_input(37, 127, 4, 4, 64, True)
    pass
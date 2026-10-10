"""Direct pytest tests for _decode_bnsd_score_attn_chunk_kernel."""
import os
from types import SimpleNamespace

import pytest

os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_DEBUG"] = "0"
torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from flash_block_score_decode import _decode_bnsd_score_attn_chunk_kernel

DEFAULTS = dict(batch=1, q_heads=4, kv_heads=1, dim=128, seq_len=32768,
                block_size=128, topk=16, chunks=8, score_type="max", dtype="bf16")
CASES = [
    pytest.param({}, id="table-max-bf16-32k"),
    pytest.param({"batch": 2, "q_heads": 8, "kv_heads": 2, "seq_len": 8193,
                  "score_type": "lse", "dtype": "fp16"}, id="table-lse-fp16-gqa-tail"),
]


@pytest.fixture(scope="module", autouse=True)
def npu_runtime():
    if not torch.npu.is_available():
        pytest.skip("An available Ascend NPU is required")
    torch.npu.set_device(0)


def check_outputs(q, k, v, output, lse, scores, indices, cfg):
    """Check each chunk's attention/LSE and global TopK against CPU FP32."""
    q, k, v = q.float().cpu(), k.float().cpu(), v.float().cpu()
    output, lse = output.float().cpu(), lse.cpu()
    scores = scores.cpu().permute(1, 2, 0, 3).reshape(cfg.q_heads, cfg.batch, -1)
    indices = indices.cpu().permute(1, 2, 0, 3).reshape(cfg.q_heads, cfg.batch, -1)
    top_scores, positions = scores.topk(cfg.topk, dim=-1)
    selected = indices.gather(-1, positions).long()
    assert bool(torch.isfinite(top_scores).all())
    num_blocks = (cfg.seq_len + cfg.block_size - 1) // cfg.block_size
    blocks_per_chunk = (num_blocks + cfg.chunks - 1) // cfg.chunks
    assert bool(((selected >= 0) & (selected < num_blocks)).all())
    group_size = cfg.q_heads // cfg.kv_heads
    for b in range(cfg.batch):
        keys = k[b * num_blocks:(b + 1) * num_blocks].reshape(-1, cfg.kv_heads, cfg.dim)
        values = v[b * num_blocks:(b + 1) * num_blocks].reshape(-1, cfg.kv_heads, cfg.dim)
        for h in range(cfg.q_heads):
            logits = (keys[:, h // group_size] @ q[b, h]) * cfg.dim ** -0.5
            logits[cfg.seq_len:] = -float("inf")
            blocks = logits.reshape(num_blocks, cfg.block_size)
            block_scores = blocks.max(-1).values if cfg.score_type == "max" else blocks.logsumexp(-1)
            assert selected[h, b].unique().numel() == cfg.topk
            threshold = block_scores.topk(cfg.topk).values[-1]
            assert bool((block_scores[selected[h, b]] >= threshold - 0.02).all())
            for c in range(cfg.chunks):
                start = c * blocks_per_chunk * cfg.block_size
                end = min(start + blocks_per_chunk * cfg.block_size, cfg.seq_len)
                if start >= end:
                    assert bool((output[c, b, h] == 0).all())
                    assert bool(torch.isneginf(lse[c, b, h]))
                    continue
                chunk_logits = logits[start:end]
                ref_output = chunk_logits.softmax(-1) @ values[start:end, h // group_size]
                # The kernel writes log2(sum(exp(logits))), not natural-log LSE.
                ref_lse = chunk_logits.logsumexp(-1) / torch.log(torch.tensor(2.0))
                torch.testing.assert_close(output[c, b, h], ref_output, atol=0.01, rtol=0.05)
                torch.testing.assert_close(lse[c, b, h], ref_lse, atol=0.02, rtol=0.005)
    print("Chunk attention, log2 LSE and TopK vs CPU FP32: PASS")


@pytest.mark.parametrize("overrides", CASES)
@torch.no_grad()
def test_decode_score_attn_chunk(overrides):
    cfg = SimpleNamespace(**{**DEFAULTS, **overrides})
    assert cfg.q_heads % cfg.kv_heads == 0
    for size in (cfg.block_size, cfg.topk, cfg.chunks):
        assert size > 0 and size & (size - 1) == 0
    assert cfg.seq_len > cfg.block_size * cfg.topk
    torch.manual_seed(123)
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16
    num_blocks = (cfg.seq_len + cfg.block_size - 1) // cfg.block_size
    num_pages = cfg.batch * num_blocks

    # Inputs: Q [B,QH,D], K/V [pages,block,KVH,D], page table [B,blocks].
    q = torch.randn(cfg.batch, cfg.q_heads, cfg.dim, device="npu:0", dtype=dtype)
    k = torch.randn(num_pages, cfg.block_size, cfg.kv_heads, cfg.dim,
                    device="npu:0", dtype=dtype)
    v = torch.randn_like(k)
    table = torch.arange(num_pages, device="npu:0", dtype=torch.int32).reshape(cfg.batch, num_blocks)
    lengths = torch.full((cfg.batch,), cfg.seq_len, device="npu:0", dtype=torch.int32)

    # Outputs: chunk attention [C,B,QH,D], log2 LSE [C,B,QH], TopK [C,QH,B,K].
    output = torch.empty((cfg.chunks, cfg.batch, cfg.q_heads, cfg.dim), device="npu:0", dtype=dtype)
    lse = torch.empty((cfg.chunks, cfg.batch, cfg.q_heads), device="npu:0", dtype=torch.float32)
    shape = (cfg.chunks, cfg.q_heads, cfg.batch, cfg.topk)
    scores = torch.full(shape, -float("inf"), device="npu:0", dtype=torch.float32)
    indices = torch.full(shape, -1, device="npu:0", dtype=torch.int32)
    print(f"\nconfig={vars(cfg)}")

    _decode_bnsd_score_attn_chunk_kernel[(cfg.batch * cfg.chunks, cfg.kv_heads)](
        q_ptr=q, sink_ptr=None,
        k_cache_ptr=k, v_cache_ptr=v, block_table_ptr=table,
        o_ptr=output, lse_ptr=lse,
        candidate_scores_ptr=scores, candidate_indices_ptr=indices,
        seq_lens=lengths,
        batch_size=cfg.batch, gqa_group_size=cfg.q_heads // cfg.kv_heads,
        head_dim=cfg.dim, block_size=cfg.block_size, sm_scale=cfg.dim ** -0.5,
        init_blocks=0, local_blocks=0,
        stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
        stride_sink_h=0, stride_sink_d=0,
        stride_k_block=k.stride(0), stride_k_offset=k.stride(1),
        stride_k_h=k.stride(2), stride_k_d=k.stride(3),
        stride_v_block=v.stride(0), stride_v_offset=v.stride(1),
        stride_v_h=v.stride(2), stride_v_d=v.stride(3),
        stride_bt_b=table.stride(0), stride_bt_n=table.stride(1),
        stride_o_c=output.stride(0), stride_o_b=output.stride(1),
        stride_o_h=output.stride(2), stride_o_d=output.stride(3),
        stride_l_c=lse.stride(0), stride_l_b=lse.stride(1), stride_l_h=lse.stride(2),
        stride_cs_c=scores.stride(0), stride_cs_h=scores.stride(1),
        stride_cs_b=scores.stride(2), stride_cs_t=scores.stride(3),
        stride_ci_c=indices.stride(0), stride_ci_h=indices.stride(1),
        stride_ci_b=indices.stride(2), stride_ci_t=indices.stride(3),
        BLOCK_SIZE_N=cfg.block_size, NUM_KV_CHUNKS=cfg.chunks,
        SCORE_TYPE=cfg.score_type, topk=cfg.topk,
        num_warps=4, num_stages=2,
    )
    torch.npu.synchronize()

    # Reference checking uses CPU only; no wrapper or merge kernel is called.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        check_outputs(q, k, v, output, lse, scores, indices, cfg)
    finally:
        torch.set_num_threads(previous_threads)

"""Direct pytest tests for _decode_bnsd_score_topk_chunk_kernel."""
import os
from types import SimpleNamespace

import pytest

os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_DEBUG"] = "0"
torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")

from flash_block_score_decode import _decode_bnsd_score_topk_chunk_kernel

DEFAULTS = dict(batch=1, q_heads=4, kv_heads=1, dim=128, seq_len=32768,
                block_size=128, topk=16, chunks=16, score_type="max",
                page_source="table", dtype="bf16")
CASES = [
    pytest.param({}, id="table-max-bf16-32k-fill"),
    pytest.param({"seq_len": 65536}, id="table-max-bf16-64k-local-topk"),
    pytest.param({"page_source": "direct", "seq_len": 65536,
                  "q_heads": 1, "score_type": "lse"},
                 id="direct-lse-bf16-64k-local-topk"),
]


@pytest.fixture(scope="module", autouse=True)
def npu_runtime():
    if not torch.npu.is_available():
        pytest.skip("An available Ascend NPU is required")
    torch.npu.set_device(0)


def check_topk(q, k, indices, cfg):
    """Check selected blocks against a CPU FP32 reference."""
    q, k = q.float().cpu(), k.float().cpu()
    indices = indices.long().cpu()
    num_blocks = (cfg.seq_len + cfg.block_size - 1) // cfg.block_size
    group_size = cfg.q_heads // cfg.kv_heads
    for b in range(cfg.batch):
        keys = k[b * num_blocks:(b + 1) * num_blocks].reshape(
            num_blocks * cfg.block_size, cfg.kv_heads, cfg.dim
        )
        for h in range(cfg.q_heads):
            logits = (keys[:, h // group_size] @ q[b, h]) * cfg.dim ** -0.5
            logits[cfg.seq_len:] = -float("inf")
            blocks = logits.reshape(num_blocks, cfg.block_size)
            scores = blocks.max(-1).values if cfg.score_type == "max" else blocks.logsumexp(-1)
            selected = indices[h, b]
            assert selected.unique().numel() == cfg.topk, "Duplicate blocks"
            threshold = scores.topk(cfg.topk).values[-1]
            assert bool((scores[selected] >= threshold - 0.02).all()), "TopK differs from FP32"
    print("TopK vs CPU FP32: PASS")


@pytest.mark.parametrize("overrides", CASES)
@torch.no_grad()
def test_decode_score_topk_chunk(overrides):
    cfg = SimpleNamespace(**{**DEFAULTS, **overrides})
    assert cfg.q_heads % cfg.kv_heads == 0
    assert cfg.block_size & (cfg.block_size - 1) == 0
    assert cfg.topk & (cfg.topk - 1) == 0
    assert cfg.seq_len > cfg.block_size * cfg.topk
    torch.manual_seed(123)
    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16
    num_blocks = (cfg.seq_len + cfg.block_size - 1) // cfg.block_size
    num_pages = cfg.batch * num_blocks
    direct = cfg.page_source == "direct"
    fill_only = (num_blocks + cfg.chunks - 1) // cfg.chunks <= cfg.topk

    # Inputs: Q [B,QH,D], K [pages,block,KVH,D], logical-to-physical page maps.
    q = torch.randn(cfg.batch, cfg.q_heads, cfg.dim, device="npu:0", dtype=dtype)
    k = torch.randn(num_pages, cfg.block_size, cfg.kv_heads, cfg.dim,
                    device="npu:0", dtype=dtype)
    lengths = torch.full((cfg.batch,), cfg.seq_len, device="npu:0", dtype=torch.int32)
    table = torch.arange(num_pages, device="npu:0", dtype=torch.int32).reshape(cfg.batch, num_blocks)
    request_ids = torch.arange(cfg.batch, device="npu:0", dtype=torch.int32)
    token_map = (
        torch.arange(num_pages * cfg.block_size, device="npu:0", dtype=torch.int32)
        .reshape(cfg.batch, num_blocks * cfg.block_size)
        if direct else table  # typed placeholder; unused in block-table mode
    )

    # Outputs: each chunk's TopK candidates. No wrapper or merge kernel is used.
    shape = (cfg.chunks, cfg.q_heads, cfg.batch, cfg.topk)
    scores = torch.full(shape, -float("inf"), device="npu:0", dtype=torch.float32)
    indices = torch.full(shape, -1, device="npu:0", dtype=torch.int32)
    print(f"\nconfig={vars(cfg)}, FILL_ONLY={fill_only}")

    _decode_bnsd_score_topk_chunk_kernel[(cfg.batch * cfg.chunks, cfg.kv_heads)](
        q_ptr=q,
        k_cache_ptr=k,
        block_table_ptr=table,
        req_to_token_ptr=token_map,
        req_pool_indices_ptr=request_ids,
        candidate_scores_ptr=scores,
        candidate_indices_ptr=indices,
        seq_lens=lengths,
        stride_sl_b=1, stride_sl_h=0,
        batch_size=cfg.batch, gqa_group_size=cfg.q_heads // cfg.kv_heads,
        head_dim=cfg.dim, block_size=cfg.block_size,
        sm_scale=cfg.dim ** -0.5, init_blocks=0, local_blocks=0,
        num_score_chunks=cfg.chunks,
        stride_q_b=q.stride(0), stride_q_h=q.stride(1), stride_q_d=q.stride(2),
        stride_k_block=k.stride(0), stride_k_offset=k.stride(1),
        stride_k_h=k.stride(2), stride_k_d=k.stride(3),
        stride_bt_b=table.stride(0), stride_bt_n=table.stride(1),
        stride_rtt_r=token_map.stride(0), stride_rtt_t=token_map.stride(1),
        max_req_to_token_cols=token_map.shape[1], num_pages=num_pages,
        stride_cs_c=scores.stride(0), stride_cs_h=scores.stride(1),
        stride_cs_b=scores.stride(2), stride_cs_t=scores.stride(3),
        stride_ci_c=indices.stride(0), stride_ci_h=indices.stride(1),
        stride_ci_b=indices.stride(2), stride_ci_t=indices.stride(3),
        BLOCK_SIZE_N=cfg.block_size, SCORE_TYPE=cfg.score_type, topk=cfg.topk,
        USE_DIRECT_PAGE_LOOKUP=direct, SANITIZE_PAGE_IDS=False,
        FILL_ONLY=fill_only, RUNTIME_FILL_ONLY=True,
        RUNTIME_ADAPTIVE_SCORE_CHUNKS=False,
        RUNTIME_SCORE_SHORT_MAX_BLOCKS=0, RUNTIME_SCORE_SHORT_CHUNKS=0,
        num_warps=4, num_stages=2,
    )
    torch.npu.synchronize()

    # Merge candidates on CPU only, for checking the selected global blocks.
    scores_cpu = scores.cpu().permute(1, 2, 0, 3).reshape(cfg.q_heads, cfg.batch, -1)
    indices_cpu = indices.cpu().permute(1, 2, 0, 3).reshape(cfg.q_heads, cfg.batch, -1)
    top_scores, positions = scores_cpu.topk(cfg.topk, dim=-1)
    selected = indices_cpu.gather(-1, positions)
    assert bool(torch.isfinite(top_scores).all())
    assert bool(((selected >= 0) & (selected < num_blocks)).all())
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        check_topk(q, k, selected, cfg)
    finally:
        torch.set_num_threads(previous_threads)

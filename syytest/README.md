# a5-ta：decode chunk kernels pytest 运行指导

服务器目录：`/home/s00943453/test/sgl-kernel-npu/syytest`。

Git 包含测试脚本、运行指导、run_env.sh 和 cann_tools 编译器兼容入口。环境安装脚本、安装包、虚拟环境、缓存、IR、日志和报告不提交。

每个 kernel 有独立的 pytest 文件，参数通过文件中的 `DEFAULTS` 和 `CASES` 定义，不再使用 argparse 或命令行 shape 参数。

## 运行

```bash
ssh a5-ta
cd /home/s00943453/test/sgl-kernel-npu/syytest
source ./run_env.sh
python -m pytest -v -s test_decode_score_topk_chunk.py test_decode_score_attn_chunk.py
```

单独运行某个 kernel：

```bash
python -m pytest -v -s test_decode_score_topk_chunk.py
python -m pytest -v -s test_decode_score_attn_chunk.py
```

选择单个已定义用例可使用 pytest 标准的 -k：

```bash
python -m pytest -v -s test_decode_score_topk_chunk.py -k direct-lse
```

默认使用物理 NPU 3，映射为逻辑 npu:0。换卡前检查其他作业，然后设置 SYY_NPU_DEVICE 并重新 source：

```bash
npu-smi info
export SYY_NPU_DEVICE=4
source ./run_env.sh
```

没有 torch/torch_npu 或可用 NPU 时模块或测试会 skip；skip 不代表设备验证通过。

## 参数化用例

| 文件 | 用例 ID | 覆盖范围 |
|---|---|---|
| test_decode_score_topk_chunk.py | table-max-bf16-32k-fill | 32K、block table、max、FILL_ONLY |
| 同上 | table-max-bf16-64k-local-topk | 64K、block table、max、正常局部 TopK |
| 同上 | direct-lse-bf16-64k-local-topk | 64K、direct-map、LSE、正常局部 TopK |
| test_decode_score_attn_chunk.py | table-max-bf16-32k | 32K、BF16、max、attention |
| 同上 | table-lse-fp16-gqa-tail | B=2、QH=8、KVH=2、L=8193、FP16、LSE、尾 block |

在各文件的 CASES 列表中添加 pytest.param 即可增加用例，例如：

```python
pytest.param({"batch": 4, "seq_len": 65536}, id="table-max-bf16-b4-64k"),
```

DEFAULTS 定义共享参数：B=1、QH=4、KVH=1、D=128、L=32768、block_size=128、topk=16、BF16、max。score 的 chunks=16，attention 的 chunks=8。直接编辑 DEFAULTS 或 CASES 字典即可调整 shape。

要求 QH 能被 KVH 整除；block_size、topk 和 attention chunks 为 2 的幂；seq_len 大于 block_size*topk。direct-map 仅 score 文件支持。两个文件通过 from flash_block_score_decode import ... 普通导入目标 kernel。run_env.sh 将仓库的 python/sgl_kernel_npu/sgl_kernel_npu/indexer 加入 PYTHONPATH，无需模型权重或安装原生算子。

## 直接调用和参考检查

两个文件都按准备输入、直接调用目标 kernel、CPU 参考检查的顺序展开。pytest fixture 仅负责检查和选择 NPU；没有 wrapper、launch 捕获、重放或 merge kernel。

- score：输出每个 chunk 的 TopK 候选，在 CPU 合并后检查范围、唯一性和 FP32 block 分数。选中分数不得低于第 K 大分数减 0.02。
- attention：输出每个 chunk 的 attention、log2 LSE 和 TopK 候选。逐 chunk 对比 CPU FP32 attention（atol=0.01、rtol=0.05）和 log2 LSE（atol=0.02、rtol=0.005）；候选在 CPU 合并后检查 TopK。

脚本固定 TRITON_ALWAYS_COMPILE=1，每次运行重新编译，不执行重复性能计时。当前用例使用等长请求、顺序物理 pages、无 sink、init/local=0。

保存 pytest 报告：

```bash
python -m pytest -v -s test_decode_score_topk_chunk.py test_decode_score_attn_chunk.py \
  --junitxml=pytest_results.xml
```

下方性能数据来自简化前的历史测试，不是当前脚本输出。

## 版本和安装

验证日期：2026-10-10。设备：Ascend950PR。Python 3.11.15；torch 2.7.1+cpu；torch_npu 2.7.1.post8；CANN 9.1.0。

sgl-kernel-npu commit：`84ce0315e4247337eb83a754bf5efaaf3104ad09`。

TA 使用 WSL `/home/shaoyiyang/code/triton-ascend` 的 `main` 当前 HEAD `ebef084711f96f7c21c09cb77574dc18b45a029e` 已有构建产物。未重新构建；构建目录存在未提交源码修改，因此 wheel 的 commit 标签不能证明它来自干净 checkout。

上传的 wheel：

```text
/home/s00943453/bin/triton_ascend-3.6.0+gitebef0847-cp311-cp311-linux_x86_64.whl
SHA256: e6def8a4f5b78e9311046bb4b4e0b4d66d3ca01e5a4ddafcf34b9434b72e3fbb
```

## 运行依赖

使用 a5-ta 已安装的 Ascend 驱动、CANN 9.1 A5 工具链、torch/torch_npu 和 syytest/.venv 中的 TA 环境。本仓库不提供环境安装脚本或安装包。

cann_tools 中的 bishengir-compile、bishengir-opt 和 hivmc 均为提交的源码入口，不需要重新生成符号链接。新 checkout 需要保留可执行权限；手动复制文件时执行：

```bash
chmod +x cann_tools/*
source ./run_env.sh
python -m pytest -sv test_decode_score_topk_chunk.py test_decode_score_attn_chunk.py
```

run_env.sh 默认 source /home/s00943453/env.sh，并激活当前测试目录的 .venv。可用 SYY_BASE_ENV 指定现有环境配置，用 SYY_CANN_BIN 指定现有 CANN A5 工具目录。运行配置和缓存路径相对本测试目录。

如果不用 run_env.sh，在仓库 syytest 目录手动设置：

```bash
export PYTHONPATH="$(cd ../python/sgl_kernel_npu/sgl_kernel_npu/indexer && pwd)${PYTHONPATH:+:$PYTHONPATH}"
```

不使用 sgl_kernel_npu.indexer 的包导入路径，因为包初始化会要求 libsgl_kernel_npu.so；这里直接普通导入同一个源码文件。

## 编译器兼容说明

直接 source 原始 `env.sh` 会选择 `/home/s00943453/bin/bs-master/bishengir-compile`（2026-10-09 构建）。首次测试缺少其相邻 `../lib` 中的 `.O0.bc`；补齐映射后虽然编译通过，score kernel 在设备同步时仍未返回，关闭 multibuffer 也没有解决。具体根因尚未确定，不能把该配置的执行算作通过。

本指导使用现有 CANN A5 工具链：

```text
bishengir-compile-a5
bishengir-opt-a5
hivmc-a5
```

它们位于 `/usr/local/Ascend/cann-9.1.0/tools/bishengir/bin/`。`cann_tools/bishengir-compile` 仅过滤新版 TA 传入、CANN 9.1 不支持的 `--enable-optimized-metaop=...` 参数，然后执行真实 CANN A5 编译器；其余参数原样传递。其他两个工具通过已提交的 shell 入口选择。TA 包和仓库 kernel 源码没有为这个适配而修改。

本次诊断临时创建的 `/home/s00943453/bin/lib` 链接已经移除，原始 `env.sh` 和 bs-master 二进制未修改。不同编译器必须使用不同 `TRITON_CACHE_DIR`，避免命中另一工具链生成的二进制。

如果测试停住，可使用 timeout 300 python -m pytest -sv 限制运行时间。不要把同步未完成的日志视为有效结果；先确认使用 `run_env.sh`、正确的工具链和专用缓存，再检查设备是否有其他作业。

## 实测结果

默认参数、多缓冲开启、物理 NPU 3，CPU FP32 参考检查通过：

| kernel | 实际 chunks | FILL_ONLY | kernel median | wrapper median |
|---|---:|---|---:|---:|
| score_topk_chunk | 16 | True | 34.950 us | 70.465 us |
| score_attn_chunk | 8 | 不适用 | 29.484 us | 81.284 us |

attention 最大绝对误差为 `0.000130665`。这是本次预热事件计时的示例结果，重新运行会受主机下发和设备负载影响。

对应最终复验日志：`score_final.log`、`attn_final.log`。首次默认测试日志为 `score_default.log`、`attn_multibuffer.log`。补充用例日志：`score_direct_lse.log`、`attn_lse_tail.log`。

补充用例也均通过参考检查：64K direct-map LSE score（`FILL_ONLY=False`）kernel median 为 `36.633 us`；B=2、QH=8、KVH=2、L=8193、FP16 LSE attention 的 kernel median 为 `30.962 us`，最大绝对误差为 `2.83197e-05`。

## 保存编译 IR

脚本和 run_env.sh 固定 TRITON_ALWAYS_COMPILE=1，不会改回 0。每次运行都重新编译。缓存清理后运行：

```bash
source ./run_env.sh
TRITON_CACHE_DIR="$PWD/cache_" TRITON_ALWAYS_COMPILE=1 \
  python -m pytest -sv test_decode_score_topk_chunk.py test_decode_score_attn_chunk.py
find cache_ -type f \( -name '*.ttadapter' -o -name '*.bcmlir' \)
```

生成的 IR 和日志是本地运行产物，不加入 Git。

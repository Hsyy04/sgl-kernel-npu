# Source from bash: source ./run_env.sh
syy_test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SYY_BASE_ENV:-/home/s00943453/env.sh}" || return 1
source "$syy_test_dir/.venv/bin/activate" || return 1
export SYY_CANN_BIN=${SYY_CANN_BIN:-/usr/local/Ascend/cann-9.1.0/tools/bishengir/bin}
export PATH="$syy_test_dir/cann_tools:$PATH"
# Import the kernel module directly without initializing the native-ops package.
export PYTHONPATH="$syy_test_dir/../python/sgl_kernel_npu/sgl_kernel_npu/indexer${PYTHONPATH:+:$PYTHONPATH}"
# Physical card 3 maps to logical npu:0 in the tests.
export ASCEND_RT_VISIBLE_DEVICES=${SYY_NPU_DEVICE:-3}
export TRITON_ALWAYS_COMPILE=1
export TRITON_DEBUG=0
export TRITON_PRINT_AUTOTUNING=0
export TRITON_CACHE_DIR="$syy_test_dir/cache_cann_compat"
export TRITON_DUMP_DIR="$syy_test_dir/dump_cann_compat"
unset syy_test_dir

from typing import Optional

import numpy as np
import tensorrt as trt
import torch

import tensorrt_llm as tllm
from tensorrt_llm import Builder
from tensorrt_llm.functional import (Tensor, concat, constant, shape, slice,
                                     stack, wan_attention)
from tensorrt_llm.layers import ColumnLinear, RmsNorm, RowLinear
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.module import Module
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

logger.set_level('verbose')

from cuda import cuda, cudart
from wan21 import WanI2VCrossAttention

ADD_DEBUG_TENSOR = True


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class CrossAttention(Module):

    def __init__(
        self,
        dim=5120,
        heads=40,
        dropout=0.0,
        eps=1e-6,
        dtype=trt.float16,
    ):
        super().__init__()

        self.inner_dim = dim
        self.dim = dim
        self.heads = heads
        self.tp_group = None
        self.tp_size = 1

        self.norm_k_img = RmsNorm(dim, eps=eps, dtype=dtype)
        self.norm_q = RmsNorm(dim, eps=eps, dtype=dtype)
        self.norm_k = RmsNorm(dim, eps=eps, dtype=dtype)
        self.q = ColumnLinear(dim,
                              self.inner_dim,
                              bias=True,
                              tp_group=self.tp_group,
                              tp_size=self.tp_size,
                              gather_output=False,
                              dtype=dtype)
        self.k = ColumnLinear(dim,
                              self.inner_dim,
                              bias=True,
                              tp_group=self.tp_group,
                              tp_size=self.tp_size,
                              gather_output=False,
                              dtype=dtype)
        self.v = ColumnLinear(dim,
                              self.inner_dim,
                              bias=True,
                              tp_group=self.tp_group,
                              tp_size=self.tp_size,
                              gather_output=False,
                              dtype=dtype)
        self.k_img = ColumnLinear(dim,
                                  self.inner_dim,
                                  bias=True,
                                  tp_group=self.tp_group,
                                  tp_size=self.tp_size,
                                  gather_output=False,
                                  dtype=dtype)
        self.v_img = ColumnLinear(dim,
                                  self.inner_dim,
                                  bias=True,
                                  tp_group=self.tp_group,
                                  tp_size=self.tp_size,
                                  gather_output=False,
                                  dtype=dtype)

        self.o = RowLinear(self.inner_dim,
                           dim,
                           bias=True,
                           tp_group=self.tp_group,
                           tp_size=self.tp_size,
                           dtype=dtype)

    # i2v
    def forward(self,
                hidden_states: Tensor,
                cu_seqlens_q: Tensor,
                cu_seqlens_kv: Tensor,
                encoder_hidden_states: Optional[Tensor] = None):
        batch_size = shape(encoder_hidden_states, 0)
        encoder_context_len = shape(encoder_hidden_states, 1)
        image_context_len = encoder_context_len - 512

        starts = concat([0, 0, 0])
        sizes = concat([batch_size, image_context_len, self.dim])

        encoder_hidden_states_img = slice(encoder_hidden_states,
                                          starts=starts,
                                          sizes=sizes)

        starts = concat([0, image_context_len, 0])
        sizes = concat([batch_size, 512, self.dim])

        encoder_hidden_states = slice(encoder_hidden_states,
                                      starts=starts,
                                      sizes=sizes)

        query = self.q(hidden_states)
        key = self.k(encoder_hidden_states)
        value = self.v(encoder_hidden_states)

        query = self.norm_q(query)

        key = self.norm_k(key)

        # `context` projections.
        key_img = self.k_img(encoder_hidden_states_img)

        key_img = self.norm_k_img(key_img)

        value_img = self.v_img(encoder_hidden_states_img)

        kv_img = stack([key_img, value_img], dim=2)

        print('kv img shape: ', kv_img.shape)
        print('query', query.shape)

        hidden_states_img = wan_attention(
            query,
            kv_img,
            cu_seqlens_q,
            cu_seqlens_kv,
            self.heads,
            self.dim // self.heads,
            q_scaling=1,
        )
        return hidden_states_img

        kv_packed = concat([key, value], dim=-1)

        cu_seqlen_q = constant(np.array([0, 49392]), as_dtype=trt.int32)
        cu_seqlen_kv = constant(np.array([0, 512]), as_dtype=trt.int32)

        hidden_states = wan_attention(
            query,
            kv_packed,
            cu_seqlen_q,
            cu_seqlen_kv,
            self.heads,
            self.dim // self.heads,
            q_scaling=False,
        )
        hidden_states = hidden_states + hidden_states_img
        return hidden_states

        hidden_states = self.o(hidden_states)
        return hidden_states


torch_attn = WanI2VCrossAttention(5120, 40).to(torch.float16)
state_dict = torch_attn.state_dict()

attn = CrossAttention()

for k, v in attn.named_parameters():
    v.value = state_dict[k]

mapping = Mapping(world_size=1, rank=0, gpus_per_node=1)
runtime = None
builder = Builder()

builder_config = builder.create_builder_config(
    name='corss_attention',
    precision='float16',
    timing_cache='model.cache',
    tensor_parallel=1,  # TP only
    use_refit=False,
    gather_context_logits=False,
    gather_generation_logits=False,
    strongly_typed=True,
)

network = builder.create_network()
network.plugin_config.to_legacy_setting()

network.plugin_config.gpt_attention_plugin = 'float16'
network.plugin_config.gemm_plugin = 'float16'

network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
network.plugin_config.remove_input_padding = False

with net_guard(network):
    # Initialize model
    network.set_named_parameters(attn.named_parameters())
    # Prepare inputs
    hidden_states = tllm.Tensor(
        name='hidden_states',
        dtype=trt.float16,
        shape=(1, 49392, 5120),
        is_network_input=True,
        network=network,
        location=trt.TensorLocation.DEVICE,
    )
    encoder_hidden_states = tllm.Tensor(
        name='encoder_hidden_states',
        dtype=trt.float16,
        shape=(1, 769, 5120),
        is_network_input=True,
        network=network,
        location=trt.TensorLocation.DEVICE,
    )
    cu_seqlen_q = tllm.Tensor(name="cu_seqlen_q",
                              dtype=trt.int32,
                              shape=(2, ),
                              network=network,
                              location=trt.TensorLocation.DEVICE)
    cu_seqlen_kv = tllm.Tensor(name="cu_seqlen_kv",
                               dtype=trt.int32,
                               shape=(2, ),
                               network=network,
                               location=trt.TensorLocation.DEVICE)

    out = attn(hidden_states, cu_seqlen_q, cu_seqlen_kv, encoder_hidden_states)
    out.mark_output('out', 'float16')

engine_buffer = builder.build_engine(network, builder_config)
runtime = tllm.runtime.generation._Runtime(engine_buffer, mapping)

print(engine_buffer)
print(runtime)

ctx_buffer = {
    'hidden_states':
    torch.randn(1, 49392, 5120, dtype=torch.float16, device='cuda'),
    'cu_seqlen_q':
    torch.tensor([0, 49392], dtype=torch.int32, device='cuda'),
    'cu_seqlen_kv':
    torch.tensor([0, 257], dtype=torch.int32, device='cuda'),
    'encoder_hidden_states':
    torch.randn(1, 769, 5120, dtype=torch.float16, device='cuda'),
}

ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

context = runtime.ctx_context
runtime._set_shape(context, ctx_shape)
runtime._set_buffer(context, ctx_buffer)
runtime._run(context)
import time

time.sleep(5)
torch.cuda.synchronize()
stream = torch.cuda.default_stream().cuda_stream
cuda_call(cudart.cudaStreamSynchronize(stream))

res = ctx_buffer['out']

torch_attn = torch_attn.cuda().eval()
with torch.inference_mode(True):
    torch_res = torch_attn(ctx_buffer['hidden_states'],
                           ctx_buffer['encoder_hidden_states'])

print(res.shape)
print(torch_res.shape)

print(res)
print(torch.all(res == 0))

print(torch_res)

torch.testing.assert_close(res, torch_res, atol=0.01, rtol=0.01)

# logger = trt.Logger(trt.Logger.VERBOSE)
# # logger = trt.Logger(trt.Logger.ERROR)

# builder = trt.Builder(logger)
# trt_network = builder.create_network(
#     1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# profile = builder.create_optimization_profile()

# network = tllm.Network()
# network._init(trt_network)

# tllm._common.set_network(network)

# hidden_states = tllm.Tensor(
#     name='hidden_states',
#     dtype=trt.float16,
#     shape=(1, 49392, 5120),
#     is_network_input=True,
#     network=network,
#     location=trt.TensorLocation.DEVICE,
# )
# encoder_hidden_states = tllm.Tensor(
#     name='encoder_hidden_states',
#     dtype=trt.float16,
#     shape=(1, 769, 5120),
#     is_network_input=True,
#     network=network,
#     location=trt.TensorLocation.DEVICE,
# )

# out = attn(hidden_states, encoder_hidden_states)
# # network._trt_network.mark_output(out)
# out.mark_output('output', trt.float16)

# config = builder.create_builder_config()
# config.set_flag(trt.BuilderFlag.FP16)
# config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 34)  # 1 MiB

# serialized_engine = builder.build_serialized_network(network._trt_network,
#                                                      config)

# with open("sample.engine.trt", "wb") as f:
#     f.write(serialized_engine)

# with open("sample.engine.trt", "rb") as f, trt.Runtime(logger) as runtime:
#     engine = runtime.deserialize_cuda_engine(f.read())
#     print(engine)

# context = engine.create_execution_context()
# print(engine, context)

# print(engine[0])
# print(engine[1])
# print(engine[2])

# stream = torch.cuda.default_stream().cuda_stream
# print(stream)

# print(context.execute_async_v3)

# tensor_dict = {
#     'hidden_states': 0,
#     'encoder_hidden_states': 1,
#     'output': 2,
# }

# hidden_states = torch.randn(1, 49392, 5120, dtype=torch.float16, device='cuda')
# encoder_hidden_states = torch.randn(1, 769, 5120, dtype=torch.float16, device='cuda')
# output = torch.randn(1, 49392, 5120, dtype=torch.float16, device='cuda')

# bindings = [
#     hidden_states.data_ptr(),
#     encoder_hidden_states.data_ptr(),
#     output.data_ptr(),
# ]

# print(bindings)

# cuda_call(cudart.cudaStreamSynchronize(stream))

# for k, v in tensor_dict.items():
#     context.set_tensor_address(k, bindings[v])

# context.execute_async_v3(stream)
# cuda_call(cudart.cudaStreamSynchronize(stream))

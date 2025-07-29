import os

import tensorrt as trt
import torch
from cuda import cuda, cudart

import tensorrt_llm as tllm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # To use GPU 0


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


logger = trt.Logger(trt.Logger.VERBOSE)

builder = trt.Builder(logger)
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 35)  # 1 MiB

trt_network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()

network = tllm.Network()
network._init(trt_network)

tllm._common.set_network(network)

x = tllm.Tensor(name="query",
                dtype=trt.float16,
                shape=(1, 5120),
                network=network,
                location=trt.TensorLocation.DEVICE)

layer = trt_network.add_cast(x.trt_tensor, trt.float32)
trt_network.add_

out.mark_output('out', trt.float16)

serialized_engine = builder.build_serialized_network(network._trt_network,
                                                     config)

with open("wan_attention.engine", "wb") as f:
    f.write(serialized_engine)

with open("wan_attention.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    print(engine)

context = engine.create_execution_context()
print(engine, context)

print(engine[0])
print(engine[1])

stream = torch.cuda.default_stream().cuda_stream
print(stream)

print(context.execute_async_v3)

tensor_dict = {
    'x': 0,
    'out': 1,
}

x = torch.randn(1, 5120, dtype=torch.float16, device='cuda') * 4
out = torch.randn(1, 5120, dtype=torch.float16, device='cuda')

bindings = [
    x.data_ptr(),
    out.data_ptr(),
]

print(bindings)

cuda_call(cudart.cudaStreamSynchronize(stream))

for k, v in tensor_dict.items():
    context.set_tensor_address(k, bindings[v])

context.execute_async_v3(stream)

cuda_call(cudart.cudaStreamSynchronize(stream))

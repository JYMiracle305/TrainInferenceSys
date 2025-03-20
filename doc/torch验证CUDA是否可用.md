在 PyTorch 中，可以通过以下代码来验证 CUDA 是否可用，并获取相关的 CUDA 信息：

```python
import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Device properties:", torch.cuda.get_device_properties(torch.cuda.current_device()))
else:
    print("CUDA is not available. Running on CPU.")
```

### 代码说明：
1. **`torch.cuda.is_available()`**：
   - 用于检查 CUDA 是否可用。如果返回 `True`，表示 PyTorch 可以使用 CUDA；如果返回 `False`，则表示无法使用 CUDA。
2. **`torch.version.cuda`**：
   - 返回 PyTorch 所使用的 CUDA 版本。
3. **`torch.cuda.device_count()`**：
   - 返回系统中可用的 CUDA 设备数量。
4. **`torch.cuda.current_device()`**：
   - 返回当前使用的 CUDA 设备的索引。
5. **`torch.cuda.get_device_name(device)`**：
   - 返回指定设备的名称。
6. **`torch.cuda.get_device_properties(device)`**：
   - 返回指定设备的属性，包括显存大小、计算能力等。

### 示例输出：
如果 CUDA 可用，输出可能如下：
```
CUDA is available!
CUDA version: 11.1
CUDA device count: 1
Current CUDA device: 0
Device name: NVIDIA GeForce RTX 3080
Device properties: _CudaDeviceProperties(name='NVIDIA GeForce RTX 3080', major=8, minor=6, total_memory=10240MB, multi_processor_count=68)
```

如果 CUDA 不可用，输出可能如下：
```
CUDA is not available. Running on CPU.
```

### 注意事项：
1. **确保安装了 CUDA 和 PyTorch**：
   - 验证 CUDA 是否可用之前，需要确保你的系统已经安装了 CUDA，并且 PyTorch 是支持 CUDA 的版本。
   - 可以通过 `pip install torch torchvision torchaudio` 安装 PyTorch。
2. **检查显卡驱动**：
   - 如果 CUDA 不可用，可能是因为显卡驱动版本过低或不兼容。确保你的显卡驱动是最新的，并且与 CUDA 版本兼容。
3. **环境变量**：
   - 确保环境变量 `CUDA_PATH` 和 `CUDA_HOME` 已正确设置，指向你的 CUDA 安装路径。

通过这段代码，你可以快速验证 PyTorch 是否能够使用 CUDA，并获取相关的设备信息。
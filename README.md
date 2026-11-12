# NumPy 图像特征编码器

这是一个使用 NumPy 实现的 OpenCLIP ViT-B/32 图像编码器。程序接收一批 RGB 图片，完成图像预处理、Patch 编码、Transformer 前向计算和特征归一化，最终为每张图片生成一个 512 维特征向量。

本项目只实现推理过程，不训练或更新模型参数。运行编码器无需 PyTorch、OpenCLIP 或 GPU。

## 功能概览

- 将任意尺寸的 RGB 图片缩放、中心裁剪并标准化为 `224×224`。
- 将图片划分为 `7×7` 个 `32×32` Patch。
- 使用 12 层 Transformer 和 12 头自注意力提取图像特征。
- 输出形状为 `[B, 512]` 的 `float32` 数组，其中每一行都是单位向量。
- 提供分阶段数值测试，便于验证中间张量和最终输出。

编码器中的主要张量形状如下：

| 阶段 | 形状 | 说明 |
| --- | --- | --- |
| 预处理后图片 | `[B, 3, 224, 224]` | NCHW 格式的 RGB 图片 |
| 展平后的 Patch | `[B, 49, 3072]` | 49 个 `3×32×32` Patch |
| Transformer 输入/输出 | `[B, 50, 768]` | 1 个 CLS token 和 49 个 Patch token |
| 图像特征 | `[B, 512]` | 归一化后的最终输出 |

## 项目主要结构

```text
.
├── image_encoder.py          # 图像编码器主体
├── helpers.py                # LayerNorm、GELU、注意力权重和向量归一化
├── preprocess_image.py       # 图片缩放、裁剪和标准化
├── runtime.py                # 配置、权重及图片加载
├── run.py                    # 特征提取命令行入口
├── check.py                  # 分阶段数值验证
├── transformer_steps.py      # 小规模 Transformer 计算示例
├── config.json               # 模型结构和预处理参数
├── requirements.txt          # Python 依赖
├── weights/
│   ├── image_encoder.npz     # 图像编码器权重
│   └── manifest.json         # 权重形状、类型和 SHA-256
└── assets/
    ├── images/               # 示例图片
    ├── images.json           # 图片 ID、来源和 SHA-256
    ├── reference_outputs.npz # 分阶段参考输出
    └── manifest.json         # 参考输出说明和校验值
```


## 环境要求

- Python 3.9 或更高版本
- NumPy
- Pillow

安装全部依赖：

```bash
python -m pip install -r requirements.txt
```


## 快速开始

### 处理仓库中的示例图片

```bash
python run.py --output output_tokens.npz
```

程序默认读取 `assets/images.json` 中登记的图片，并将图片 ID 和对应的 512 维特征保存到 `output_tokens.npz`。

### 处理自定义图片

```bash
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

输入图片可以具有不同尺寸。`preprocess_image.py` 会自动完成 RGB 转换、缩放、中心裁剪和标准化，输出特征的顺序与命令行中的图片顺序一致。

### 读取输出

```python
import numpy as np

with np.load("output_tokens.npz", allow_pickle=False) as result:
    print(result.files)
    print(result["ids"])
    print(result["output_tokens"].shape)
```

## 实现原理

1. **图像预处理**：将图片转换为 RGB，按配置缩放并中心裁剪到 `224×224`，随后使用指定的通道均值和标准差进行标准化。
2. **Patch 编码**：将图片按从左到右、从上到下的顺序划分为 49 个 Patch，并将每个 Patch 从 3072 维投影到 768 维。
3. **Token 组合**：在 Patch token 前添加 CLS token，再加入学习得到的位置编码。
4. **Transformer**：依次执行 12 个 Transformer block。每个 block 包含 LayerNorm、多头自注意力、MLP 和两次残差连接。
5. **特征输出**：选取最后一层的 CLS token，经过 LayerNorm 和输出投影后得到 512 维向量，最后将其归一化为单位长度。

实现遵循以下参数布局：

- 自注意力包含 12 个 head，每个 head 的维度为 64。
- 注意力和 MLP 线性层权重保存为 `[out, in]`，矩阵乘法时使用其转置。
- `visual.proj` 保存为 `[in, out] = [768, 512]`，使用时不转置。
- 模型参数和中间结果使用 `float32`。

## 数据与模型文件

运行示例和数值测试所需的文件位于以下目录：

- `assets/images/`：10 张示例图片。
- [`assets/images.json`](assets/images.json)：图片 ID、本地路径、来源链接和 SHA-256。
- `assets/reference_outputs.npz`：预处理、Patch、Transformer 和最终输出的参考数组。
- `weights/image_encoder.npz`：与本实现配套的模型权重。
- `weights/manifest.json`：权重张量的名称、形状、数据类型和 SHA-256。

完整项目包已经包含示例图片、参考输出和模型权重，无需额外下载。也可以使用自己拍摄或具有合法使用权限的 JPG/PNG 图片进行测试。

## 数值验证

验证数据、配置和权重是否完整：

```bash
python check.py --stage assets
```

按计算顺序运行各阶段测试：

```bash
python check.py --stage preprocess
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch
```

一次运行全部测试：

```bash
python check.py --stage all
```

`check.py` 会检查返回值的形状、`float32` 数据类型、有限性以及与独立 FP32 参考结果的误差：

- 预处理结果：`atol=2e-6, rtol=2e-4`
- 中间 token：`atol=2e-4, rtol=2e-4`
- 最终输出：`atol=2e-5, rtol=2e-4`

`assets/reference_outputs.npz` 仅用于验证，编码器推理过程不会读取或返回其中缓存的结果。



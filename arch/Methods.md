# NTIRE 2021 Low-level Methods

## Deblurring
### overview
NTIRE 2021 Challenge on Image Deblurring
Real-Time Quantized Image Super-Resolution on Mobile NPUs, Mobile AI 2021 Challenge: Report
### track 1: deblur and SR
Champiam: DBPN
### track 2: deblur and deblock

## multi-frame SR
Champiam: EBSR: Feature Enhanced Burst Super-Resolution with Deformable Alignment

### Mobile AI: Image SR
challenge:
1. accuracy drop resulted from model quantization
2. tiny model and FLOPs
3. specific deveice & platform, though perform well on desktop CPU & GPU



### 各种tricks
1. ImageNet Pretrain
2. 复制图片作为输入,self-similarity
3. X2 Pretrain
4. DF2K + DIV2K
5. 不将RGB视为等价通道，加强某一个通道，人造raw
6. multiple tasks
7. 超采样
8. 数据筛选，通过non-reference的方法，调出特别适合的patch
9. 针对指标，增加优化约束，如图像的二阶梯度
   现有图像恢复的瓶颈是什么？是纹理还是色彩，还是更加可量化的高阶特征？
   《A Patch-centric Error Analysis of Image Super-Resolution》

注意：
一些在bicubic only的情况下没有特别大提升效果的trick
在real-world的情况下，可能有比较大的提升
例如：数据筛选、最后一层dropout等


## 主要目标
在NTIRE2022中拿下优秀的名词
主要工作是基于Transformer网络结构的改进

## Efficient Transformer
![](assets/Methods-e1918815.png)
### Linear Transformer
Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention


### Performer
Rethinking Attention with Performers
gaussian_orthogonal_random_matrix的作用和Linear有何区别？

## Methods
### performance
ESA + CW效果好，但是下采样6倍可能不合理，可以探索
MESA的方法基本没啥提升


### Runtime
由于使用了BSConvU导致深度很深
最主要是Layernorm使得网络运行时间长
另外一个瓶颈来源于ESA和concat一个较大的feature导致的

想办法减少网络深度，
并行化ESA，
将concat变为对各个feature进行cw加权后add
### Params
BSConvU使得网络的参数量不大，暂时是优势

### FLOPs
现在也有较大的提升

### Activation
减少使用conv，多用Linear就行

### Memory
没有进一步扩大
下降memory的方法暂时不清楚
开大Kernal size就行:D

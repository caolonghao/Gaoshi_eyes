## 测试环境安装
测试环境为 Ubuntu 22.04 LTS, gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)

### 1. 安装Python环境依赖

在conda虚拟环境下安装pytorch
```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
安装nnunetv2
```
    pip install nnunetv2
```
安装flask
```
    pip install flask
```
安装glob
```
    pip install glob
```
### 2. 运行server
更改代码文件中模型文件夹路径 `model_folder = '/share1/clh/gaoshi_model_pack'`

运行server服务
```
python server.py
```
### 3. 测试
```
python client.py
```
模型将会对`test_images`文件夹中的文件进行分割，并将结果保存至`pred_results`文件夹中。
注意，由于nnunet的多模态处理特性，对于单模态数据，`test_images`文件夹中的文件名需要带有`_0000`后缀，如`48_0000.png`

TODO: 现在如果预先将图像读取为array，使用`predictor.predict_from_list_of_npy_arrays`推理，速度会明显慢于`predictor.predict_from_files`，事实上，读取文件并不是瓶颈，若直接使用`NaturalImage2DIO`读取文件，三个文件仅耗时0.01s，具体原因可能需要后续读代码
| 序号 | 迭代次数 | 模型 | 学习率 | 批次大小 | L2正则化 | 特征  |   计划器   |     正确率      |  测试集  |             日期             |
| :--: | :------: | ---- | :----: | :------: | :------: | ----- | :--------: | :-------------: | :------: | :--------------------------: |
|  1   |    50    | /    |  1e-4  |    32    |   0.1    | V0    |    None    | [99.575,91.667] | [90.602] |     2023-03-19 15:56:23      |
|  2   |    80    | CT1  |  2e-4  |    32    |   0.1    | V0    | LR(30,0.3) |  [100, 96.63]   | [98.88]  |     2023-03-22 22:42:15      |
|  3   |    80    | CT0  |  2e-4  |    32    |   0.1    | V0    | LR(30,0.3) |  [99.91,96.25]  | [97.77]  |     2023-03-22 23:05:52      |
|  4   |    80    | CT1  |  2e-4  |    32    |   0.1    | V0    | LR(30,0.3) |  [99.53,97.38]  | [99.26]  |        2023-03-23 08:        |
|  5   |    60    | SET  |  8e-4  |    64    |   0.1    | V2    | LR(30,0.3) |  [100, 92.33]   | [92.05]  | 2023-03-30 20:58:50-21:05:06 |
|  6   |    60    | CMT0 |  6e-4  |    64    |   0.1    | V2    | LR(30,0.3) |  [99.90, 96.0]  | [96.69]  |     2023-03-30 21:42:45      |
|  7   |    60    | CT0  |  6e-4  |    64    |   0.1    | V2    | LR(30,0.3) |   [100,95.33]   | [95.36]  |     2023-03-30 22:01:00      |
|  8   |    60    | CT1  |  6e-4  |    64    |   0.1    | V2    | LR(30,0.3) |    [100, 97]    | [96.03]  |     2023-04-02 14:17:59      |
|  9   |    80    | CT1  |  4e-4  |    64    |   0.1    | V2    | LR(30,0.3) |    [100,97]     | [98.01]  |     2023-04-02 14:48:57      |
|  10  |    60    | CMT1 |  6e-4  |    64    |   0.1    | V2    | LR(30,0.3) |    [100,98]     | [98.01]  |     2023-04-02 15:08:18      |
|  11  |    60    | CT0  |  6e-4  |    64    |   0.1    | V2(2) | LR(30,0.3) | [99.33, 95.00]  | [98.01]  |     2023-04-02 15:53:26      |
|  12  |    60    | TT0  |  1e-3  |    64    |   0.1    | V2    | LR(30,0.3) |  [99.71,93.33]  | [92.71]  |     2023-04-03 09:44:13      |
|      |          |      |        |          |          |       |            |                 |          |                              |
|      |          |      |        |          |          |       |            |                 |          |                              |

CT1: CNN_Transformer(use pretrain) CT0: CNN_Transformer(no use pretrain) SET:(speech emotion transformer)

CMT0: CNN_ML_Transformer(no use pretrain) CMT1: CNN_ML_Transformer(use pretrain)

TT0: Transformer_TIM(no use pretrain)

V0: 同一目录下的文件连接起来，然后裁成10s每段的音频   V2: 短的音频补零至10s，长的音频截断成10s

1: addition: 	d_model: 256, n_head: 8, d_qkv: 64, d_ff: 1024, n_layer: 4	dilation: 6, filters: 39, drop_rate: 0.1	seq_len: 313, scheduler_type: 0	在训练时，可能由于学习率比较大，导致在23个epoch左右时，验证集上的准确率出现大幅下降。 后面应该加入计划器

2 4 : mode: train model_name: train_drop1_mfcc_smoothTrue_epoch80_l2re1_lr0002_pretrainTrue   epochs: 80 lr: 0.0002 batch_size: 32 weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 300 initial_lr: 0.08 **load_weight: True** d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 2 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 seq_len: 313 num_class: 2 	TIM使用了weightlayer

3: mode: train model_name: train_drop1_mfcc_smoothTrue_epoch80_l2re1_lr0002_pretrainFalse epochs: 80 lr: 0.0002 batch_size: 32 weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 300 initial_lr: 0.08 **load_weight: False** d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 2 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 seq_len: 313 num_class: 2  TIM使用了weightlayer

5: mode: train model_name: SET_official_train_order3_drop1_mfcc_smoothTrue_epoch60_l2re1_lr0008_pretrainFalse epochs: 60 lr: 0.0008 batch_size: 64 weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 400 initial_lr: 0.05 **load_weight: False** order: 3 is_mask: False d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 3 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 is_weight: True seq_len: 313 num_class: 2   将模型替换成了SET_official，特征的划分方式为V2  spilt_rate = [0.7,0.2,0.1]

6: mode: train model_name: CNN_ML_Transformer_train_order3_drop1_mfcc_smoothTrue_epoch60_l2re1_lr0006_pretrainFalse epochs: 60 lr: 0.0006 batch_size: 64 weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 400 initial_lr: 0.05 **load_weight: False** order: 3 is_mask: False d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 3 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 is_weight: True seq_len: 313 num_class: 2  将cnn_transformer中的Transformer部分加上了多重表示（类似于TIM-Net中的weight-layer）spilt_rate = [0.7,0.2,0.1] 特征的划分方式为V2

7: mode: train model_name: CNN_Transformer_train_order3_drop1_mfcc_smoothTrue_epoch60_l2re1_lr0006_pretrainFalse epochs: 60 lr: 0.0006 batch_size: 64 weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 400 initial_lr: 0.05 **load_weight: False** order: 3 is_mask: False d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 3 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 is_weight: True seq_len: 313 num_class: 2  使用原来的cnn_transformer模型， spilt_rate = [0.7,0.2,0.1],特征的划分方式为V2。

8 9: mode: train model_name: CNN_Transformer_train_order3_drop1_mfcc_epoch60_l2re1_lr0006_pretrainTrue epochs: 60 lr: 0.0006 batch_size: 64 spilt_rate: [0.7, 0.2, 0.1] weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 model_type: CNN_Transformer data_type: mfcc save: True scheduler_type: 1 warmup: 400 initial_lr: 0.05 **load_weight: True** dataset_name: MODMA order: 3 version: V2 is_mask: False d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 3 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 is_weight: True seq_len: 313 num_class: 2   使用原来的cnn_transformer模型， spilt_rate = [0.7,0.2,0.1],特征的划分方式为V2。

10: mode: train model_name: CNN_ML_Transformer_train_order3_drop1_mfcc_epoch60_l2re1_lr0006_pretrainTrue epochs: 60 lr: 0.0006 batch_size: 64 spilt_rate: [0.7, 0.2, 0.1] weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 model_type: CNN_ML_Transformer data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 400 initial_lr: 0.05 load_weight: True dataset_name: MODMA order: 3 version: V2 is_mask: False d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 3 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 is_weight: True seq_len: 313 num_class: 2 加载预训练模型

11: mode: train model_name: CNN_Transformer_train_order2_drop1_mfcc_epoch60_l2re1_lr0006_pretrainFalse epochs: 60 lr: 0.0006 batch_size: 64 spilt_rate: [0.7, 0.2, 0.1] weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 model_type: CNN_Transformer data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 400 initial_lr: 0.05 load_weight: False dataset_name: MODMA order: 2 version: V2 is_mask: False d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 3 feature_dim: 26 dilation: 8 filters: 26 kernel_size: 2 drop_rate: 0.1 is_weight: True seq_len: 313 num_class: 2  改用26维的特征 filters也是26

12: mode: train model_name: Transformer_TIM_train_order3_drop1_mfcc_epoch60_l2re1_lr001_pretrainFalse epochs: 60 lr: 0.001 pretrain_model_path: pretrain_model.pt batch_size: 64 spilt_rate: [0.7, 0.2, 0.1] weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 model_type: Transformer_TIM data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 400 initial_lr: 0.05 load_weight: False dataset_name: MODMA order: 3 version: V2 is_mask: False d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 3 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 is_weight: True seq_len: 313 num_class: 2  忘记更新Transformer的参数了

在全连接层加入dropout后可以一定程度上减小模型的抖动幅度

直接使用TIM进行MODMA的实验效果也很好，不逊色与CT0,CT1这些，甚至会更好。

使用torchaudio提取特征（性能差但是速度快）与使用librosa提取特征相比，性能上会有一定的差异(3%)

torchaudio.compliance.kaldi.mfcc也可以用于提取MFCC特征，torchaudio.functional.compute_deltas可用于计算差分

使用nnAudio提取特征时，模型容易过拟合

主要花费时间的是resample，使用torchaudio resample后，系统性能仍有较大差异。

使用librosa读取文件，再使用torchaudio保存文件，不会出现精度损失

torchaudio读出来的数据和librosa读出来的数据基本一致。

使用pytorch自带的Transformer，Transformer会有较大性能的提升。



在IEMOCAP上预训练模型

| 序号 | 迭代次数 | 学习率 | 批次大小 | L2正则化 | 标签平滑 | 计划器 |     优化器      |    正确率     | 测试集  |        日期         |
| :--: | :------: | :----: | :------: | :------: | :------: | :----: | :-------------: | :-----------: | :-----: | :-----------------: |
|  1   |   100    |  2e-4  |    32    |   0.1    |   true   |  None  | Adam(0.93,0.98) | [96.54,96.27] | [96.55] | 2023-03-22 20:43:48 |
|      |          |        |          |          |          |        |                 |               |         |                     |

1：mode: pretrain model_name: pretrain_drop1_mfcc_smoothTrue_epoch100_l2re1_lr0002 epochs: 100 lr: 0.0002 batch_size: 32 weight_decay: 0.1 smooth: True beta1: 0.93 beta2: 0.98 gamma: 0.3 step_size: 30 random_seed: 34 data_type: mfcc save: True augment: False scheduler_type: 1 warmup: 300 initial_lr: 0.08 d_model: 256 n_head: 8 d_qkv: 64 d_ff: 1024 n_layer: 2 feature_dim: 39 dilation: 8 filters: 39 kernel_size: 2 drop_rate: 0.1 seq_len: 313 num_class: 6   TIM不需要使用weight_layer



由于二阶差分与MFCC呈反相关关系，所以可以选择去除二阶差分，效果无明显差异。只用13阶的MFCC也能取得不错的结果...

训练好的模型在直接截断/补零的音频上效果并不好...

TIM中的dilation比较重要，较大的dilation可以增加模型的感受野范围

使用<10s补零，大于10s分段补零的数据效果与之前同类别连接补零的效果差不多。

1. 在截断，补零的语音库中fine-tune  ×
2. 将量表分数也作为分类依据（分配一个较小的权重）无论是作为输入，还是作为输出计算loss，几乎无增益  ×
3. 区分中英文？×
4. 实现一下BertForSequenceClassification
5. 元学习，目前来看效果不是很好，暂时放弃
6. 使用域适应方法只用TIM模型在语音情感数据库和MODMA上训练出一个预训练模型出来
   1. DANN方法不太行，域损失降不下来
7. 对TIM网络的改进
   1. 将下一级的TAB输出减去上一级的TAB输出
   2. 用并行化处理下面一行的dilation为[2 4 6 8 16 32 64 128] 将下面一行的模型参数作为上面一行的模型参数


在语音情感识别上预训练模型，放在抑郁语音上
只用一阶差分
记录实验中一些东西（比较不同网络结构的性能差异）

IEMOCAP: 10s		num_class: 6

CASIA: 6s				num_class: 6

RAVDESS: 6s			num_class: 8


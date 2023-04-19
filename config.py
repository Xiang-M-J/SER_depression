# beta1 = 0.93
# beta2 = 0.98
# gamma = 0.3
# step_size = 50
# random_seed = 34
# data_type = "mfcc"  # ["mfcc", "mel"]
# save = True  # 是否保存结果与模型
# augment = True  # 是否使用增强后的数据
# use_scheduler = True
# use_noam = False  # 用于Transformer
# warmup = 300  # noam参数
# initial_lr = 0.08  # 初始学习率(noam)
import yaml
import argparse


class Args:
    def __init__(self,
                 epochs=100,
                 lr=4e-4,
                 batch_size=64,
                 spilt_rate=None,
                 weight_decay=0.1,
                 patience=10,
                 attention_type='AF',
                 multi_type='ADD_DIFF',
                 pretrain_model_path="iemocap_pretrain.pt",
                 optimizer_type=2,
                 beta1=0.93,
                 beta2=0.98,
                 random_seed=34,
                 model_type="",
                 data_type="mfcc",
                 dataset_name="MODMA",
                 save=True,
                 augment=False,
                 scheduler_type=1,
                 gamma=0.3,
                 step_size=30,
                 warmup=400,
                 initial_lr=0.05,
                 order=3,
                 version='V2',
                 filters=39,
                 kernel_size=2,
                 drop_rate=0.1,
                 dilation=8,
                 is_weight=False,
                 is_cluster=False,
                 is_mask=False,
                 d_model=256,
                 n_head=8,
                 n_layer=3,
                 dim_feed_forward=1024,
                 d_qkv=64,
                 seq_len=313,
                 num_class=6,
                 load_weight=False,
                 ):
        """
        Args:
            optimizer_type: 优化器种类(0: SGD, 1:Adam, 2:AdamW)
            beta1: adam优化器参数
            beta2: adam优化器参数
            random_seed: 随机数种子
            data_type: 数据类型
            save: 是否保存模型和结果
            augment: 是否启用增强
            scheduler_type: scheduler类型 [0:None, 1:'LR', 2:'Noam', 3:'cosine']
            gamma: LR scheduler参数
            step_size: LR scheduler参数
            warmup: Warm scheduler参数
            is_weight: 是否对不同dilation的层进行加权融合
            attention_type: 注意力机制的形式 ['MH':多头注意力, 'AF': attention free ]

        """
        if spilt_rate is None:
            spilt_rate = [0.8, 0.1, 0.1]
        self.model_name = " "
        self.epochs = epochs
        self.lr = lr
        self.pretrain_model_path = pretrain_model_path
        self.batch_size = batch_size
        self.spilt_rate = spilt_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.step_size = step_size
        self.random_seed = random_seed
        self.model_type = model_type
        self.data_type = data_type
        self.save = save
        self.augment = augment
        self.scheduler_type = scheduler_type
        self.warmup = warmup
        self.initial_lr = initial_lr
        self.load_weight = load_weight
        self.is_cluster = is_cluster
        self.dataset_name = dataset_name
        self.order = order
        self.version = version
        self.attention_type = attention_type
        self.multi_type = multi_type
        self.patience = patience
        # Transformer begin
        self.is_mask = is_mask
        self.d_model = d_model
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.d_ff = dim_feed_forward
        self.n_layer = n_layer

        # Transformer end

        # CasualConv begin
        self.feature_dim = self.order * 13
        self.dilation = dilation
        self.filters = self.order * 13
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.is_weight = is_weight
        # CasualConv end
        self.seq_len = seq_len
        self.num_class = num_class

    def write(self, name="hyperparameter"):
        with open("./config/" + name + ".yml", 'w', encoding='utf-8') as f:
            yaml.dump(data=self.__dict__, stream=f, allow_unicode=True)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            hyperparameter = yaml.load(f.read(), Loader=yaml.FullLoader)
        for para in hyperparameter:
            self.__setattr__(para, hyperparameter[para])

    def item(self):
        return self.__dict__

    def addition(self) -> str:
        info = "parameter setting:\t"
        for parameter in self.__dict__:
            info += f"{parameter}: {self.__dict__[parameter]}\t"
        info += '\n'
        return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="hyperparameter", type=str)
    args = parser.parse_args()
    arg = Args()
    arg.write(args.name)

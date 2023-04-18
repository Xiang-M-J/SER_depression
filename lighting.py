import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from old.v1.TIM_ import TIMNet
from config import Args
from utils import myDataset, accuracy_cal, plot_matrix, RAVDESS_LABELS


def load_callbacks():
    callbacks = [plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=20,
        min_delta=0.001
    ), plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ), plc.LearningRateMonitor(
        logging_interval='epoch')]
    return callbacks


spilt_rate = [0.8, 0.1, 0.1]  # 训练集、验证集、测试集分割比例
loss_fn = torch.nn.CrossEntropyLoss()


class LitModel(pl.LightningModule):
    def __init__(self, arg: Args, test_num):
        super(LitModel, self).__init__()
        self.arg = arg
        self.test_pred = torch.zeros(test_num)
        self.test_true = torch.zeros(test_num)
        self.net = TIMNet(feature_dim=arg.feature_dim, drop_rate=arg.drop_rate, num_class=arg.num_class,
                          filters=arg.filters, dilation=arg.dilation, kernel_size=arg.kernel_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.net(x)
        loss = loss_fn(z, y)
        correct_num = accuracy_cal(z, y)
        self.log('train_acc', correct_num / y.shape[0],
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.net(x)
        val_loss = loss_fn(z, y)
        self.log("val_loss", val_loss)
        correct_num = accuracy_cal(z, y)
        self.log('val_acc', correct_num / y.shape[0],
                 on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.net(x)
        test_loss = loss_fn(z, y)
        self.log("test_loss", test_loss)
        batch_size = self.arg.batch_size
        self.test_pred[batch_idx * batch_size: batch_idx * batch_size + y.shape[0]] = torch.max(z.data, 1)[1]
        self.test_true[batch_idx * batch_size: batch_idx * batch_size + y.shape[0]] = torch.max(y.data, 1)[1]
        # self.test_pred
        # self.test_true.append(list(torch.max(y.cpu().data, 1)[1]))
        correct_num = accuracy_cal(z, y)
        self.log('test_acc', correct_num / y.shape[0],
                 on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.arg.step_size,
                                                    gamma=self.arg.gamma)
        return [optimizer], [scheduler]


arg = Args()
data = np.load('preprocess/data/RAVDESS.npy', allow_pickle=True).item()
x = data['x']
y = data['y']
arg.num_class = y.shape[1]
Num = x.shape[0]  # 样本数
dataset = myDataset(x, y)  # input shape of x: [样本数，特征维度，时间步]  input shape of y: [样本数，类别数]
train_num = int(Num * spilt_rate[0])
val_num = int(Num * spilt_rate[1])
test_num = Num - train_num - val_num

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_num, val_num, test_num],
                                                                         generator=torch.Generator().manual_seed(34))
train_loader = DataLoader(train_dataset, batch_size=arg.batch_size)
val_loader = DataLoader(val_dataset, batch_size=arg.batch_size)
test_loader = DataLoader(test_dataset, batch_size=arg.batch_size)
torch.set_float32_matmul_precision('medium')
logger = TensorBoardLogger("tb_logs", name="my_model")
model = LitModel(arg, test_num)
trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=load_callbacks(), max_epochs=arg.epochs, logger=logger, precision=16)
# trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model, dataloaders=test_loader, ckpt_path="lightning_logs/version_0/checkpoints/best-epoch=29-val_acc=1"
                                                       ".000.ckpt")
cm = confusion_matrix(model.test_pred, model.test_true, labels=np.arange(arg.num_class))
fig = plot_matrix(cm, labels_name=RAVDESS_LABELS, model_name="test_RAVDESS")
logger.log_graph("model", model)

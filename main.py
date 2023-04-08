from Net import Net_Instance
from config import Args
from utils import load_dataset, dataset_num_class
from DANN import DANNModel
from ADDA import Encoder

args = Args()
spilt_rate = [0.7, 0.2, 0.1]  # 训练集、验证集、测试集分割比例
# seed_everything(args.random_seed)
model_type = 'AT_DeltaTIM'
dataset_name = "MODMA"
args.model_type = model_type
args.dataset_name = dataset_name
args.spilt_rate = spilt_rate

# args.pretrain_model_path = "models/ADDA/tgt_encoder_best.pt"
if __name__ == "__main__":
    print("dataset name: ", args.dataset_name)
    print("model type: ", args.model_type)
    model_name = f"{model_type}_{args.mode}_order{args.order}_drop{str(args.drop_rate).split('.')[-1]}_{args.data_type}_epoch{args.epochs}_l2re{str(args.weight_decay).split('.')[-1]}_lr{str(args.lr).split('.')[-1]}_pretrain{args.load_weight}"
    option = input(f"{args.save} model name: {model_name}, (y(default)/n):")
    if option == '' or option == 'y' or option == 'yes' or option is None:
        args.model_name = model_name
        train_dataset, val_dataset, test_dataset = load_dataset(
            dataset_name, spilt_rate, args.random_seed, order=args.order, version="V2")
        instance = Net_Instance(args)
        instance.train(train_dataset, val_dataset, args.batch_size, args.epochs, args.lr, args.weight_decay,
                       is_mask=False)
        instance.test(test_dataset, batch_size=args.batch_size)
    else:
        print("请修改模型名后再次执行")

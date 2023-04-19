from Net import Net_Instance

from config import Args
from utils import load_dataset, get_newest_file

args = Args()
spilt_rate = [0.6, 0.2, 0.2]  # 训练集、验证集、测试集分割比例
# seed_everything(args.random_seed)
model_type = 'MultiTIM'
dataset_name = "MODMA"
args.model_type = model_type
args.dataset_name = dataset_name
args.spilt_rate = spilt_rate
args.num_class = 2
args.multi_type = "AT_DIFF"
# args.is_cluster = True
args.pretrain_model_path = "models/MultiTIM_train_ADD_DIFF_IEMOCAP_order3_drop1_mfcc_epoch100_l2re2_lr0004_pretrainFalse_clusterFalse.pt"
if __name__ == "__main__":
    print("dataset name: ", args.dataset_name)
    print("model type: ", args.model_type)
    if model_type == "MultiTIM":
        model_name = f"{model_type}_{args.multi_type}_{args.dataset_name}_order{args.order}_drop{str(args.drop_rate).split('.')[-1]}_{args.data_type}_epoch{args.epochs}_l2re{str(args.weight_decay).split('.')[-1]}_lr{str(args.lr).split('.')[-1]}_pretrain{args.load_weight}_cluster{args.is_cluster}"
    else:
        model_name = f"{model_type}_{args.dataset_name}_order{args.order}_drop{str(args.drop_rate).split('.')[-1]}_{args.data_type}_epoch{args.epochs}_l2re{str(args.weight_decay).split('.')[-1]}_lr{str(args.lr).split('.')[-1]}_pretrain{args.load_weight}_cluster{args.is_cluster}"
    option = input(f"{args.save} model name: {model_name}, (y(default)/n):")
    if option == '' or option == 'y' or option == 'yes' or option is None:
        args.model_name = model_name
        train_dataset, val_dataset, test_dataset = load_dataset(
            dataset_name, spilt_rate, args.random_seed, order=args.order, version="V1", is_cluster=args.is_cluster)
        args.seq_len = train_dataset[0][0].shape[-1]
        instance = Net_Instance(args)
        instance.train(train_dataset, val_dataset)
        instance.test(test_dataset, model_path=None)
    else:
        print("请修改模型名后再次执行")

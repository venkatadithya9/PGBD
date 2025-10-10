import argparse


def get_arguments_MCL():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/WRN-16-1-badnet.pth.tar', help='path of student model')
    parser.add_argument('--t_model', type=str,
                        default='./weight/CIFAR10/WRN-16-1-badnet.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='WRN-16-1', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='WRN-16-1', help='name of student')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='badnet')
    parser.add_argument('--trigger_type', type=str,
                        default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/WRN-16-1-badnet.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data used')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="pois", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.1,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Update cav  with weighting or not')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="synth",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=100, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='WRN-16-1', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='WRN-16-1', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'block3.layer.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=5, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='badnet')
    parser.add_argument('--trigger_type', type=str,
                        default='squareTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_1():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='gtsrb', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/GTSRB/vgg19_bn-blended.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=43, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.1,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Update cav  with weighting or not')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="synth",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=100, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='GTSRB', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='vgg19_bn', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='vgg19_bn', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'features.49'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='blended')
    parser.add_argument('--trigger_type', type=str,
                        default='blendTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_2():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/vgg19_bn-badnet.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.9,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Update cav  with weighting or not')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=1,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="synth",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=100, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='vgg19_bn', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='vgg19_bn', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'features.49'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='badnet')
    parser.add_argument('--trigger_type', type=str,
                        default='squareTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_2_convnext():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/convnext_tiny-badnet.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.9,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Updatecav  with weighting or nott')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="synth",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=100, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='convnext_tiny', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='convnext_tiny', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='badnet')
    parser.add_argument('--trigger_type', type=str,
                        default='squareTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=5,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=5,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_2_preact():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/preactresnet18-badnet.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Updatecav  with weighting or nott')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='prectresnet18', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='preactresnet18', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='badnet')
    parser.add_argument('--trigger_type', type=str,
                        default='squareTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_2_preact_tiny():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='tinyImagenet', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/tinyImagenet/preactresnet18-badnet.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Updatecav  with weighting or nott')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='tinyImagenet', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='prectresnet18', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='preactresnet18', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='badnet')
    parser.add_argument('--trigger_type', type=str,
                        default='squareTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_semantic_preact():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='ROF',
                        help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/ROF/resnet50-semantic.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Updatecav  with weighting or nott')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='ROF', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='resnet50', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='resnet50', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.2.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='semantic')
    parser.add_argument('--trigger_type', type=str,
                        default='semanticTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_2_preact34():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/preactresnet18-badnet.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=95.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.9,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Updatecav  with weighting or nott')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=100, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='prectresnet34', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='preactresnet34', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.2.conv3'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='badnet')
    parser.add_argument('--trigger_type', type=str,
                        default='squareTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_3():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/vgg19_bn-blended.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Updatecav  with weighting or nott')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='vgg19_bn', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='vgg19_bn', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'features.49'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='blended')
    parser.add_argument('--trigger_type', type=str,
                        default='blendTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_3_preact():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/preactresnet18-blended.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Updatecav  with weighting or nott')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='preactresnet18', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='preactresnet18', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='blended')
    parser.add_argument('--trigger_type', type=str,
                        default='blendTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_4_preact():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/preactresnet18-trojannn.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Update cav  with weighting or not')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="synth",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='prectresnet18', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='preactresnet18', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='trojannn')
    parser.add_argument('--trigger_type', type=str,
                        default='trojanTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_signal_preact():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/preactresnet18-sig.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav', action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Update cav  with weighting or not')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='prectresnet18', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='preactresnet18', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='sig')
    parser.add_argument('--trigger_type', type=str,
                        default='signalTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_lc_preact():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/preactresnet18-sig.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Update cav  with weighting or not')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='prectresnet18', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='preactresnet18', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='sig')
    parser.add_argument('--trigger_type', type=str,
                        default='signalTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_wanet_preact():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/preactresnet18-wanet.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="trig", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.75,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Update cav  with weighting or not')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="proto",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=-1, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='prectresnet18', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='preactresnet18', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'layer4.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=0, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='wanet')
    parser.add_argument('--trigger_type', type=str,
                        default='wanetTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser


def get_arguments_signal():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--checkpoint_root', type=str,
                        default='./weight/', help='models weight are saved here')
    parser.add_argument('--log_root', type=str,
                        default='./results', help='logs are saved here')
    parser.add_argument('--dataset', type=str,
                        default='CIFAR10', help='name of image dataset')
    parser.add_argument(
        '--model', type=str, default='./weight/CIFAR10/WRN-16-1-signal.pth.tar', help='path of student model')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int,
                        default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='ratio of training data')
    parser.add_argument('--threshold_clean', type=float,
                        default=70.0, help='threshold of save weight')
    parser.add_argument('--threshold_bad', type=float,
                        default=99.0, help='threshold of save weight')
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', type=int, default=1)

    # others
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--note', type=str, default='try',
                        help='note for this run')

    parser.add_argument('--knn_k', type=int, default=3,
                        help='number of centers being considered while clustering classwise for prototypes')
    parser.add_argument('--num_csets', type=int, default=1,
                        help='number of concept sets used')
    parser.add_argument('--cset_size', type=int, default=30,
                        help='size of concept sets used for TCAV')
    parser.add_argument('--cset_type', type=str,
                        default="pois", help='type of concept sets used')
    parser.add_argument('--clean_noise_percentage', type=float, default=0.1,
                        help='Percentage of random noise in clean concept images for trigger concept set')
    parser.add_argument('--distill',  action='store_false',
                        help='Use distillation method or not')
    parser.add_argument('--use_gt_loss',   action='store_false',
                        help='Use ground truth loss or not')
    parser.add_argument('--use_proto',   action='store_false',
                        help='Use prototype loss')
    parser.add_argument('--use_cav_loss',
                        action='store_false', help='Use concept loss')
    parser.add_argument('--loss_type',  type=str,
                        default="L1_cos", help='Type of concept loss')
    parser.add_argument('--wtcav',  type=float, default=10,
                        help='Weight for concept loss')
    parser.add_argument('--loss_interval',  type=int, default=1,
                        help='Interval between applying concept loss')
    parser.add_argument(
        '--use_wandb',   action='store_false', help='Use wandb')
    parser.add_argument('--eval_mode',   action='store_false',
                        help='Run with model in eval mode')
    parser.add_argument('--agg_cav',  action='store_true',
                        help='Use avg of all inter class cav or not')
    parser.add_argument('--delta', type=float, default=0.1,
                        help='Weight of prev iteration proto and cav')
    parser.add_argument('--update_cav',  action='store_true',
                        help='Update cav or not')
    parser.add_argument('--weight_pav',  action='store_true',
                        help='Updatecav  with weighting or nott')
    parser.add_argument('--weight_proto',  action='store_true',
                        help='Update protos with weighting or not')
    parser.add_argument('--update_gap', type=int, default=5,
                        help='Update cav after how many epochs')
    parser.add_argument('--cav_type', type=str, default="synth",
                        help='What protos to be used to get cav')
    parser.add_argument('--sched_delta',  action='store_true',
                        help='Use scheduler for delta or not')
    parser.add_argument('--update_gap_iter', type=int,
                        default=100, help='Update cav after how many epochs')
    parser.add_argument('--dino_acts',  action='store_true',
                        help='Use DINO features before prototype stage')
    parser.add_argument('--dino_acts_mm',  action='store_true',
                        help='Use mapped DINO features before prototype stage')
    parser.add_argument('--use_kmedoids',  action='store_true',
                        help='Use kmedoids instead of kmeans')

    # net and dataset choosen
    parser.add_argument('--data_name', type=str,
                        default='CIFAR10', help='name of dataset')
    parser.add_argument('--t_name', type=str,
                        default='WRN-16-1', help='name of teacher')
    parser.add_argument('--s_name', type=str,
                        default='WRN-16-1', help='name of student')
    parser.add_argument('--bottlenecks',  type=list, default=[
                        'block3.layer.1.conv2'], help='name of conv layer being considered as activation space')

    parser.add_argument('--attack_size', default=50, type=int,
                        help='number of samples for inversion')
    # backdoor attacks
    parser.add_argument('--ml_mmdr', action='store_true',
                        help='Use ML MMDR adaptive backdoor attack')
    parser.add_argument('--inject_portion', type=float,
                        default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int,
                        default=5, help='class of target label')
    parser.add_argument('--attack_method', type=str, default='signal')
    parser.add_argument('--trigger_type', type=str,
                        default='signalTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str,
                        default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3,
                        help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3,
                        help='height of trigger pattern')

    parser.add_argument('--temperature', type=float, default=0.5)

    return parser

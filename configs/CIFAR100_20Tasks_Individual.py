import sys
from types import SimpleNamespace


args = SimpleNamespace()
args.exp_name = 'CIFAR100_20Tasks'
args.rng_seed = 0xCAFFE
args.save_opt = 'best'
args.workers  = 4

args.dataset       = sys.argv[2]
args.train_type    = sys.argv[3]
args.backbone      = sys.argv[4] if len(sys.argv) >= 5 else 'Conv4'
args.pretrain      = sys.argv[5] if len(sys.argv) >= 6 else ''
args.chkpt_postfix = sys.argv[6] if len(sys.argv) >= 7 else ''

args.image_size = 32
args.batch_size = 32
args.padding = 4
args.transform_name = 'standard_cifar'
args.num_classes = 5


optimizer_type_list = []
optimizer_args_list = []
scheduler_type_list = []
scheduler_args_list = []

if args.train_type == 'baseline':
    args.max_epoch, args.param_types = 100, ['all']

    # Param type: all
    optimizer_type_list.append('SGD')
    optimizer_args_list.append({'lr': 1e-2, 'weight_decay': 4e-5, 'momentum': 0.9, 'nesterov': True})
    scheduler_type_list.append('LookUpTableLR')
    scheduler_args_list.append({'look_up_table': [(50, 1e-2), (80, 1e-3), (100, 1e-4)]})

elif args.train_type == 'finetune':
    args.max_epoch, args.param_types = 100, ['all']

    # Param type: all
    optimizer_type_list.append('SGD')
    optimizer_args_list.append({'lr': 5e-3, 'weight_decay': 4e-5, 'momentum': 0.9, 'nesterov': True})
    scheduler_type_list.append('LookUpTableLR')
    scheduler_args_list.append({'look_up_table': [(50, 5e-3), (80, 5e-4), (100, 5e-5)]})

else:
    raise NotImplementedError()

args.optimizer_infos = [{'type': optimizer_type_list[idx], 'args': optimizer_args_list[idx]}
                        for idx in range(len(args.param_types))]
args.scheduler_infos = [{'type': scheduler_type_list[idx], 'args': scheduler_args_list[idx]}
                        for idx in range(len(args.param_types))]

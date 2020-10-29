import sys
from types import SimpleNamespace


args = SimpleNamespace()
args.exp_name = 'ImageNet50_5Tasks'
args.zero_init_expand = True
args.rng_seed = 0xCAFFE
args.save_opt = 'last'
args.workers  = 4

args.task_idx       = int(sys.argv[2])
args.dataset        = sys.argv[3]
args.backbone       = sys.argv[4]      if len(sys.argv) >= 5 else 'ResNet18'
args.final_task_idx = int(sys.argv[5]) if len(sys.argv) >= 6 else 5

args.image_size = 32
args.batch_size = 64
args.padding = 4
args.transform_name = 'standard_imagenet50'
args.num_classes = 10


optimizer_type_list = []
optimizer_args_list = []
scheduler_type_list = []
scheduler_args_list = []


args.max_epoch, args.param_types = 100, ['routing', 'other']

# Param type: routing
optimizer_type_list.append('Adam')
optimizer_args_list.append({'lr': 5e-3})
scheduler_type_list.append('LookUpTableLR')
scheduler_args_list.append({'look_up_table': [(50, 5e-3), (75, 5e-4), (100, 5e-5)]})

# Param type: other
optimizer_type_list.append('MaskedSGD')
optimizer_args_list.append({'lr': 5e-2, 'weight_decay': 5e-4, 'momentum': 0.9, 'nesterov': True})
scheduler_type_list.append('LookUpTableLR')
scheduler_args_list.append({'look_up_table': [(50, 5e-2), (75, 5e-3), (100, 5e-4)]})


args.optimizer_infos = [{'type': optimizer_type_list[idx], 'args': optimizer_args_list[idx]}
                        for idx in range(len(args.param_types))]
args.scheduler_infos = [{'type': scheduler_type_list[idx], 'args': scheduler_args_list[idx]}
                        for idx in range(len(args.param_types))]

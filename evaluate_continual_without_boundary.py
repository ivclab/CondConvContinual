import sys
import os
import importlib
import json
import numpy as np
import torch
import torch.nn as nn
from core.utils import load_checkpoint, find_task_dir_by_idx
from core.utils import build_imagedataloaders, build_backbone_info


def main(*args, **kwargs):

    # ---------------------------------
    # Loading the config
    # ---------------------------------
    config_module = importlib.import_module('configs.'+sys.argv[1])
    args = config_module.args
    print(args)

    # ---------------------------------
    # General settings
    # ---------------------------------
    device = 'cuda'
    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)
    np.random.seed(args.rng_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---------------------------------
    # Dataset settings
    # ---------------------------------
    image_size = args.image_size
    batch_size = args.batch_size
    padding = args.padding
    transform_name = args.transform_name

    # ---------------------------------
    # backbone settings
    # ---------------------------------
    backbone_info = build_backbone_info(args.backbone, 'cond', image_size)

    # ---------------------------------
    # Method settings
    # ---------------------------------
    experiment_dir = 'CHECKPOINTS/Continual/{}/{}'.format(
            args.exp_name, args.backbone)
    output_path = 'CHECKPOINTS/Continual/{}/{}/RESULTS_WITHOUT_BOUNDARY.json'.format(
            args.exp_name, args.backbone)

    # ---------------------------------
    # Run evaluation
    # ---------------------------------
    task_dir = find_task_dir_by_idx(experiment_dir, args.final_task_idx)
    chkpt_dir = os.path.join(experiment_dir, task_dir, 'finetune')
    model, manager = load_checkpoint(chkpt_dir)

    # ---------------------------------
    # Random initialization strategy
    # ---------------------------------
    task_dirs = filter(lambda x: x.split('_', 1)[0][:4] == 'Task', os.listdir(experiment_dir))
    task_dirs = sorted(list(task_dirs), key=lambda x: int(x.split('_', 1)[0][4:]))
    num_tasks = len(task_dirs)
    num_total_classes = args.num_classes * num_tasks
    task_class_ids = np.split(np.arange(num_total_classes), num_tasks)

    for index, task_class_idx in enumerate(task_class_ids):
        manager.load_task_exclusive_params(model, index+1)
        org_cls_state_dict = model.classifier.state_dict()
        model.build_classification_head(num_total_classes)
        new_cls_state_dict = model.classifier.state_dict()

        for name, org_param in org_cls_state_dict.items():
            new_param = new_cls_state_dict[name]
            cls_loc = torch.from_numpy(task_class_idx).long()
            new_param.index_copy_(0, cls_loc, org_param)

        manager.save_task_exclusive_params(model, index+1)

    # ---------------------------------
    # Run evaluation without boundary
    # ---------------------------------
    task_accs, rough_accs = [], []
    total_corrects = 0
    total_examples = 0
    for dataset_idx, task_dir in enumerate(task_dirs):
        dataset = task_dir.split('_', 1)[1]
        print('Current Dataset: {}'.format(dataset))

        test_loader = build_imagedataloaders(
                'evaluate', os.path.join(args.exp_name, dataset), transform_name,
                image_size, batch_size, padding, args.save_opt, args.workers)
        test_iter = test_loader()
        num_iters = len(test_loader)

        with torch.no_grad():

            # Inference using all tasks
            task_output_list = []
            task_labels_list = []
            for task_idx in range(1, num_tasks+1):
                manager.load_task_exclusive_params(model, task_idx)
                model.to(device)
                model.eval()

                output_list = []
                labels_list = []
                for batch_idx, batch_data in enumerate(test_iter):
                    sys.stdout.write('Task {}: {}/{}   ..... \r'.format(
                        task_idx, batch_idx+1, num_iters))
                    sys.stdout.flush()
                    images, labels = batch_data
                    images = images.to(device)
                    labels = labels.to(device) + dataset_idx * args.num_classes
                    output = model(images)
                    output_list.append(output.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())

                task_output_list.append(np.concatenate(output_list, 0))
                task_labels_list.append(np.concatenate(labels_list, 0))
                print()

            # Decide final predictions
            argmax_probs = np.argmax(np.concatenate(task_output_list, 1), 1)
            num_rough = np.sum((argmax_probs // num_total_classes) == dataset_idx)
            predis = argmax_probs % num_total_classes
            labels = task_labels_list[-1]
            num_corrects = np.sum(predis == labels)
            num_examples = labels.shape[0]
            task_accs.append(num_corrects / num_examples)
            rough_accs.append(num_rough / num_examples)
            total_corrects += num_corrects
            total_examples += num_examples

    content = {}
    for index, task_acc in enumerate(task_accs):
        print('Task {} Acc: {:.4f}, ({:.4f})'.format(
            index+1, task_acc, rough_accs[index]))
    content['Task_Acc']  = [round(x, 2) for x in task_accs]
    content['Rough_Acc'] = [round(x, 2) for x in rough_accs]

    final_acc = total_corrects / total_examples
    print('Final Acc: {:.4f}'.format(final_acc))
    content['Final_Acc'] = round(final_acc, 2)

    with open(output_path, 'w') as f:
        json.dump(content, f)
    return


if __name__ == '__main__':
    main()

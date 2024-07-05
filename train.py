import argparse
import os
import sys
import ruamel_yaml as yaml
import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist

from models.model_vqa import MUMC_VQA
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from data.utils import save_result
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from utils import cosine_lr_schedule

from vqaEvaluate import compute_vqa_acc

import torch._dynamo
torch._dynamo.config.suppress_errors = True
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# from vqaEvaluate import compute_vqa_acc

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    for i, (image, question, answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(torch.bfloat16).to(device, non_blocking=True)
        # print(image.shape)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question, answer, train=True, alpha=alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(torch.bfloat16).to(device, non_blocking=True)
        topk_ids, topk_probs = model(image, question, answer_list, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = ques_id
            _, pred = topk_prob.max(dim=0)
            result.append({"ground_truth": ques_id, "model_answer": data_loader.dataset.answer_list[topk_id[pred]]})
    return result


def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())

    #### Loading Dataset ####
    print('Creating vqa datasets')
    data_config = yaml.safe_load(open("./configs/dataset.yaml", 'r'))
    datasets = create_dataset(data_config, 'vqa_rad')
    print('train dataset size: ', len(datasets[0]))
    print('test dataset size: ', len(datasets[1]))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Creating Model ####
    print("Creating model")
    model = MUMC_VQA(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(torch.bfloat16).to(device)
    # print(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        # state_dict = checkpoint

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                             model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                if 'text_encoder' in key:
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < 6:
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num - 6)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    state_dict[decoder_key] = state_dict[key]

                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    start_epoch = 0
    print("\nStart training\n")
    start_time = time.time()

    if (args.evaluate):
        print("Testing ...")
        vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
        result_file = save_result(vqa_result, args.result_dir, 'vqa_result')
        return

    for epoch in range(start_epoch, config['max_epoch']):
        # if not args.evaluate:
        #     if args.distributed:
        #         train_loader.sampler.set_epoch(epoch)

        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

        train_stats = train(model, train_loader, optimizer, epoch, device, config)

        # if utils.is_main_process():

        save_obj = {
            'model': model_without_ddp.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'config': config,
            # 'epoch': epoch,
        }
        prefix = args.checkpoint.split('/')[-1].split('.')[0]
        # if args.is_save_path and epoch > 20:
        torch.save(save_obj, os.path.join(args.output_dir, '%s_rad_%02d.pth' % (prefix, epoch)))
            # vqa_result = evaluation(model, test_loader, device, config)
            # json.dump(vqa_result, open(os.path.join(args.result_dir, '%s_vqa_result_%s.json' % (prefix, epoch)), 'w'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        with open(os.path.join(args.output_dir, "loss.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # compute acc
    # res_file_path = '%s/result/%s_vqa_result_<epoch>.json' % (args.output_dir, prefix)
    # compute_vqa_acc(answer_list_path=config[args.dataset_use]['test_file'][0], epoch=config['max_epoch'], res_file_path=res_file_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='rad', help='choose medical vqa dataset(rad, pathvqa, slake)')
    parser.add_argument('--is_save_path', default=True)
    parser.add_argument('--checkpoint', default='./output/vqa/rad/rad_19_1/med_pretrain_19_rad_v1.pth')
    parser.add_argument('--output_suffix', default='/rad_19_1', help='output suffix, eg. ../rad_29_1')
    parser.add_argument('--output_dir', default='./output', help='the final output path, need not to assign')
    parser.add_argument('--evaluate', default=True)
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)

    args = parser.parse_args()

    args.output_dir = './output/vqa/' + args.dataset_use + args.output_suffix

    config = yaml.load(open('./configs/VQA.yaml', 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # set log, set console print info to file
    sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)

    print("config: ", config)
    print("args: ", args)
    print("running")
    main(args, config)
import time
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from exposer.models.opt_peft_adapter import OPTForCausalLM
from exposer.models.gpt_peft_adapter import GPT2LMHeadModel

from exposer.utils.config_utils import get_opt_adapter_config
from exposer.utils.config_utils import get_gpt2_adapter_config

from exposer.utils.data_utils import FT_Dataset
from exposer.utils.peft_utils import mark_only_adapter_as_trainable


parser = argparse.ArgumentParser(description='Attn block sparse')
parser.add_argument('--model_name', type=str, default='facebook/opt-125m', help='model name')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--init_checkpoint', type=str, help='initial checkpoint')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict')
parser.add_argument('--adapter_dim', type=int, default=64, help='adapter dimension')


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = get_gpt2_adapter_config(args.model_name, args.adapter_dim)
    n_layers, n_heads = config.num_hidden_layers, config.num_attention_heads

    model = GPT2LMHeadModel(config)
    # if args.init_checkpoint is not None:
    #     state_dict = torch.load(args.init_checkpoint, map_location='cpu')
    #     state_dict = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
    #     for k, v in state_dict.items():
    #         k = 'model.' + k  if args.model_name == 'facebook/opt-350m' else k  # for opt-350m
    #         if k in model.state_dict():
    #             model.state_dict()[k].copy_(v)
    #         else:
    #             print('skip', k)
    model.to(args.device)

    from torch.optim import AdamW, lr_scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 4000, 1.0))

    if args.adapter_dim > 0:
        mark_only_adapter_as_trainable(model)

    valid_data = FT_Dataset(args.data, args.batch_size, args.seq_len)
    valid_loader = DataLoader(valid_data, 
                              batch_size=args.batch_size, 
                              num_workers=0, 
                              shuffle=False, 
                              pin_memory=True, 
                              drop_last=False)
    
    model.eval()
    warmup = 10
    repitition = 50
    total_time = 0
    forward_time = 0
    backward_time = 0
    optimizer_time = 0
    scheduler_time = 0
    zero_grad_time = 0
    
    #Get timestamps of each step
    all_start = torch.cuda.Event(enable_timing=True)
    all_end = torch.cuda.Event(enable_timing=True)
    forward_step = torch.cuda.Event(enable_timing=True)
    backward_step = torch.cuda.Event(enable_timing=True)
    optimizer_step = torch.cuda.Event(enable_timing=True)
    scheduler_step = torch.cuda.Event(enable_timing=True)
    
    
    for batch_idx, data in enumerate(valid_loader):
        all_start.record()

        with torch.autocast(device_type=args.device, dtype=torch.float16):
            data = {key: value for key, value in data.items()}
            _input = data['input'].to(args.device)
            _target = data['target'].to(args.device)
            _msk = data['mask'].to(args.device)
            _batch, _len = _input.shape
            _lm_logits = model(_input).logits

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none', label_smoothing=0.1)
            loss = loss_fct(_lm_logits.view(-1, _lm_logits.size(-1)), _target.view(-1)).view(_batch, _len)
            loss = loss * _msk
            loss = loss.sum() / (_msk.sum() + 0.0001)
            # print('loss:', loss)
        forward_step.record()

        #Backward Pass (compute gradients)
        loss.backward()
        backward_step.record()

        #Optimizer Pass (update gradients)
        optimizer.step()
        optimizer_step.record()

        #Scheduler step (update learning rate)
        scheduler.step()
        scheduler_step.record()

        #Set gradients back to zero
        model.zero_grad()
        
        all_end.record()
        torch.cuda.synchronize()

        if batch_idx >= warmup + repitition:
            break
        if batch_idx >= warmup:
            total_time += all_start.elapsed_time(all_end)
            forward_time += all_start.elapsed_time(forward_step)
            backward_time += forward_step.elapsed_time(backward_step)
            optimizer_time += backward_step.elapsed_time(optimizer_step)
            scheduler_time += optimizer_step.elapsed_time(scheduler_step)
            zero_grad_time += scheduler_step.elapsed_time(all_end)

    print(f"""--average time: {total_time / repitition} ms, 
            --calculated average time: {(forward_time + backward_time + optimizer_time + scheduler_time + zero_grad_time)/ repitition} 
            --forward time: {forward_time / repitition} ms,
            --backward time: {backward_time / repitition} ms,
            --optimizer time: {optimizer_time / repitition} ms,
            --scheduler time: {scheduler_time / repitition} ms,
            --zero_grad time: {zero_grad_time / repitition} ms,
            --file_name: {os.path.basename(__file__)}, 
            --model: {args.model_name}, 
            --batch size: {args.batch_size}, 
            --sequence length: {args.seq_len}, 
            --data: {args.data}, 
            --device: {args.device}""")
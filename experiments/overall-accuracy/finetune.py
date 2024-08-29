import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from exposer.utils.config_utils import get_opt_exposer_lora_attn_config
from exposer.utils.peft_utils import mark_only_lora_as_trainable
from exposer.models import opt_exposer_lora_attn, opt_peft_lora


prompt_template_e2e = '''
### Instruction:
Write a restaurant description according to provided information: "{description}"

### Response:
{response}
'''


def format_prompts_e2e(example):
    output_texts = []
    for description, response in zip(example['meaning_representation'], example['human_reference']):
        text = prompt_template_e2e.format(
            description=description, response=response)
        output_texts.append(text)
    return output_texts


prompt_template_winogrande = '''
### Instruction:
Commonsense reasoning: complete the following sentence.

### Response:
{sentence}
'''


def format_prompts_winogrande(cols):
    output_texts = []
    for sentence, option1, option2, answer in zip(cols['sentence'], cols['option1'], cols['option2'], cols['answer']):
        option = option1 if answer == '1' else option2
        sentence = sentence.replace('_', option)
        text = prompt_template_winogrande.format(sentence=sentence)
        output_texts.append(text)
    return output_texts


def format_prompts_winogrande_v2(cols):
    output_texts = []
    for sentence, option1, option2, answer in zip(cols['sentence'], cols['option1'], cols['option2'], cols['answer']):
        option = option1 if answer == '1' else option2
        text = sentence + ' @ ' + option
        output_texts.append(text)
    return output_texts


def format_prompts_alpaca(cols):
    output_texts = []
    for text in cols['text']:
        output_texts.append(text)
    return output_texts


def load_initial_weights(model, ref_model):
    state_dict = model.state_dict()
    ref_state_dict = ref_model.state_dict()
    for k, v in ref_state_dict.items():
        if k in state_dict:
            state_dict[k].copy_(v)
        else:
            print('skip', k)


def finetune(model_name, device, dataset_name, is_baseline=False, n_epochs=5, batch_size=4):
    if dataset_name == 'e2e':
        dataset = load_dataset('e2e_nlg')
        format_prompt = format_prompts_e2e
    elif dataset_name == 'winogrande':
        dataset = load_dataset('winogrande', 'winogrande_debiased')
        format_prompt = format_prompts_winogrande
    elif dataset_name == 'alpaca':
        #Download the dataset from HuggingFace website
        dataset = load_dataset('tatsu-lab/alpaca')
        #Loads the prompts and answers into a list
        format_prompt = format_prompts_alpaca
    else:
        raise NotImplementedError

    #Loads in the configs for a model's class in the CPU
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu')
    #Loads correct tokenizer class according to model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sparse_pattern = 'bigbird'
    #Sets configurations for attention and scarsity
    config = get_opt_exposer_lora_attn_config(model_name, sparse_config=sparse_pattern)

    #Creates model with or without sparsity attention
    if is_baseline:
        model = opt_peft_lora.OPTForCausalLM(config)
    else:
        model = opt_exposer_lora_attn.OPTForCausalLM(config)

    #Marks certain LORA parameters to be trainable in model
    mark_only_lora_as_trainable(model, 'lora_only')
    if hasattr(config, 'sparsity_config'):
        del config.sparsity_config  # remove this since SparsityConfig is not JSON serializable

    #Copies values for each key that is in ref_model and model
    load_initial_weights(model, ref_model)

    #Offload model to GPU
    model.to(device)

    model_name = model_name.split('/')[-1]

    #Create collator for grabbing prompts
    response_template = '### Response:'
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, pad_to_multiple_of=16)

    output_dir_suffix = f'-{dataset_name}'
    if is_baseline:
        output_dir_suffix += '-base'
    else:
        output_dir_suffix += f'-{sparse_pattern}'

    #Set training parameters (HuggingFace)
    training_args = TrainingArguments(
        output_dir=model_name + output_dir_suffix,
        num_train_epochs=n_epochs,
        # evaluation_strategy='steps',
        # eval_steps=250,
        save_strategy='epoch',
        bf16=True,  # fp16 may cause loss always zero for OPT-1.3B and above
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir='logs',
        logging_steps=100,
    )

    #Create trainer class (TRL)
    trainer = SFTTrainer(
        model,
        # peft_config=peft_config,
        args=training_args,
        train_dataset=dataset['train'],
        # eval_dataset=dataset['validation'],
        formatting_func=format_prompt,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=1024,
    )

    #Set the module to training mode
    trainer.train()


if __name__ == '__main__':
    torch.manual_seed(42)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='facebook/opt-350m', choices=['facebook/opt-350m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--base', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='alpaca', choices=['e2e', 'winogrande', 'alpaca'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=6)
    args = parser.parse_args()

    finetune(args.model_name, args.device, args.dataset, args.base, args.epochs, args.batch_size)
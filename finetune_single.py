#? This script is used to fine-tune a model on the CodeComplexity dataset
import argparse
import torch
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
from datasets import ClassLabel, DatasetDict, load_dataset, Dataset
from evaluate import load
from math import ceil

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

from TransformerLibrary.src.transformers import ( 
    AutoModelForSequenceClassification, #For loading model
    AutoTokenizer, #For tokenizing data
    DataCollatorWithPadding, #For padding batches
    Trainer, #For training
    TrainerCallback, #For callbacks
    TrainingArguments, #For training arguments
    set_seed,
)

#Parse arguments from command line or default
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="openai-community/gpt2")
    parser.add_argument("--num_epochs", type=int, default=5) 
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--output_dir", type=str, default="./Results_Single") 
    
    #New arguments
    parser.add_argument("--group_by_length", action="store_true", help="Turn on group_by_length") 
    parser.add_argument('--no_shuffle', action="store_true", help="Turn on smartbatching")
    parser.add_argument('--worst_case', action="store_true", help="Use worst case group_by_length scenario")
    return parser.parse_args()

#Used to compute accuracy of the model
metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    args = get_args()
    set_seed(21) 

    #Load and split dataset into training, testing, and validation 
    dataset = load_dataset("codeparrot/codecomplex", split="train")
    train_test = dataset.train_test_split(test_size=0.2)
    test_validation = train_test["test"].train_test_split(test_size=0.5)

    train_test_validation = DatasetDict(
        {
            "train": train_test["train"],
            "test": test_validation["train"],
            "valid": test_validation["test"],
        }
    )

    #Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token #?
    tokenizer.pad_token_id = tokenizer.eos_token_id #?
    model = AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=7)
    model.config.pad_token_id = tokenizer.pad_token_id #?

    #Offload model to GPU
    model.to('cuda:0')

    #Create labels for training
    labels = ClassLabel(num_classes=7, names=list(set(train_test_validation["train"]["complexity"])))

    #Tokenize data (should stay the same)
    def tokenize(example):
        inputs = tokenizer(example["src"], truncation=True)
        label = labels.str2int(example["complexity"])
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": label,
        }

    tokenized_datasets = train_test_validation.map(
        tokenize,
        batched=True,
        remove_columns=train_test_validation["train"].column_names,
    )

    #Create worst case dataset for group_by_length
    if args.worst_case == True:
        tokenized_datasets["train"] = tokenized_datasets["train"].map(lambda x: {"input_id_length": len(x["input_ids"])})
        tokenized_datasets["train"] = tokenized_datasets["train"].sort("input_id_length", reverse=True)

        mega_batch_mult = min(train_test["train"].num_rows // (args.batch_size * 4), 50) #floor
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1
        megabatch_size = mega_batch_mult * args.batch_size

        # print("Before")
        # print(tokenized_datasets["train"]["input_id_length"][:megabatch_size + 1])
        # print(tokenized_datasets["train"]["input_id_length"][megabatch_size: megabatch_size * 2 + 1])
        # print(tokenized_datasets["train"]["input_id_length"][::-1])

        counter = 0
        new_train = []
        for i in range(0, ceil(tokenized_datasets["train"].num_rows / megabatch_size)):
            if counter + megabatch_size > tokenized_datasets["train"].num_rows:
                for j in range(0, tokenized_datasets["train"].num_rows - counter):
                    new_train.append(tokenized_datasets["train"][i + j])
                break
            else:
                new_train.append(tokenized_datasets["train"][i])
                for j in range(1, megabatch_size):
                    new_train.append(tokenized_datasets["train"][(i * (megabatch_size - 1) + j) * -1])
            counter += megabatch_size

        tokenized_datasets["train"] = Dataset.from_list(new_train)

        # print("After")
        # print(tokenized_datasets["train"]["input_id_length"][:megabatch_size + 1])
        # print(tokenized_datasets["train"]["input_id_length"][megabatch_size: megabatch_size * 2 + 1])

        tokenized_datasets["train"] = tokenized_datasets["train"].remove_columns("input_id_length")
        
    #This is where dynamic batching happens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #We want TrainingArguments and Deepspeed arguments to match
    training_args = TrainingArguments(
        #optim="adamw_torch",
        output_dir=args.output_dir,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        eval_strategy="no", #"epoch"
        save_strategy="no", 
        logging_strategy="no",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        torch_compile=True,
        fp16=True,
        report_to="none",
        metric_for_best_model="accuracy",
        deepspeed="/people/hoan163/SmartBatch/ds_config.json", #"/scratch/user/u.ah287219/Project2/ds_config.json"
        group_by_length=args.group_by_length,
        no_shuffle_group_by_length=args.no_shuffle, #! New Parameter
        do_train=False #Set this to True to output trained model
    )

    #Create trainer that handles training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    #Custom Callbacks
    class CustomCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer

        #Function only used if evaluation is done at the end of each epoch
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate: 
                print("\nBegin evaluation for current epoch")
            
        def on_train_begin(self, args, state, control, **kwargs):
            print("Training Begins")
        
        def on_train_end(self, args, state, control, **kwargs):
            print("Training Complete")
            # print("\nFinal Test")
            # print(self._trainer.evaluate(eval_dataset=tokenized_datasets["test"]))

        #After evaluation phase
        def on_evaluate(self, args, state, control, **kwargs):
            print("Evaluation complete")

    trainer.add_callback(CustomCallback(trainer))
    print("Model: ", args.model_ckpt)
    print("Group_By_Length: ", args.group_by_length)
    print("Smart_Batch: ", args.no_shuffle)
    print("Worst Case: ", args.worst_case)

    trainer.train() #Train model

    total_runtime = trainer.state.log_history[0]['train_runtime']
    return total_runtime

if __name__ == "__main__":
    runtime = main()
    # print("Runtime: " + str(runtime))

'''
export PATH="/scratch/user/u.ah287219/.conda/envs/SmartBatch-env/bin:$PATH"
export PYTHONPATH="/scratch/user/u.ah287219/.conda/envs/SmartBatch-env/lib/python3.11/site-packages:$PYTHONPATH"

* Split dataset 80/20 for train/test
* Sort dataset from largest to shortest in the train section
* For every megabatch, put one long sequence and fill the rest with short sequence
* Pass in new dataset and measure zero padding
'''
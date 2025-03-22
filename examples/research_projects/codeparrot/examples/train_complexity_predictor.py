import argparse
from copy import deepcopy
import torch
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
from datasets import ClassLabel, DatasetDict, load_dataset
from evaluate import load

from transformers.src.transformers import ( #!Changed import transformers to transformers.src.transformers
    AutoModelForSequenceClassification, #For loading model
    AutoTokenizer, #For tokenizing data
    DataCollatorWithPadding, #For padding batches
    Trainer, #For training
    TrainerCallback, #For callbacks
    TrainingArguments, #For training alterations
    set_seed,
)

#Parse arguments from command line or default
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default="codeparrot/codeparrot-small")
    parser.add_argument("--num_epochs", type=int, default=2) #
    parser.add_argument("--batch_size", type=int, default=8) #
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1) #
    parser.add_argument("--freeze", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-4) #
    parser.add_argument("--seed", type=int, default=0) #
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine") #
    parser.add_argument("--num_warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01) #
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

#Used to compute accuracy of the model
metric = load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    #Function only used if evaluation is done at the end of each epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate: #?If evaluation is done at the end of each epoch
            # control_copy = deepcopy(control)

            # #?This line of code does evaluation on the training dataset
            # self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train") 
            # return control_copy
            print('\033[96m' + "\nBegin evaluation and logging for current epoch\n" + '\033[0m')
        
    def on_train_begin(self, args, state, control, **kwargs):
        print('\033[92m' + "\nTraining for sortedTransformers Begins\n" + '\033[0m')
        #return super().on_train_begin(args, state, control, **kwargs)
    
    def on_train_end(self, args, state, control, **kwargs):
        print('\033[92m' + "\nTraining for sortedTransformers Complete\n" + '\033[0m')
        # if control.should_evaluate:
        #     control_copy = deepcopy(control)
        #     self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
        #     return control_copy

    #After evaluation phase
    def on_evaluate(self, args, state, control, **kwargs):
        print('\033[96m' + "\nEvaluation for current epoch complete\n" + '\033[0m')
        #return super().on_evaluate(args, state, control, **kwargs)


def main():
    args = get_args()
    set_seed(args.seed)

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
    print("\nLoading tokenizer and model\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=7)
    model.config.pad_token_id = model.config.eos_token_id

    #Offload model to GPU
    model.to(args.device)

    #If model is based of RoBERTa we can freeze parameters
    # if args.freeze:
    #     for param in model.roberta.parameters():
    #         param.requires_grad = False

    #Create labels for training
    labels = ClassLabel(num_classes=7, names=list(set(train_test_validation["train"]["complexity"])))

    #Tokenize data
    def tokenize(example):
        inputs = tokenizer(example["src"], truncation=True, max_length=1024)
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

    #This is where dynamic batching happens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #We want TrainingArguments and Deepspeed arguments to match
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        eval_strategy="no",
        #eval_steps=0.5,
        save_strategy="no", #? You need to set this to "epoch" to output the model
        logging_strategy="no",
        #logging_steps=0.5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        metric_for_best_model="accuracy",
        run_name="complexity-java",
        report_to="none",
        deepspeed="/people/hoan163/project/ZeroPadding/ds_config.json",
        group_by_length=True,
        do_train=False #? Set this to True to actually train the model
    )

    #Create trainer that handles training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    trainer.add_callback(CustomCallback(trainer))

    start.record()
    #Train model
    trainer.train()
    end.record()
    
    print('\033[92m' + f"Total Runtime: {str(start.elapsed_time(end)/1000)} seconds\n" + '\033[0m')


    #* Extract the total runtime of inner_training_loop from the metrics
    # metrics = trainer.state.log_history['train_runtime]  # Get the latest log entry
    # total_runtime = metrics.get("train_runtime", None)
    # print(f"Total runtime of inner_training_loop: {total_runtime} seconds")

#! Working Version
if __name__ == "__main__":
    main()

#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
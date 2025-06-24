#? This script is used to fine-tune a model on the CodeComplexity dataset
import argparse
import torch
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
from datasets import ClassLabel, DatasetDict, load_dataset
from evaluate import load

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
    parser.add_argument("--learning_rate", type=float, default=2e-5) 
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine") 
    parser.add_argument("--output_dir", type=str, default="./Results_Multi") 
    parser.add_argument('--no_shuffle', action="store_true", help="Turn off shuffling and megabatches during group_by_length")
    parser.add_argument("--group_by_length", action="store_true", help="Turn on group_by_length") 
    parser.add_argument("--local_rank", type=int, default=-1)  # Add this line
    return parser.parse_args()

#Used to compute accuracy of the model
metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    args = get_args()
    set_seed(0) #? Consider changing

    #Load and split dataset into training, testing, and validation to cache
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
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=7)
    model.config.pad_token_id = model.config.eos_token_id

    #Offload model to GPU
    model.to('cuda:0')

    #Create labels for training
    labels = ClassLabel(num_classes=7, names=list(set(train_test_validation["train"]["complexity"])))

    #Tokenize data (should stay the same)
    def tokenize(example):
        inputs = tokenizer(example["src"], truncation=True) #, max_length=1024
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

    #TODO: Fix multi-GPU training
    #TODO: Random print statements
    #TODO: Every GPU runs the same number of epochs
    #TODO: Without num_gpus, it will run every GPU on system
    #TODO: What is DistributedLenghthGroupedSampler? How is it called? What does it do? Does it affect our SmartBatching?

    #This is where dynamic batching happens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #We want TrainingArguments and Deepspeed arguments to match
    training_args = TrainingArguments(
        #optim="adamw_torch",
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
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
        metric_for_best_model="accuracy",
        run_name="complexity-java",
        report_to="none",
        deepspeed="/people/hoan163/project/ds_multi_config.json",
        group_by_length=args.group_by_length,
        no_shuffle_group_by_length=args.no_shuffle, #! New Parameter
        do_train=False #? Set this to True to actually train the model
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

    # start = torch.cuda.Event(enable_timing=True) #!Remove this or no?
    # end = torch.cuda.Event(enable_timing=True)

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
            print("\nTraining Begins\n")
            # start.record()
        
        def on_train_end(self, args, state, control, **kwargs):
            # end.record()
            print("\nTraining Complete")
            # print(f"Total Runtime: {str(start.elapsed_time(end)/1000)} seconds\n")

            print("Final Test")
            print(self._trainer.evaluate(eval_dataset=tokenized_datasets["test"]))

        #After evaluation phase
        def on_evaluate(self, args, state, control, **kwargs):
            print("Evaluation complete")

    trainer.add_callback(CustomCallback(trainer))
    print("Model: ", args.model_ckpt)
    print("Group_By_Length: ", args.group_by_length)
    print("Smart_Batch: ", args.no_shuffle)

    trainer.train() #Train model

    #? Extract the total runtime of inner_training_loop from the metrics
    total_runtime = trainer.state.log_history[0]['train_runtime']
    return total_runtime

if __name__ == "__main__":
    runtime = main()
    print("Runtime: " + str(runtime))


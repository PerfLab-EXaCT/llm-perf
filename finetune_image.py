#? This script is used to fine-tune a model on the CodeComplexity dataset
import argparse
import torch
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
from datasets import load_dataset
from evaluate import load

from TransformerLibrary.src.transformers import ( 
    ViTForImageClassification, #For loading model
    DataCollatorWithPadding, #For padding batches
    Trainer, #For training
    TrainerCallback, #For callbacks
    TrainingArguments, #For training arguments
    ViTImageProcessor, #For image processing
)

#Parse arguments from command line or default
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument("--num_epochs", type=int, default=1) 
    parser.add_argument("--batch_size", type=int, default=8) 
    parser.add_argument("--learning_rate", type=float, default=2e-5) 
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine") #? linear?
    parser.add_argument("--output_dir", type=str, default=".") 
    parser.add_argument("--group_by_length", action="store_true", help="Turn on group_by_length") 
    parser.add_argument('--no_shuffle', action="store_true", help="Turn on smartbatching")
    return parser.parse_args()

#Used to compute accuracy of the model
metric = load("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
#         'labels': torch.tensor([x['labels'] for x in batch])
#     }

def main():
    args = get_args()

    #load dataset
    ds = load_dataset('nateraw/beans')
    #load labels
    labels = ds['train'].features['labels'].names
    #load processor (tokenizer)
    processor = ViTImageProcessor.from_pretrained(args.model_ckpt)

    #Pass image here
    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = processor([x for x in example_batch['image']], return_tensors='pt')

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['labels']
        return inputs

    prepared_ds = ds.with_transform(transform)

    model = ViTForImageClassification.from_pretrained(args.model_ckpt, 
                                                      num_labels=len(labels),
                            id2label={str(i): c for i, c in enumerate(labels)},
                            label2id={c: str(i) for i, c in enumerate(labels)})

    processor.pad_token = processor.eos_token #?
    processor.pad_token_id = processor.eos_token_id #?
    model.config.pad_token_id = processor.pad_token_id #?

    #Offload model to GPU
    model.to('cuda:0')

    #Tokenize data (should stay the same)
    # def tokenize(example):
    #     inputs = tokenizer(example["src"], truncation=True)
    #     label = labels.str2int(example["complexity"])
    #     return {
    #         "input_ids": inputs["input_ids"],
    #         "attention_mask": inputs["attention_mask"],
    #         "label": label,
    #     }

    #This is where dynamic batching happens
    data_collator = DataCollatorWithPadding(tokenizer=processor)

    #We want TrainingArguments and Deepspeed arguments to match
    training_args = TrainingArguments(
        #optim="adamw_torch",
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        eval_strategy="no", #"epoch"
        save_strategy="no", 
        logging_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        torch_compile=True,
        fp16=False, #True
        metric_for_best_model="accuracy",
        run_name="complexity-java",
        report_to="none",
        deepspeed="/people/hoan163/SmartBatch/ds_config.json", #"/scratch/user/u.ah287219/Project2/ds_config.json"
        remove_unused_columns=False, 
        group_by_length=args.group_by_length,
        no_shuffle_group_by_length=args.no_shuffle, #! New Parameter
        do_train=False #? Set this to True to output trained model
    )

    #Create trainer that handles training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        compute_metrics=compute_metrics,        
        processing_class=processor,
        data_collator=data_collator,
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
            print("\nTraining Begins\n")
        
        def on_train_end(self, args, state, control, **kwargs):
            print("\nTraining Complete")
            print("\nFinal Test")
            print(self._trainer.evaluate(eval_dataset=prepared_ds["test"]))

        #After evaluation phase
        def on_evaluate(self, args, state, control, **kwargs):
            print("Evaluation complete")

    trainer.add_callback(CustomCallback(trainer))
    print("Model: ", args.model_ckpt)
    print("Batch size: ", args.batch_size)
    print("Group_By_Length: ", args.group_by_length)
    print("Smart_Batch: ", args.no_shuffle)

    trainer.train() #Train model

    #? Extract the total runtime of inner_training_loop from the metrics
    total_runtime = trainer.state.log_history[0]['train_runtime']
    return total_runtime

if __name__ == "__main__":
    runtime = main()
    # print("Runtime: " + str(runtime))


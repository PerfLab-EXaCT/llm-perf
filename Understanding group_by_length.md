`group_by_length` is used to group data based on length while ensuring a certain level of randomness within each group.

### How to Use

In `TrainingArguments`, set the attribute `group_by_length=True`. By default, it is set to `False`.

### Scope of Influence

When `group_by_length` is specified, the `_get_train_sampler` function utilizes the `LengthGroupedSampler`:

```python
def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    if self.train_dataset is None or not has_length(self.train_dataset):
        return None

    # Build the sampler.
    if self.args.group_by_length:
        if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
            lengths = (
                self.train_dataset[self.args.length_column_name]
                if self.args.length_column_name in self.train_dataset.column_names
                else None
            )
        else:
            lengths = None
        model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            dataset=self.train_dataset,
            lengths=lengths,
            model_input_name=model_input_name,
        )

    else:
        return RandomSampler(self.train_dataset)
```

Code for` LengthGroupedSampler`：

```python
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)

```

This is a PyTorch data sampler used to group data by length while maintaining randomness within groups. If the dataset explicitly defines a `length` attribute, it is used; otherwise, the length of `tokenizer.model_input_names[0]` is used as the grouping basis.

Notably, this feature is independent of Deepspeed and can be used with it.

### Understanding LLM Input

Taking Qwen2 as an example, the dataset format is as follows:

```json
{"type": "chatml", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is up?"}, {"role": "assistant", "content": "Hello! How can I help you today?"}]}
{"type": "chatml", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is up?"}, {"role": "assistant", "content": "Hello! How can I help you today?"}]}
```

A commonly used preprocessing function for this data is as follows:

```python
def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                chat_template=TEMPLATE,
                tokenize=True,
                add_generation_prompt=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )
```

Here, each `msg` is a data entry in the dataset, and `apply_chat_template` applies a template. The `TEMPLATE` used is as follows:

```python
TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
```

The whole conversation is used for training, not segments within it. After tokenization, the function outputs three tensors: `input_ids`, `target_ids`, and `attention_mask`, where:

- `input_ids` are the actual inputs.
- `target_ids` indicate tokens for which loss should be calculated.
- `attention_mask` is the attention mask.

### Using `group_by_length`

In `TrainingArguments`, set `group_by_length=True`, and modify the `preprocess` function to add a `length` attribute for each input’s token count. In `Trainer`'s `train_step` function, verify if the input meets the requirements.

Construct a dataset with both short- and long-token examples. During training, short and long token data will not form a single batch.

### Relationship with Dynamic Padding

From my understanding, `group_by_length` and dynamic padding are unrelated and can be used independently.

### Strictly Grouping by Length

To ensure no randomness, modify `LengthGroupedSampler` to sample indices strictly by length (from shortest to longest). Here’s the modified code:

```python
from torch.utils.data import Sampler
from typing import Optional, List
import torch

class LengthGroupedSampler(Sampler):
    """
    Sampler that strictly samples indices in a way that groups together features of the dataset
    of exactly the same length from smallest to largest without randomness.
    """

    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()

        self.lengths = lengths

        # sorts by length and saves index into list
        self.sorted_indices = sorted(range(len(lengths)), key=lambda idx: lengths[idx], reverse=True) 
        

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        for i in range(0, len(self.sorted_indices), self.batch_size):
            yield self.sorted_indices[i:i + self.batch_size]
        #This sends back each batch individually, without having to save the entire indice list into memory
        #Out of memory error? (not with python)

        #Test this class in experiment
```

This strict ordering is not recommended as it may impact convergence.
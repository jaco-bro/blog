## UnslothTrainer Gotcha: Keep All Columns

1. When using UnslothTrainer with a custom collator, you must set `remove_unused_columns=False`. Otherwise, fields like "attention_mask" can be dropped before batching, causing `KeyError` inside the collator.

2. If you are using pretokenized inputs (with input ids, labels, etc.), you must also set `dataset_text_field=None`. This prevents UnslothTrainer from trying to tokenize a raw text field that doesn't exist.

3. When working with tokenized examples, you can use either `Dataset.from_list()`: 
   ```python
   ds = Dataset.from_list([
       {
           "input_ids": [1, 2, 3],
           "attention_mask": [1, 1, 1],
           "labels": [-100, 1, 2]
       },
       {
           "input_ids": [4, 5],
           "attention_mask": [1, 1],
           "labels": [-100, 5]
       }
   ])
   ```
   Or `from_dict()`:
   ```python
   ds = Dataset.from_dict({
       "input_ids":      [[1, 2, 3], [4, 5]],
       "attention_mask": [[1, 1, 1], [1, 1]],
       "labels":         [[-100, 1, 2], [-100, 5]]
   })
   ```

4. After dataset construction, always call:
   ```python
   ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
   ```
   This ensures each field is returned as a `torch.Tensor`, which is necessary for DataLoader.
   ```python
   ds[:2]
   {'input_ids': [tensor([1, 2, 3]), tensor([4, 5])],
    'attention_mask': [tensor([1, 1, 1]), tensor([1, 1])],
    'labels': [tensor([-100,    1,    2]), tensor([-100,    5])]}
   ```

5. HuggingFace datasets support ragged sequences but do not pad or batch automatically. You need a collator to handle padding at load time (e.g., using `pad_sequence()`), unless you are passing raw text and letting the trainer tokenize and pad internally. 

6. If all sequences are the same length, indexing with `dataset[:N]` yields batched tensors automatically:
   ```python
   ds=Dataset.from_list([
       {
           "input_ids": [1, 2, 3],
           "attention_mask": [1, 1, 1],
           "labels": [-100, 1, 2]
       },
       {
           "input_ids": [4, 5, 6],
           "attention_mask": [1, 1, 0],
           "labels": [-100, 5, -100]
       }
   ])
   ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
   ds[:2]
   {'input_ids': tensor([[1, 2, 3],
            [4, 5, 6]]),
    'attention_mask': tensor([[1, 1, 1],
            [1, 1, 0]]),
    'labels': tensor([[-100,    1,    2],
            [-100,    5, -100]])}
   ```

<details><summary>Click to expand code</summary><pre>
train_data = Dataset.from_list(list_tokenized)
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

def collate_fn(batch):
    return {
        "input_ids": pad_sequence([e["input_ids"] for e in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence([e["attention_mask"] for e in batch], batch_first=True, padding_value=0),
        "labels": pad_sequence([e["labels"] for e in batch], batch_first=True, padding_value=-100),
    }

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field=None,
    train_dataset=train_data,
    data_collator=collate_fn,
    max_seq_length=max_seq_length_train,
    packing=True,
    args=UnslothTrainingArguments(
        output_dir=base_adapter_storage,
        per_device_train_batch_size=256,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=10,
        optim="adamw_8bit",
        save_strategy="no",
        report_to="none",
        seed=42,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        remove_unused_columns=False,
    )
)

class SaveBestLoraCallback(TrainerCallback):
    def __init__(self, model, save_path):
        self.model = model
        self.save_path = save_path
        self.best_loss = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        loss = logs.get("loss")
        if loss is not None and loss < self.best_loss:
            self.best_loss = loss
            torch.save(get_peft_model_state_dict(self.model), self.save_path)

adapter_path = os.path.join(base_adapter_storage, f"base_adapter_{gpu_id}.pth")
trainer.add_callback(SaveBestLoraCallback(model, adapter_path))
trainer.train()
</pre></details><br>

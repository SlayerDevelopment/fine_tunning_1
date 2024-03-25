import evaluate
import torch
import numpy as np
from datasets import load_dataset
from peft.tuners import LoraConfig
from peft.mapping import get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


def tokenizer_func(examples: dict):
    text = examples["text"]
    tokenizer.truncation_side = "left"
    tokenized_input = tokenizer(
        text=text, return_tensors="np", truncation=True, max_length=512
    )
    return tokenized_input


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}


if __name__ == "__main__":
    print("=" * 30, "PROCESS START")
    model_checkpoint = "distilbert-base-uncased"
    print("=" * 30, f"MODEL USED IS:{model_checkpoint}")
    id_label = {0: "negative", 1: "positive"}
    label_id = {"negative": 0, "positive": 1}
    print("=" * 30, "MODEL INIT")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2, id2label=id_label, label2id=label_id
    )
    print("=" * 30, "LOAD DATASET")
    dataset = load_dataset("shawhin/imdb-truncated")
    print("=" * 30, "TOKENIZER MODEL")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    tokenized_dataset = dataset.map(tokenizer_func, batched=True)
    print("=" * 30, "DATA COLLATOR")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("=" * 30, "EVALUATE ACCURATE METRIC SETTING")
    accuracy = evaluate.load("accuracy")
    print("=" * 30, "APPLIED MODEL WITHOUT TRAINING")
    text_list = [
        "It was good.",
        "Not a fan, don't recommed.",
        "Better than the first one.",
        "This is not worth watching even once.",
        "This one is a pass.",
    ]
    for text in text_list:
        # tokenize text
        inputs = tokenizer.encode(text, return_tensors="pt")
        # compute logits
        logits = model(inputs).logits
        # convert logits to label
        predictions = torch.argmax(logits)
        print(text + " - " + id_label[predictions.tolist()])
    print("=" * 30, "TRAINING MODEL")
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["q_lin"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    learning_rate = 1e-3
    batch_size = 4
    num_epochs = 10
    training_args = TrainingArguments(
        output_dir=model_checkpoint + "-lora-text-classification",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.to('cuda')
    print("=" * 30, "VALIDATE MODEL")
    for text in text_list:
        inputs = tokenizer.encode(text, return_tensors="pt").to('cuda')
        logits = model(inputs).logits
        predictions = torch.max(logits, 1).indices
        print(text + " - " + id_label[predictions.tolist()[0]])

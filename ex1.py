import os
import wandb
import numpy as np
import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding, AutoTokenizer,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on MRPC")
    # Sample limits
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    # Training config
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16)
    # Action flags
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true")
    # Model path
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    return parser.parse_args()

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=512,
    )

def train(args, tokenizer, datasets):
    # Tokenize train and validation sets
    encoded_train = datasets["train"].map(lambda x: preprocess_function(x, tokenizer), batched=True)
    encoded_val = datasets["validation"].map(lambda x: preprocess_function(x, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
    args_train = TrainingArguments(
        output_dir="./results",
        save_strategy="no",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.do_train:
        trainer.train()
    return trainer

def save_predictions(preds, test_dataset):
    fname = "predictions.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for s1, s2, p in zip(test_dataset["sentence1"], test_dataset["sentence2"], preds):
            f.write(f"{s1}###{s2}###{p}\n")
    print(f"Saved predictions to {fname}")

    # Log predictions to Weights & Biases
    table = wandb.Table(columns=["sentence1", "sentence2", "prediction"])
    for s1, s2, p in zip(test_dataset["sentence1"], test_dataset["sentence2"], preds):
        table.add_data(s1, s2, int(p))
    wandb.log({"predictions": table})

def prepare_datasets(args, preprocess_function):
    raw = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    train_dataset = raw["train"]
    val_dataset = raw["validation"]
    test_dataset = raw["test"]
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples > 0:
        val_dataset = val_dataset.select(range(args.max_eval_samples))
    if args.max_predict_samples > 0:
        test_dataset = test_dataset.select(range(args.max_predict_samples))
    encoded_test = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    return train_dataset, val_dataset, test_dataset, encoded_test, tokenizer

def main():
    wandb.init(project="mrpc-paraphrase")
    # -------------------- PREPARE DATA --------------------
    args = parse_args()
    train_dataset, val_dataset, test_dataset, encoded_test, tokenizer = prepare_datasets(args, preprocess_function)
    # -------------------- TRAIN --------------------
    trainer = train(args, tokenizer, {"train": train_dataset, "validation": val_dataset})
    # -------------------- PREDICT --------------------
    if args.do_predict:
        trainer.model.eval()
        predictions_output = trainer.predict(encoded_test)
        preds = np.argmax(predictions_output.predictions, axis=1)
        save_predictions(preds, test_dataset)
    wandb.finish()



if __name__ == "__main__":
    main()
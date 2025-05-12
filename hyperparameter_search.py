import json
import os
import wandb
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding, AutoTokenizer,
)
from sklearn.metrics import accuracy_score
from datasets import load_dataset


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


def hyperparameter_search():
    raw = load_dataset("glue", "mrpc")
    train_dataset = raw["train"]
    val_dataset = raw["validation"]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    base_dir = "./results"
    os.makedirs(base_dir, exist_ok=True)
    val_results = {}
    # Define hyperparameters
    params_list = [
        {"lr": 0.0001, "batch_size": 16, "epochs": 3},
        {"lr": 0.001, "batch_size": 16, "epochs": 3},
        {"lr": 0.00001, "batch_size": 32, "epochs": 3},
        {"lr": 0.00001, "batch_size": 32, "epochs": 5},
    ]
    # Tokenize train and validation sets
    encoded_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    encoded_val = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)
    # Train and evaluate with different hyperparameter configurations
    for cfg in params_list:
        name = f"lr{cfg['lr']}_bs{cfg['batch_size']}_ep{cfg['epochs']}"
        output_dir = os.path.join(base_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Training config {name}")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        args_train = TrainingArguments(
            output_dir=output_dir,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"],
            learning_rate=cfg["lr"],
            num_train_epochs=cfg["epochs"],
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

        wandb.init(project="mrpc-hyperparameter-search", config=cfg, name=name)
        trainer.train()
        for log in trainer.state.log_history:
            if 'loss' in log:
                wandb.log({f"train/loss_{name}": log["loss"]})
        eval_result = trainer.evaluate()
        val_accuracy = eval_result["eval_accuracy"]
        val_results[name] = val_accuracy
        wandb.log({f"val_acc_{name}": val_accuracy})
        wandb.finish()

    # Save results to JSON file
    results_path = os.path.join(base_dir, "val_results.json")
    with open(results_path, "w") as f:
        json.dump(val_results, f, indent=4)

    # Find the best configuration (highest validation accuracy)
    best_config = max(val_results, key=val_results.get)
    best_accuracy = val_results[best_config]
    print(f"Best configuration: {best_config} with validation accuracy: {best_accuracy:.4f}")

hyperparameter_search()

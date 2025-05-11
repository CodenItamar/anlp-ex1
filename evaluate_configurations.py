import os
import wandb
import pandas as pd
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
def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=512,
    )
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

def evaluate_configurations():
    """
    Load all checkpoint dirs in ./results, evaluate on test set, return dict config->(loss, accuracy).
    """
    wandb.init(project="eval-configs")
    table = wandb.Table(columns=["Config", "Test Loss", "Test Accuracy"])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    raw = load_dataset("glue", "mrpc")
    test_dataset = raw["test"]
    encoded_test = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    base_dir = "./results"
    results = {}
    for name in sorted(os.listdir(base_dir)):
        cfg_dir = os.path.join(base_dir, name)
        if not os.path.isdir(cfg_dir):
            continue
        # find last checkpoint
        ckpts = [d for d in os.listdir(cfg_dir) if d.startswith("checkpoint-")]
        if not ckpts:
            continue
        ckpts.sort(key=lambda x: int(x.split("-")[-1]))
        last = os.path.join(cfg_dir, ckpts[-1])
        print(f"Evaluating {name} at {last}")
        model = AutoModelForSequenceClassification.from_pretrained(last)
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=TrainingArguments(
                output_dir="./tmp",
                per_device_eval_batch_size=32,
                do_eval=True,
                logging_strategy="no",
            ),
            compute_metrics=compute_metrics,
        )
        out = trainer.predict(encoded_test)
        loss = out.metrics["test_loss"]
        preds = np.argmax(out.predictions, axis=1)
        labels = np.array(encoded_test["label"])
        misclassified = np.sum(preds != labels)
        acc = accuracy_score(labels, preds)
        print(f"{name}: {misclassified} misclassifications out of {len(labels)}")
        results[name] = {
            "loss": loss,
            "accuracy": acc,
            "preds": preds,
            "misclassified": misclassified,
        }
        wandb.log({f"test_loss_{name}": loss, f"test_acc_{name}": acc})
        print({f"test_loss_{name}": loss, f"test_acc_{name}": acc})
        table.add_data(name, loss, acc)
    wandb.log({"evaluation_table": table})

    # -------- Logging the examples of the validation set where best succeeded but worst failed --------
    preds_success = results["lr0.0001_bs16_ep3"]["preds"]
    preds_fail = results["lr0.01_bs32_ep5"]["preds"]
    labels = test_dataset["label"]
    sentences1 = test_dataset["sentence1"]
    sentences2 = test_dataset["sentence2"]
    table = wandb.Table(columns=["Sentence1", "Sentence2", "Label", "Correct_Pred", "Wrong_Pred", "Index"])

    for i in range(len(labels)):
        if preds_success[i] == labels[i] and preds_fail[i] != labels[i]:
            table.add_data(
                sentences1[i],
                sentences2[i],
                labels[i],
                preds_success[i],
                preds_fail[i],
                i
            )

    wandb.log({"correct_in_success_wrong_in_fail": table})

    return results

evaluate_configurations()


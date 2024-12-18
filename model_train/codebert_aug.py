from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset, random_split
from datasets import DatasetDict, Dataset
import torch.nn as nn
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
import os
from torch.nn import functional as F
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_slices_txt(filename):
    with open(filename, 'r') as file:
        content = file.read()
        segments = content.split('------------------------------')

        segments = list(filter(lambda x: len(x) >= 2, segments))

    return  segments


def load_data(tokens_path, labels_path, location_path, segments):
    with open(tokens_path, 'rb') as file1, open(labels_path, 'rb') as file2, open(location_path, 'rb') as file3:
        codes = pickle.load(file1)
        labels = pickle.load(file2)
        location = pickle.load(file3)
    labels = [int(i) for i in labels]

    combined = list(zip(codes, labels, location, segments))
    random.shuffle(combined)
    codes, labels, location, segments = zip(*combined)

    return codes, labels, location, segments


def split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1):
    total_size = len(dataset)
    train_end = int(train_size * total_size)
    val_end = train_end + int(val_size * total_size)

    train_dataset = dataset.select(range(train_end))
    val_dataset = dataset.select(range(train_end, val_end))
    test_dataset = dataset.select(range(val_end, total_size))

    return train_dataset, val_dataset, test_dataset


def preprocess_data(codes, labels, location, segments, tokenizer, max_length=512):
    dataset_dict = {'text': codes, 'label': labels, 'location': location, 'segments': segments}
    dataset = Dataset.from_dict(dataset_dict)

    def process_examples(examples):
        tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
        return {**tokenized_inputs, 'label': examples['label'], 'location': examples['location'], 'segments': examples['segments']}

    dataset = dataset.map(process_examples, batched=True)
    return dataset


def create_data_loaders(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def initialize_model(model_name, num_labels):
    return RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def plot_loss_and_accuracy(trainer, save_path):
    log_history = trainer.state.log_history

    train_loss = [entry['train_loss'] for entry in log_history if 'train_loss' in entry]
    eval_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
    eval_accuracy = [entry['eval_accuracy'] for entry in log_history if 'eval_accuracy' in entry]

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.savefig(save_path + 'Train_loss_results.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(eval_loss, label='Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.savefig(save_path + 'Validation_loss_results.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(eval_accuracy, label='Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.savefig(save_path + 'validation_Accuracy_results.png')
    plt.close()


def plot_confusion_matrix(labels, predictions, matrix_path):
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.savefig(matrix_path + 'confusion_matrix.png')
    plt.close()


def plot_roc_and_pr_curve(labels, predictions, roc_path, pr_path):
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, predictions)

    plt.figure(figsize=(12, 6))

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.savefig(roc_path + 'roc.png')
    plt.close()

    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.savefig(pr_path + 'pr.png')
    plt.close()


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, weight=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        class_weights = torch.tensor([0.3, 0.7]).to(device)
        loss_fct = WeightedFocalLoss(weight=class_weights)

        loss = loss_fct(logits.view(-1, model.module.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'tpr': tpr,
        'fpr': fpr
    }


def train_model(trainer, model_path):
    trainer.train()
    trainer.save_model(model_path)


def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    eval_results_str = str(eval_results).replace("'", "")

    with open('./res/evaluation_results.txt', 'a') as fwrite:
        fwrite.write("Evaluation results:\n" + eval_results_str + '\n')
        fwrite.write("----------------------------------------" + '\n')


def predict(model_path, test_dataset, tokenizer):
    model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    model.eval()

    predictions = trainer.predict(test_dataset)

    predictions = predictions.predictions.argmax(-1)

    test_labels = test_dataset['label']

    count = 0
    s_sum = len(test_labels)
    for i in range(s_sum):
        if predictions[i] != test_labels[i]:
            count += 1

    acc = (s_sum - count) / (s_sum * 1.0)

    with open('./res/predict.txt', 'a') as fwrite:
        fwrite.write("test_predictions: " + str(predictions) + '\n')
        fwrite.write("test_labels:      " + str(test_labels) + '\n')
        fwrite.write("准确率: " + str(acc) + "  总数：" + str(s_sum) + "  预测错误数：" + str(count) + '\n')
        fwrite.write("------------------------------------------" + '\n')

    return predictions


def main():
    tokens_path = "data_new_one/tokens/train_tokens.pkl"
    labels_path = "data_new_one/tokens/train_labels.pkl"
    location_path = "data_new_one/tokens/train_location.pkl"
    
    train_slices_path = "./data_new_one/pointersuse_slices.txt"

    model_path = "./model/codebert_test_2_new.model"

    save_images_1 = "./images_test_2_new/1/"
    matrix_path = "./images_test_2_new/2/"
    roc_path = "./images_test_2_new/3/"
    pr_path = "./images_test_2_new/4/"

    segments = open_slices_txt(train_slices_path)
    codes, labels, location, segments = load_data(tokens_path, labels_path, location_path, segments)

    tokenizer = RobertaTokenizer.from_pretrained('./codebert-base')

    dataset = preprocess_data(codes, labels, location, segments, tokenizer)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    model = initialize_model('./codebert-base', num_labels=2)
    print(model)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )
    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    train_model(trainer, model_path)

    evaluate_model(trainer)

    plot_loss_and_accuracy(trainer, save_images_1)

    predictions = predict(model_path, test_dataset, tokenizer)

    test_labels = test_dataset['label']
    codes_p = test_dataset['text']
    segment = test_dataset['segments']
    locations = test_dataset['location']

    plot_confusion_matrix(test_labels, predictions, matrix_path)

    plot_roc_and_pr_curve(test_labels, predictions, roc_path, pr_path)


if __name__ == "__main__":
    main()



from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, random_split
from transformers import Trainer, TrainingArguments
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./bert-base-uncased',
                                                      num_labels=2,
                                                      output_attentions=False,
                                                      output_hidden_states=False).to(device)


# Custom Dataset class
class CodeVulnDataset(Dataset):
    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = labels

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        label = self.labels[idx]
        encoding = tokenizer(code, truncation=True, padding='max_length', return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


tokens_path = "./data/tokens/train_tokens.pkl"
labels_path = "./data/tokens/train_labels.pkl"
with open(tokens_path, 'rb') as file1, open(labels_path, 'rb') as file2:
    codes = pickle.load(file1)
    labels = pickle.load(file2)

labels = [int(i) for i in labels]

dataset = CodeVulnDataset(codes, labels)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)


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


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./model/bert_base_test.model")

eval_results = trainer.evaluate()

eval_results_str = str(eval_results).replace("'", "")

with open('./res/evaluation_results.txt', 'a') as fwrite:
    fwrite.write("Evaluation results:\n" + eval_results_str + '\n')
    fwrite.write("----------------------------------------" + '\n')

# Load the trained model
model = BertForSequenceClassification.from_pretrained("./model/bert_base_test.model").to(device)

# Set model to evaluation mode
model.eval()


# Function to make predictions
def predict(model, dataloader):
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels.extend(batch['labels'].numpy())
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
    return predictions, labels


# Predict on the test dataset
test_predictions, test_labels = predict(model, test_dataloader)

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:17:50 2025

@author: Admin
"""
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import precision_recall_fscore_support

# Load the dataset
def load_data(file_path):
    """Load the CSV dataset."""
    return pd.read_csv(file_path)

# Preprocess and tokenize function
def preprocess_and_tokenize(row, tokenizer):
    """Preprocess a single row and tokenize it."""
    # Extract text and additional features
    text = row["about_me"]
    intermixed_words = row["intermixed_words"]
    clusters = row["high_density_clusters"]

    # Extract intermixed words and clusters as lists
    intermixed_words_list = re.findall(r"[a-zA-Z0-9@#._-]+", intermixed_words)
    cluster_tokens = re.findall(r"[a-zA-Z0-9@#._-]+", clusters)

    # Tokenize components separately
    text_tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    intermixed_tokens = tokenizer(
        ' '.join(intermixed_words_list),
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt"
    )
    cluster_tokens = tokenizer(
        ' '.join(cluster_tokens),
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt"
    )

    # Combine tokenized outputs into a single dictionary
    combined_tokens = {
        "input_ids": text_tokens["input_ids"].squeeze(0),
        "attention_mask": text_tokens["attention_mask"].squeeze(0),
        "intermixed_input_ids": intermixed_tokens["input_ids"].squeeze(0),
        "intermixed_attention_mask": intermixed_tokens["attention_mask"].squeeze(0),
        "cluster_input_ids": cluster_tokens["input_ids"].squeeze(0),
        "cluster_attention_mask": cluster_tokens["attention_mask"].squeeze(0)
    }

    return combined_tokens

# Normalize non-alphanumeric density
def normalize_non_alpha_density(data):
    """Normalize the non_alpha_density column to [0, 1]."""
    scaler = MinMaxScaler()
    data["non_alpha_density_normalized"] = scaler.fit_transform(data["non_alpha_density"].values.reshape(-1, 1))
    return data

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokenized = preprocess_and_tokenize(row, self.tokenizer)
        appropriateness_label = torch.tensor(row["label"], dtype=torch.float32)
        contact_info_label = torch.tensor(row["contact details"], dtype=torch.float32)
        return tokenized, appropriateness_label, contact_info_label

# Define the mBERT-based model
class ContentModerationModel(nn.Module):
    def __init__(self):
        super(ContentModerationModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(0.3)

        # Appropriateness classifier
        self.appropriateness_classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # Contact information classifier
        self.contact_info_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(bert_outputs.pooler_output)

        appropriateness_logits = self.appropriateness_classifier(pooled_output)
        contact_info_logits = self.contact_info_classifier(pooled_output)

        return appropriateness_logits, contact_info_logits
    
    def save_pretrained(self, save_directory):
        """Save the model's state dictionary."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        print(f"Model saved to {save_directory}/pytorch_model.bin")


# Training function
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        # Move inputs to the device
        tokenized, app_labels, contact_labels = batch
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        app_labels = app_labels.to(device)
        contact_labels = contact_labels.to(device)

        # Forward pass
        app_logits, contact_logits = model(input_ids, attention_mask)

        # Compute losses
        app_loss = nn.BCEWithLogitsLoss()(app_logits.squeeze(-1), app_labels)
        contact_loss = nn.BCEWithLogitsLoss()(contact_logits.squeeze(-1), contact_labels)

        loss = app_loss + contact_loss
        total_train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Validate the model
    val_loss, metrics = validate_model(model, val_dataloader, device)

    return avg_train_loss, val_loss, metrics

# Validation function
def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_app_preds = []
    all_contact_preds = []
    all_app_labels = []
    all_contact_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to the device
            tokenized, app_labels, contact_labels = batch
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            app_labels = app_labels.to(device)
            contact_labels = contact_labels.to(device)

            # Forward pass
            app_logits, contact_logits = model(input_ids, attention_mask)

            # Compute losses
            app_loss = nn.BCEWithLogitsLoss()(app_logits.squeeze(-1), app_labels)
            contact_loss = nn.BCEWithLogitsLoss()(contact_logits.squeeze(-1), contact_labels)

            loss = app_loss + contact_loss
            total_loss += loss.item()

            # Collect predictions and labels
            all_app_preds.extend(torch.sigmoid(app_logits).cpu().numpy())
            all_contact_preds.extend(torch.sigmoid(contact_logits).cpu().numpy())
            all_app_labels.extend(app_labels.cpu().numpy())
            all_contact_labels.extend(contact_labels.cpu().numpy())

    # Binarize predictions for metrics calculation
    all_app_preds = [1 if p >= 0.5 else 0 for p in all_app_preds]
    all_contact_preds = [1 if p >= 0.5 else 0 for p in all_contact_preds]

    # Calculate metrics
    app_precision, app_recall, app_f1, _ = precision_recall_fscore_support(all_app_labels, all_app_preds, average="binary")
    contact_precision, contact_recall, contact_f1, _ = precision_recall_fscore_support(all_contact_labels, all_contact_preds, average="binary")

    avg_loss = total_loss / len(dataloader)
    
    # Return validation loss and average metrics
    return avg_loss, {
        "appropriateness": {
            "precision": app_precision,
            "recall": app_recall,
            "f1": app_f1
        },
        "contact_info": {
            "precision": contact_precision,
            "recall": contact_recall,
            "f1": contact_f1
        }
    }

# Main function
def main():
    # File path to your dataset
    file_path = "Scam_Not_scam_325_.csv"

    # Load the dataset
    print("Loading dataset...")
    data = load_data(file_path)

    # Normalize non-alphanumeric density
    print("Normalizing non-alphanumeric density...")
    data = normalize_non_alpha_density(data)

    # Load mBERT tokenizer
    print("Loading mBERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Split the dataset
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    print("Creating datasets...")
    train_dataset = CustomDataset(train_data, tokenizer)
    val_dataset = CustomDataset(val_data, tokenizer)

    print("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContentModerationModel().to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * 3
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    print("Starting training...")
    for epoch in range(3):
        print(f"Epoch {epoch + 1}/{3}")
        train_loss, val_loss, metrics = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Metrics: {metrics}")

    print("Training complete. Saving model...")
    model.save_pretrained("content_moderation_model")
    tokenizer.save_pretrained("content_moderation_model")
    print("Model saved.")

def fine_tune_model(model_path, new_train_data_path):
    print("Loading new training dataset...")
    data = load_data(new_train_data_path)
    data = normalize_non_alpha_density(data)

    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = ContentModerationModel()
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Preparing fine-tuning dataset...")
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(train_data, tokenizer)
    val_dataset = CustomDataset(val_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    print("Initializing fine-tuning...")
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = len(train_dataloader) * 3
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    print("Fine-tuning...")
    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3")
        train_loss, val_loss, metrics = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Metrics: {metrics}")

    print("Saving fine-tuned model...")
    model.save_pretrained("fine_tuned_content_moderation_model")
    tokenizer.save_pretrained("fine_tuned_content_moderation_model")
    print("Fine-tuned model saved.")

if __name__ == "__main__":
    mode = input("Enter mode (train/evaluate/fine_tune): ").strip().lower()
    if mode == "train":
        main()  # Calls the main training logic.
    #elif mode == "evaluate":
    #    evaluate_model("content_moderation_model", "test_data.csv")  # Replace with your paths.
    elif mode == "fine_tune":
        fine_tune_model("content_moderation_model", "fine_tune_data.csv")  # Replace with your paths.
    else:
        print("Invalid mode. Please enter 'train', 'evaluate', or 'fine_tune'.")

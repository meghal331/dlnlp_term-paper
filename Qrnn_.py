import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

class ZoneoutLayer(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training:
            return x
            
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
        return x * mask + x.detach() * (1 - mask)

class DynamicMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, lengths=None):
        if lengths is None:
            return torch.max(x, dim=1)[0]
        
        batch_size, seq_len, hidden_size = x.size()
        mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(x)
        x = x.masked_fill(~mask, float('-inf'))
        return torch.max(x, dim=1)[0]

class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, zoneout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        
        # Convolution for gates
        self.conv = nn.Conv1d(
            input_size, 
            3 * hidden_size,  # Z, F, O gates
            kernel_size,
            padding=kernel_size-1
        )
        
        self.zoneout = ZoneoutLayer(zoneout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Convolution expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        conv_out = self.conv(x)
        conv_out = conv_out.transpose(1, 2)
        
        # Split into gates
        z, f, o = conv_out.chunk(3, dim=-1)
        
        # Apply activations
        z = torch.tanh(z)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        
        # Initialize hidden state if needed
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
            
        # Process sequence
        outputs = []
        for t in range(seq_len):
            c = f[:, t] * hidden + (1 - f[:, t]) * z[:, t]
            hidden = o[:, t] * c
            outputs.append(hidden)
            
        outputs = torch.stack(outputs, dim=1)
        outputs = self.layer_norm(outputs)
        outputs = self.zoneout(outputs)
        
        return outputs, hidden

class QRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=256, num_layers=3, kernel_size=3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList([
            QRNNLayer(
                embed_dim if i == 0 else hidden_size,
                hidden_size,
                kernel_size=kernel_size
            ) for i in range(num_layers)
        ])
        
        self.pool = DynamicMaxPool()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        lengths = attention_mask.sum(dim=1) if attention_mask is not None else None
        
        hidden = None
        skip_connections = []
        
        for layer in self.layers:
            residual = x
            x, hidden = layer(x, hidden)
            skip_connections.append(x)
            
            # Add skip connection
            if len(skip_connections) > 1:
                x = x + skip_connections[-2]
        
        # Global pooling
        x = self.pool(x, lengths)
        return self.classifier(x)

def train_qrnn():
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Convert to PyTorch datasets
    tokenized_datasets.set_format("torch")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=32,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["test"],
        batch_size=32
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QRNN(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        hidden_size=256,
        num_layers=3
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    num_epochs = 3
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_dataloader):.4f}, Accuracy={accuracy:.2f}%")

if __name__ == "__main__":
    train_qrnn()
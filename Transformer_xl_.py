import torch
import torch.nn as nn
from transformers import XLNetTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torch.nn import functional as F

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Parameter(torch.randn(2 * max_len - 1, d_model))
        
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions += self.max_len - 1
        relative_positions = torch.clamp(relative_positions, 0, 2 * self.max_len - 2)
        return self.embedding[relative_positions]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, rel_pos_enc, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        rel_pos_bias = torch.matmul(q.transpose(1, 2), rel_pos_enc.transpose(-2, -1))
        scores = scores + rel_pos_bias
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(context)

class TransformerXLLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, rel_pos_enc, attention_mask=None):
        attended = self.attention(x, rel_pos_enc, attention_mask)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.feed_forward(x))
        return x

class TransformerXLClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, num_classes=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = RelativePositionEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerXLLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids)
        rel_pos_enc = self.pos_encoding(seq_len, input_ids.device)
        
        for layer in self.layers:
            x = layer(x, rel_pos_enc, attention_mask)
        
        # Get the last token's representation for classification
        last_hidden = x[torch.arange(x.size(0)), attention_mask.sum(dim=1) - 1]
        return self.classifier(last_hidden)

def train_model():
    # Hyperparameters
    max_len = 512
    batch_size = 16
    num_epochs = 5
    d_model = 256
    num_layers = 6
    num_heads = 8
    d_ff = 1024
    dropout = 0.1
    learning_rate = 2e-5
    warmup_steps = 1000
    
    # Load dataset
    dataset = load_dataset('imdb')
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    
    # Create datasets
    train_dataset = IMDBDataset(
        texts=dataset['train']['text'],
        labels=dataset['train']['label'],
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    test_dataset = IMDBDataset(
        texts=dataset['test']['text'],
        labels=dataset['test']['label'],
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerXLClassifier(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_accuracy = 100. * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            
        print('-' * 50)
    
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    train_model()
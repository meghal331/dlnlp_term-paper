
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

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

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, n_layers=2, dropout=0.2):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # *2 for bidirectional
        
        # Gate layers
        self.input_gate = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.forget_gate = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.output_gate = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        # Embed input tokens
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Pack sequence for LSTM
        packed_len = attention_mask.sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            packed_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply gates
        i = torch.sigmoid(self.input_gate(lstm_out))
        f = torch.sigmoid(self.forget_gate(lstm_out))
        o = torch.sigmoid(self.output_gate(lstm_out))
        
        # Gate-weighted output
        gated_output = o * (i * lstm_out + f * lstm_out)
        
        # Get final hidden state for classification
        mask = attention_mask.unsqueeze(-1).expand(gated_output.size())
        masked_output = gated_output * mask
        
        # Pool over sequence length
        summed = masked_output.sum(dim=1)
        lengths = mask.sum(dim=1)
        averaged = summed / lengths
        
        # Classification layers
        out = self.dropout(averaged)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

def train_bilstm():
    # Hyperparameters
    max_len = 256
    batch_size = 32
    num_epochs = 5
    embedding_dim = 256
    hidden_dim = 512
    n_layers = 2
    dropout = 0.2
    learning_rate = 2e-4
    
    # Load dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
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
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=4
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
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
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for batch in test_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                test_pbar.set_postfix({
                    'loss': f'{test_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        test_accuracy = 100. * correct / total
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_bilstm_model.pth')
        
        print('-' * 50)
    
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')

if __name__ == "__main__":
    train_bilstm()

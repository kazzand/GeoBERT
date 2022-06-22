import torch.nn as nn
import torch


__all__ = ["BertRegressionModelBase"]



class BertRegressionModelBase(nn.Module):
    def __init__(self, bert_model, tokenizer, device = torch.device('cuda'), hidden_dim=768):
        super(BertRegressionModelBase, self).__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.bert_model = bert_model.to(device)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2)
        ).to(device)



    def forward(self, texts):
        encoded_text = self.tokenizer(texts,
                                padding = 'max_length',
                                truncation=True,
                                max_length=64,
                                return_tensors='pt')
        encoded_text = encoded_text.to(self.device)
        outputs = self.bert_model(**encoded_text)
        lhs = outputs['last_hidden_state'].mean(dim=1)
        logits = self.mlp(lhs)
        return logits




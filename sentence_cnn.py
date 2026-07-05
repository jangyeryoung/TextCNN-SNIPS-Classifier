import torch
from torch import nn


class ConvFeatures(nn.Module):
    def __init__(self, word_dimension, filter_lengths, filter_counts, dropout_rate):
        super().__init__()
        # Convolution filter 모듈 저장
        conv = [] 
        for size, num in zip(filter_lengths, filter_counts): 
            conv2d = nn.Conv2d(1, num, (size, word_dimension)) 
            nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu') 
            nn.init.zeros_(conv2d.bias) 
            conv.append(nn.Sequential(conv2d, nn.ReLU(inplace=True))) 

        self.conv = nn.ModuleList(conv) 
        self.filter_sizes = filter_lengths 
        self.dropout = nn.Dropout(dropout_rate) 

    def forward(self, embedded_words):
        features = [] 
        for conv in self.conv: 
            conv_output = conv(embedded_words)
            conv_output = conv_output.squeeze(-1).max(dim=-1)[0] 
            features.append(conv_output)
            del conv_output

        features = torch.cat(features, dim=1) 
        dropped_features = self.dropout(features)
        return dropped_features

class SentenceCnn(nn.Module):
    def __init__(self, nb_classes, filter_lengths, filter_counts, dropout_rate):
        super().__init__()

        vocab_size = 30000  
        word_dimension = 300  

        self.word_embedding = nn.Embedding(
            vocab_size,
            word_dimension,
            padding_idx=0
        )
        self.word_embedding.weight.requires_grad = True

        # Convolutional layer
        self.features = ConvFeatures(word_dimension, filter_lengths, filter_counts, dropout_rate)

        # Fully-connected layer
        nb_total_filters = sum(filter_counts)
        self.linear = nn.Linear(nb_total_filters, nb_classes)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, input_x):
        x = self.word_embedding(input_x)
        x = x.unsqueeze(1) # Conv2D 입력을 위해 채널 차원 추가
        x = self.features(x)
        logits = self.linear(x)
        return logits
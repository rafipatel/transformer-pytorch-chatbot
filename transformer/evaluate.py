import json
import torch
from torch.utils.data import Dataset
import torch.utils.data
from models import *
from utils import *
from preprocess import *
from train import *

load_checkpoint = True
ckpt_path = r"C:\Users\rafip\Downloads\checkpoint_256_250.pth.tar"
ckpt_path_1 = r"C:\Users\rafip\Downloads\checkpoint_250.pth.tar"



def evaluate(transformer, question, question_mask, max_len, word_map):
    """
    Performs Greedy Decoding with a batch size of 1
    """
    rev_word_map = {v: k for k, v in word_map.items()}
    transformer.eval()
    start_token = word_map['<start>']
    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)
    
    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim = 1)
        next_word = next_word.item()
        if next_word == word_map['<end>']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim = 1)   # (1,step+2)
        
    # Construct Sentence
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()
        
    sen_idx = [w for w in words if w not in {word_map['<start>']}]
    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])
    
    return sentence


if load_checkpoint:
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    transformer = checkpoint['transformer']

    checkpoint1 = torch.load(ckpt_path_1, map_location=torch.device('cpu'))
    transformer_1 = checkpoint1['transformer']



while(1):
    question = input("Question: ") 
    question = question.lower()
    if question == 'quit':
        break
    max_len = 25 #input("Maximum Reply Length: ")
    enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
    question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    question_mask = (question!=0).to(device).unsqueeze(1).unsqueeze(1)  
    sentence = evaluate(transformer, question, question_mask, int(max_len), word_map)
    print("250:",sentence)

    # enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
    # question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    # question_mask = (question!=0).to(device).unsqueeze(1).unsqueeze(1)  
    sentence1 = evaluate(transformer_1, question, question_mask, int(max_len), word_map)
    print("100:",sentence1)

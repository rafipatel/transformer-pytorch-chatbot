import json
import torch
from torch.utils.data import Dataset
import torch.utils.data
from models import *
from utils import *
# from preprocess import *
from train import *

from torcheval.metrics.text import Perplexity
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
import nltk

import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output

# Ignore warnings
warnings.filterwarnings("ignore")

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
    raw_preds = torch.zeros(0, len(word_map), device=device)
    #print(raw_preds.size())

    metric = Perplexity()
    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        #print('*'*50)
        #print(decoded.size())
        #print('*'*50)
        
        #print(raw_preds.size())
        #print(predictions.size())

        #print('='*10)
        #print(raw_preds)
        raw_preds = torch.cat((raw_preds,predictions),dim=0)
        _, next_word = torch.max(predictions, dim = 1)
        next_word = next_word.item()
        if next_word == word_map['<end>']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim = 1)   # (1,step+2)

 
    word_tensor = words
    # Construct Sentence
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()

    # rows_to_remove = words
    # for index,word in enumerate(output_words):

    
    # new_logits = torch.cat([logits[i].unsqueeze(0) for i in range(logits.size(0)) if i not in rows_to_remove])

    #print(words)   
    #print(raw_preds.size())   
    sen_idx = [w for w in words if w not in {word_map['<start>']}]

    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])

    #print(word_tensor)   
    #print(word_tensor.size())   

    # softmax_rawpreds_tensor = F.softmax(raw_preds,dim=1)
    softmax_rawpreds_tensor = raw_preds


    #print("softu",sum(softmax_rawpreds_tensor.unsqueeze(0)[0][0].tolist()))


    metric.update(softmax_rawpreds_tensor.unsqueeze(0),word_tensor)

    perplexity = metric.compute()
    
    return sentence, perplexity, softmax_rawpreds_tensor.unsqueeze(0), word_tensor


if load_checkpoint:
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    transformer = checkpoint['transformer']

    checkpoint1 = torch.load(ckpt_path_1, map_location=torch.device('cpu'))
    transformer_1 = checkpoint1['transformer']




def chat_with_chatbot(loadCheckpoint):
    # Load model checkpoint
    checkpoint = torch.load(loadCheckpoint, map_location=torch.device('cpu'))
    transformer = checkpoint['transformer']
    def send_message(b):

        question = text_input.value
        questions = question.lower()
        if question.lower() in ['q', 'quit']:
            output_area.append_stdout("Quitting the chat.\n")
            return

        sentence = evaluateInput(evaluate, questions)

        
        # Display question and answer
        output_area.append_stdout(f"Question: {questions}\n")
        output_area.append_stdout(f"ChatGPT-10: {sentence}\n")
        
        # Clear input after processing
        text_input.value = ''
    
    
      # Text input field
    text_input = widgets.Text(
        placeholder='Type something and press enter...',
        description='User:',
        disabled=False
    )
    # Button to send message
    send_button = widgets.Button(description="Send")
    send_button.on_click(send_message)
    
    # Output display area
    output_area = widgets.Output()
    
    # Layout components
    input_box = widgets.HBox([text_input, send_button])
    display(input_box, output_area)

inputs = ["hi","how are you?","do you know a place?"]
responses = ["hi", "i am fine","yes i know"]

def evaluateInput(evaluate, inputs):
    all_perplexity_score = list()
    metric = Perplexity()
    all_bleu = list()
    # for (question,response) in zip(inputs,responses):
    for question in [inputs]:
    # question = input("Question: ") 
        question = question.lower()
        if question == 'q':
            break
        max_len = 25 #input("Maximum Reply Length: ")
        enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
        question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
        question_mask = (question!=0).to(device).unsqueeze(1).unsqueeze(1)  
        sentence, perplexity_score, raw_preds,word_tensor = evaluate(transformer, question, question_mask, int(max_len), word_map)
        
        # metric.update(raw_preds,word_tensor)
        # #print(perplexity_score)
        # all_perplexity_score.append(perplexity_score.item())
        # print("chatGPT-10:",sentence)
    #     response_token = nltk.word_tokenize(response)
    #     output_token = nltk.word_tokenize(sentence)
    #     #print(response_token,output_token)
    #     bleu_score = sentence_bleu([response_token], output_token)

    #     #print('BLEU Score:', bleu_score)
    #     all_bleu.append(bleu_score)
    # avg_perplexity_score = sum(all_perplexity_score)/len(all_perplexity_score)
    # avg_bleu_score = sum(all_bleu)/len(all_bleu)
    #print("Average Perplexity Score =", avg_perplexity_score)
    #print("Average bleu Score =", avg_bleu_score)
    #print(all_perplexity_score)
    #print(metric)
    #print("metric.compute()", metric.compute())



    return sentence #all_perplexity_score,  avg_perplexity_score, metric.compute()

# evaluateInput(evaluate, inputs)

    # enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
    # question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    # question_mask = (question!=0).to(device).unsqueeze(1).unsqueeze(1)  
    # sentence1 = evaluate(transformer_1, question, question_mask, int(max_len), word_map)
    # #print("100:",sentence1)


import json
import torch
from torch.utils.data import Dataset
import torch.utils.data
from models import *
from utils import *

from logger import Logger

d_model = 256
heads = 8
num_layers = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 500

# with open('WORDMAP_corpus.json', 'r') as j:
with open('/users/adfx757/transformer-pytorch-chatbot/WORDMAP_corpus.json', 'r') as j:
    word_map = json.load(j)
    
transformer = Transformer(d_model = d_model, heads = heads, num_layers = num_layers, word_map = word_map)
transformer = transformer.to(device)
adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size = d_model, warmup_steps = 4000, optimizer = adam_optimizer)
criterion = LossWithLS(len(word_map), 0.1)



def train(train_loader, transformer, criterion, epoch,logger):
    
    transformer.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader):
        
        samples = question.shape[0]

        # Move to device
        question = question.to(device)
        reply = reply.to(device)

        # Prepare Target Data
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        # Create mask and add dimensions
        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        # Get the transformer outputs
        out = transformer(question, question_mask, reply_input, reply_input_mask)

        # Compute the loss
        loss = criterion(out, reply_target, reply_target_mask)
        
        # Backprop
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()
        
        sum_loss += loss.item() * samples
        count += samples
        
        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count))
        # logger.log("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count))
            result = sum_loss/count
            logger.log({"loss": result})
            

        
            
            
def training(train_loader, transformer, criterion, epoch,logger):
    for epoch in range(epoch):
        
        train(train_loader, transformer, criterion, epoch,logger)
        
        if epoch > 0 and (epoch % 100) == 0:
            state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
            print("Saving the model")
            torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')


def main():
    # Set random seed for reproducibility

    train_loader = torch.utils.data.DataLoader(mydataset(),
                                           batch_size = 100, 
                                           shuffle=True, 
                                           pin_memory=True)



        # Initialise "wandb" for logging
    wandb_logger = Logger(f"inm706_Transformer", project = "inm706_Trans")
    logger = wandb_logger.get_logger()

    training(train_loader, transformer, criterion, epochs,logger)



if __name__ == "__main__":
    main()


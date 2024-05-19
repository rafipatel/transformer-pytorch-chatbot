from utils import *
from train import *

from model import *

from nltk.translate.bleu_score import sentence_bleu
import nltk
from torcheval.metrics.text import Perplexity

  # Needed for tokenization in BLEU calculation

import ipywidgets as widgets
from IPython.display import display

# Example reference and candidate responses
inputs = [
    "hi",
    "how are you?"
    "What's your favorite movie?",
    "Can you explain the plot of Inception?",
    "Who directed The Shawshank Redemption?",
    "What movies would you recommend for a horror night?",
    "Who starred in Titanic?",
    "Tell me a famous quote from The Godfather.",
    "Which movie won the Best Picture Oscar in 2000?",
    "What's a good romantic comedy?",
    "Explain the rules of Fight Club.",
    "Who played Batman in The Dark Knight?",
    "Can movies change society?",
    "What is the theme of 2001: A Space Odyssey?",
    "Who composed the soundtrack for Interstellar?",
    "What's the highest-grossing film of all time?",
    "Can you name a classic Alfred Hitchcock movie?",
    "What's the best movie for children?",
    "Tell me about the plot of Star Wars.",
    "What year was Casablanca released?",
    "Who is the most famous movie villain?",
    "What does 'I'll be back' refer to?",
    "Why is Citizen Kane considered a great film?",
    "Who won the Academy Award for Best Actor in 1994?",
    "What's the significance of the color red in Schindler's List?",
    "Can you recommend a movie about World War II?",
    "What makes a good documentary?",
    "What is the longest movie ever made?",
    "Who played the lead role in The Matrix?",
    "What is a cult classic?",
    "Why do people like scary movies?",
    "Who directed Avatar?",
    "What was the first animated feature film?",
    "Tell me a joke from a comedy movie.",
    "What's a famous line from Forrest Gump?",
    "What are the main awards in film festivals?",
    "Who is considered the best actress of all time?",
    "What was the budget of Avengers: Endgame?",
    "What are the key elements of a thriller?",
    "How do you define an indie film?",
    "What is noir cinema?",
    "Who wrote the screenplay for Pulp Fiction?",
    "What is the role of a film director?",
    "Can you list some films about virtual reality?",
    "What is the most expensive movie ever made?",
    "What's a good film to watch on Halloween?",
    "Who voiced the genie in Aladdin?",
    "What is the Bechdel Test?",
    "Explain the concept of auteur theory in film.",
    "What movie features the song 'Somewhere Over the Rainbow'?",
    "Why do people enjoy superhero movies?",
    "What's a good spy movie?"
]

responses = [
    "hi",
    "i am fine",
    "I enjoy science fiction movies, like Blade Runner.",
    "Inception revolves around the concept of invading dreams to plant an idea.",
    "Frank Darabont directed The Shawshank Redemption.",
    "For a horror night, I recommend The Conjuring and Get Out.",
    "Leonardo DiCaprio and Kate Winslet starred in Titanic.",
    "A famous quote is, 'I'm gonna make him an offer he can't refuse.'",
    "American Beauty won the Best Picture Oscar in 2000.",
    "You might enjoy watching When Harry Met Sally.",
    "The first rule of Fight Club is you do not talk about Fight Club.",
    "Christian Bale played Batman in The Dark Knight.",
    "Yes, movies can influence societal values and norms.",
    "The theme involves evolution and the nature of humanity.",
    "Hans Zimmer composed the soundtrack for Interstellar.",
    "Avatar is the highest-grossing film of all time.",
    "A classic Alfred Hitchcock movie is Psycho.",
    "Toy Story is great for children.",
    "Star Wars is about the battle between the Rebel Alliance and the Galactic Empire.",
    "Casablanca was released in 1942.",
    "Darth Vader is one of the most famous movie villains.",
    "It's a famous line by Arnold Schwarzenegger from The Terminator.",
    "Its narrative structure and cinematography were revolutionary.",
    "Tom Hanks won for his role in Forrest Gump.",
    "It highlights important elements, contrasting the monochrome setting.",
    "Saving Private Ryan is a great WWII movie.",
    "It should educate, inform, and captivate the audience.",
    "Gone with the Wind has a runtime of nearly 4 hours.",
    "Keanu Reeves played the lead role in The Matrix.",
    "A cult classic gains a dedicated following, often after its initial release.",]


# model_name = data['model']['model_name']
# attn_model = data['model']['attn_model']
# hidden_size = data['model']['hidden_size']
# encoder_n_layers = data['model']['encoder_n_layers']
# decoder_n_layers = data['model']['decoder_n_layers']
# dropout = data['model']['dropout']
# batch_size = data['model']['batch_size']


# loadFilename = None # Set checkpoint to load from; set to None if starting from scratch
# checkpoint_iter = data['model']['checkpoint_iter']
# attention = data['model']['attention']



# clip = data['clip'] 
# teacher_forcing_ratio = data['teacher_forcing_ratio'] 
# learning_rate = data['learning_rate']
# decoder_learning_ratio = data['decoder_learning_ratio']
# n_iteration = data['n_iteration']
# print_every = data['print_every']
# save_every = data['save_every']


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        all_logits = torch.zeros([0, self.decoder.output_size], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Save logits (before softmax)
            all_logits = torch.cat((all_logits, decoder_output), dim=0)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores, all_logits


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores, logits = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words, logits

def input_to_chatbot(encoder, decoder, searcher, voc, inputs, reference_responses):
    nltk.download('punkt')
    metric = Perplexity()
    bleu_scores = list()
    for input_sentence, reference in zip(inputs, reference_responses):
        try:
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words, logits = evaluate(encoder, decoder, searcher, voc, input_sentence)


            output_sentence = ' '.join(output_words).split('.')[0]
            output_sentence = normalizeString(output_sentence)
            # #print input and output
            print("Input:", input_sentence)
            print('kelu:', output_sentence)
            # Calculate BLEU score
            reference_tokens = nltk.word_tokenize(reference)
            output_tokens = nltk.word_tokenize(output_sentence)
            bleu_score = sentence_bleu([reference_tokens], output_tokens)
            #print('BLEU Score:', bleu_score)

            bleu_scores.append(bleu_score)

             # Convert reference sentence to index tensor
            reference_indices = [voc.word2index[word] for word in output_words if word in voc.word2index]

            rows_to_remove = list()
            for index,word in enumerate(output_words):

                if word == "EOS":
                    rows_to_remove.append(index)
            
            new_logits = torch.cat([logits[i].unsqueeze(0) for i in range(logits.size(0)) if i not in rows_to_remove])


            
            #print('='*50)
            #print(output_words)
            #print(rows_to_remove)
            #print(reference_indices)
            #print("logits dimension are", logits.unsqueeze(0).size(),new_logits.unsqueeze(0).size(), torch.tensor([reference_indices]).size() )

            # # Update metric with logits and reference indices
            metric.update(new_logits.unsqueeze(0), torch.tensor([reference_indices]))



        except KeyError:
            print("Error: Encountered unknown word.")
    
    print("Average Bleu Score = ", sum(bleu_scores)/len(bleu_scores))

    # Compute perplexity
    perplexity = metric.compute()
    print("Perplexity = ", perplexity)

   

def chat_with_chatbot(encoder, decoder, searcher, voc):
    def send_message(b):
        input_sentence = text_input.value
        if input_sentence.lower() in ['q', 'quit']:
            output_area.append_stdout("Quitting the chat.\n")
            return
        
        try:
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words, logits = evaluate(encoder, decoder, searcher, voc, input_sentence)
            output_sentence = ' '.join(output_words).split('.')[0]
            output_sentence = normalizeString(output_sentence).replace('eos','')

            
            # Display input and output
            output_area.append_stdout(f"User: {input_sentence}\n")
            output_area.append_stdout(f"Chatbot: {output_sentence}\n")
        
        except Exception as e:
            output_area.append_stdout(f"Error: Encountered unknown word.\n,{e}")
        
        # Clear input after sending
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


# # Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

def conversation(loadFilename):

    checkpoint = torch.load(loadFilename,map_location=torch.device('cpu'))

    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # print('Building optimizers')

    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)



    # print('Models built and ready to go!')


    # # Set dropout layers to ``eval`` mode
    encoder.eval()
    decoder.eval()

    # # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # # Begin chatting (uncomment and run the following line to begin)
    # input_to_chatbot(encoder, decoder, searcher, voc, inputs, responses)
    chat_with_chatbot(encoder, decoder, searcher, voc)



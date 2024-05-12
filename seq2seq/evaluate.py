from utils import *
from train import *

from model import *

######################################################################
# Define Evaluation
# -----------------
#
# After training a model, we want to be able to talk to the bot ourselves.
# First, we must define how we want the model to decode the encoded input.
#
# Greedy decoding
# ~~~~~~~~~~~~~~~
#
# Greedy decoding is the decoding method that we use during training when
# we are **NOT** using teacher forcing. In other words, for each time
# step, we simply choose the word from ``decoder_output`` with the highest
# softmax value. This decoding method is optimal on a single time-step
# level.
#
# To facilitate the greedy decoding operation, we define a
# ``GreedySearchDecoder`` class. When run, an object of this class takes
# an input sequence (``input_seq``) of shape *(input_seq length, 1)*, a
# scalar input length (``input_length``) tensor, and a ``max_length`` to
# bound the response sentence length. The input sentence is evaluated
# using the following computational graph:
#
# **Computation Graph:**
#
#    1) Forward input through encoder model.
#    2) Prepare encoder's final hidden layer to be first hidden input to the decoder.
#    3) Initialize decoder's first input as SOS_token.
#    4) Initialize tensors to append decoded words to.
#    5) Iteratively decode one word token at a time:
#        a) Forward pass through decoder.
#        b) Obtain most likely word token and its softmax score.
#        c) Record token and score.
#        d) Prepare current token to be next decoder input.
#    6) Return collections of word tokens and scores.
#

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
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


######################################################################
# Evaluate my text
# ~~~~~~~~~~~~~~~~
#
# Now that we have our decoding method defined, we can write functions for
# evaluating a string input sentence. The ``evaluate`` function manages
# the low-level process of handling the input sentence. We first format
# the sentence as an input batch of word indexes with *batch_size==1*. We
# do this by converting the words of the sentence to their corresponding
# indexes, and transposing the dimensions to prepare the tensor for our
# models. We also create a ``lengths`` tensor which contains the length of
# our input sentence. In this case, ``lengths`` is scalar because we are
# only evaluating one sentence at a time (batch_size==1). Next, we obtain
# the decoded response sentence tensor using our ``GreedySearchDecoder``
# object (``searcher``). Finally, we convert the response’s indexes to
# words and return the list of decoded words.
#
# ``evaluateInput`` acts as the user interface for our chatbot. When
# called, an input text field will spawn in which we can enter our query
# sentence. After typing our input sentence and pressing *Enter*, our text
# is normalized in the same way as our training data, and is ultimately
# fed to the ``evaluate`` function to obtain a decoded output sentence. We
# loop this process, so we can keep chatting with our bot until we enter
# either “q” or “quit”.
#
# Finally, if a sentence is entered that contains a word that is not in
# the vocabulary, we handle this gracefully by printing an error message
# and prompting the user to enter another sentence.
#

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
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")



######################################################################
# Run Evaluation
# ~~~~~~~~~~~~~~
#
# To chat with your model, run the following block.
#

# Set dropout layers to ``eval`` mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher, voc)


######################################################################
# Conclusion
# ----------
#
# That’s all for this one, folks. Congratulations, you now know the
# fundamentals to building a generative chatbot model! If you’re
# interested, you can try tailoring the chatbot’s behavior by tweaking the
# model and training parameters and customizing the data that you train
# the model on.
#
# Check out the other tutorials for more cool deep learning applications
# in PyTorch!
#

# Chatbot Project

This repository contains the coursework for INM706, focused on the design and development of an intelligent chatbot using modern Natural Language Processing (NLP) techniques. 

The project employs Seq2Seq and Transformer architectures, enhanced with attention mechanisms to enable the chatbot to conduct coherent and contextually relevant conversations. Applications of this chatbot span various sectors, including customer service, healthcare, and education.

## Repository Structure

- `seq2seq/` or `transformer/`: Contains all source code for the project.
- `data/`: Directory for datasets used, including links to external sources.
- `model_name/train.py`: To train the model.

## Installation

To set up the project environment to run the code, follow these steps:

```bash
git clone https://github.com/rafipatel/transformer-pytorch-chatbot.git
cd transformer-pytorch-chatbot
pip install -r requirements.txt
```

## Usage

- Change the directory to desired model (either Seq2Seq or Transformer)
To train the chatbot model, run:

```bash
python train.py
```

To interact with the trained chatbot, run:

```bash
python conversations.py
```

## Datasets

The chatbot is trained on the Cornell Movie-Dialogs Corpus. The dataset can be accessed through the following link:

- [Cornell Movie-Dialogs Corpus](https://cityuni-my.sharepoint.com/:f:/g/personal/rafi_patel_city_ac_uk/Eg9BcKirvnVIt_W48I0TTRIBeJDurThlus1Uaq0AY-NkTQ?e=hAI8Eo)

## Models

The project includes two main types of models:

- **Seq2Seq Model**: A sequence-to-sequence model with optional attention mechanisms.
- **Transformer Model**: Utilizes multi-head attention to improve the understanding of context.

## Checkpoints
- **Checkpoints** - [Checkpoints for both models can be found here](https://cityuni-my.sharepoint.com/:f:/g/personal/rafi_patel_city_ac_uk/Eo_76PrXABFBpKQbLkZwy6UBxgk9C0d_fewsFGeI58hcvw?e=ihmNwB)

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- Code and architecture insights were adapted from the following repositories:
  - [Practical PyTorch - Seq2Seq](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation)
  - [FloydHub Textutil](https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus)

Thank you to the instructors and peers for their guidance and feedback throughout this coursework.
```
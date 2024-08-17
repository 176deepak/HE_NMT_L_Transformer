# Hindi to English Translation (NMT) using a Transformer (from Scratch)

Welcome to the wild world of neural wizardry! 🧙‍♂️✨ In this project, we’re diving deep into building a Transformer for Neural Machine Translation entirely from scratch. Our journey is broken into four exciting quests:

1. **Building a Hindi-English Tokenizer from scratch** 📝🔤 (because who doesn't enjoy some linguistic alchemy?)
2. **Training Word2Vec Word Embeddings from the ground up** 💫📚 (we’re making it from scratch, of course)
3. **Creating a Transformer Model from scratch** 🧠⚡ and conducting rigorous training
4. **Evaluating the model's performance** 🧪📊

Get ready for an exciting journey through the mystical realms of NLP! 🚀🌟


## Data
One important question is, where does the data come from? We used the IIT Bombay Hindi-English Parallel Corpus for training our tokenizer, embeddings, and model.
[Data Link](https://www.cfilt.iitb.ac.in/iitb_parallel/)
[Data Source](https://huggingface.co/datasets/cfilt/iitb-english-hindi)

## Tokenizer
The first step in building any NLP application is the tokenizer. Here, we used the Byte Pair Encoding (BPE) algorithm to create our tokenizer.

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

## Word Embedding
The next step in building our application is to train embeddings model on data, so that we can use better context embedding. Here we have used the word2vec solution for training our embedding model. 

`Note: We have used the tokenizers library for BPE tokenizer to train our dataset. Later, we will create our BPE(something else) tokenizer from scratch. Later on we will train our word2vec solution using CBOW or skipgrams model.`

## Transformer
Now, It's time to build our transformer model. We build a transformer model using the very first transformer model(`Attention Is All You Need`)[Research Paper](https://arxiv.org/abs/1706.03762). 


# How to Run:
For running this project on your local machine, first you need to clone the repo using below command. 

`Note: Below instructions are for windows machine, you can run this project on any machine with some minor changes in commands.`

```cmd
git clone https://github.com/176deepak/HE_NMT_L_Transformer.git
```

Jump to project folder
```
cd HE_NMT_L_Transformer
```

Now, you need to create your python venv using below command.
```
python -m venv [ENV_NAME]
```

Activate the your venv
```
[ENV_NAME]\Scripts\activate.bat
```

Install the dependencise, using
```
pip install -r requirements.txt
```

Now, run the below command for executing the main.py file
```
python main.py --TrainTokenizer --TrainEmbedding
```
After executing above command in terminal, you will see below output logs on your screen:

![command](/screenshot/Screenshot%202024-09-03%20164641.png)
![Data Loading](/screenshot/Screenshot%202024-09-03%20164612.png)
![BPE Tokenizer](/screenshot/Screenshot%202024-09-03%20162842.png)
![Word Embedding](/screenshot/Screenshot%202024-09-03%20163724.png)

The above command execution, create the following directories and stores the data, tokenizer configs, embedding checkpoints, word embeddings.
![Artifacts Folder](/screenshot/Screenshot%202024-09-04%20223101.png)

`Note: You can execute the main file without args also. This is just for training the tokenizer and embedding on loaded data.`
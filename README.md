# Classifying Emotions in Text with Deep Learning and Neural Networks
### By Larry Feng

## Introduction

For my final project, I chose a dataset on Kaggle about Natural Language Processing (NLP), https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp. Emotion: Instinctive or intuitive feeling as distinguised from reasoning or knowledge. They can be found in verbal conversations or in text. For this project I will be focusing on text data. Emotions reflect different perspectives for an action, so they are innately expressed in "dynamic linguistic forms". And, accurately capturing these linguistic forms is difficult because of our ever-growing language, which is full of slang and context. For example, the sentence: "i still love my so and wish the best for him..." is categorized as sadness. But, if we parse through this sentence word by word, this sentence would not make sense in formal English. This is because of the abbreviation so => S.O. => Significant Other, which is basically an abbreviation for romantic partner. But, how would an algorithm know that? Well, I will be using Deep Learning and Neural Network methods to train a model so that it can learn these 

## Data

The data is all text data that contains sentences with some type of emotion: anger, sadness, joy, suprise, and love. Although there is an option to use the pre-processing method to clean the data, I decided to first try to make a model without pre-processing to see the results. In addition, reviewing the data, I believe the words that can be cleaned out are articles (i.e. the, a, an, etc.), which would provide a minimal affect on the model. To begin, I plotted the data in a histogram format to get a visualization of the data:

![image](https://user-images.githubusercontent.com/58920498/168455580-a685ff3e-e75d-4fde-b34e-a11d04ab7e7b.png)

The plot above represents the classification each sentence is defined as and their count within the train data. As plotted, there are significantly more joy and sadness data than others. Some other plots to consider are the number of characters present in each sentence:

![image](https://user-images.githubusercontent.com/58920498/168455613-76b9bb68-1cd1-433e-a4f6-5069a12bd3ff.png)

The number of words appearing in each sentence:

![image](https://user-images.githubusercontent.com/58920498/168455630-5f9f69d5-1644-45f7-94d8-dff479238f1e.png)

A plot of the stopwords:

![image](https://user-images.githubusercontent.com/58920498/168455637-8d6444ab-3ccb-4d33-a218-50a8ab704ee9.png)

Although I do not use stopwords in my models, I think its interesting to see the data plotted. The same applies for the rest of the plots. I don't have any code that can make use of the length of the data, but it is interesting to see how the length is distributed.

## Models

### BERT (Bidirectional Encoder Represetations from Transformers)

BERT is a machine learning technique developed by Google and is transformer based for Natural Language Processing (NLP). The key technical innovaiton is applying the bidirectional training of "transformer" to language modelling. This is in contrast to previous efforts which focus on the text sequence from either the left-right or right-left training. Results from other studies have shown that a language model that has been bidirectionally trained can have a deeper sense of langauge context and flow compared to a single-direction model. BERT uses "transformer's" techniques of producing predictions for tasks to generate a language model. Compared to directional models, a "transformer" encoder reads the entire sequence of words at once. This allows the model to learn the context of a word based on all of its surroundings. 

The chart in the below section is a high-level description of the "transformer" encoder. The input is a sequence of tokens which are first embedded into vectors and then processed in the neural network. The output is a sequence of vectors of size H, in which each vector corresponds to an input token with the same index.

![image](https://user-images.githubusercontent.com/58920498/168461460-7da5c944-ffec-477f-8e25-a9e9898672cb.png)

When training model languages, there is a challenge of defining a prediction goal- many models predict the next word in a sequence which is a directional approach which inherently limits context learning. So, to overcome this challenge, BERT uses two training strategies: Masked LM (MLM) and Next Sentence Prediction (NSP).

#### Masked LM

MLM consists of giving BERT a sentence and optimizing the weights inside BERT to output the same sentence on the other side. So we input a sentence and ask that BERT outputs the same sentence. But, before we actually give BERT the input, we mask a few tokens.

![image](https://user-images.githubusercontent.com/58920498/168461527-c73cd4bf-fede-40d3-8c46-2fd178c40329.png)

Basically, we are inputting an incomplete sentence and asking BERT to complete it for us. There are a couple of steps that occur within MLM.

1. We tokenize our text just like we usually do with transformers, and we begin text tokenization. From tokenization we receive three different tensors: *input_ids*, *token_type_ids*, and *attention_mask*. 
2. We create a labels tensor to train our model and use the labels tensor to calculate the loss and optimize after.
3. We masks tokens in *input_ids*. Now that we created a copy of *input_ids* for labels, we can mask a random selection of tokens.
4. We calculate loss by processing the *input_ids* and labels tensors through our BERT model and calculate the loss between both. 

#### Next Sentence Prediction (NSP)

For NSP, themodel receives pairs of sentences as input and learns to predict if the econd sentence in the pair is the subsequent sentence in the oiginal document. During training, 50% of the inputs are a pair in which the second sentence is the subsequent sentence in the original document, while in the other 50%, a random sentence from the corpus is chosen as a second sentence. We assume that the rando msentence will be disconnected from the first sentence.

To help the model distinguish between the two sentences in training, the input is processed in the following way before entering the model:

1. a [CLS] token is inserted at the beginning of the first sentence and a [SEP] token is inserted at the end of the sentence.
2. A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings ar esimilar in concept to token embeddings with a vocabulary of 2.
3. A positional embedding is added to each token to indicate its position in the sequence.

![image](https://user-images.githubusercontent.com/58920498/168491733-1395812c-524d-4717-b3b0-3fbd780a2994.png)

To predict if the second sentence is indeed connected to the first, we need a few steps:
1. The entire input sequence passes through the the transformer model.
2. The output of the [CLS] token is transformed into a 2 x 1 shaped vector, using a simple classification layer
3. We calculate the probability of IsNextSequence with softmax.

When training a BERT model, MLM and NSP are trained together, and their common goal is to minimize the combined loss function of the two techniques.

### Recurrent Neural Networks (RNN)

RNNs are a form of machine learning algorithm that are ideal for sequential data such as text data for natural language processing. They are ideal for solving problems where the sequence is more important than the individual items themselves. So, they're essentially a fully connected neural network that contains a refactoring of some of its layers into a loop. That loop is typically an iteration over the addition or concatenation of two inputs, which is a matrix multiplication and a non-linear function.

Let's compare an RNN to a fully connected neural network. If we take a sequence of three words of text and a network that predicts the fourth word, the network had three hidden layers, each of which are an affine function (i.e. matrix dot product multiplication), followed by a non-linear function, then the last hidden layer is wrapped up with an output from the last layer activation function. 

The input vectors representing each word in the sequence are the lookups in a word embedding matrix, based on a one hot encoded vector representing the word in the vocabulary. Not that all inputted words use the same word embedding. In this context a word is actually a token that could represent a word or a punctuation mark. The output will then be a one hot encoded vector representing the predicted fourth word in the sequence.

The first hidden layer takesa vector that represents the first word in the sequence as an input, and the output activations serve as one of the inputs into the second hidden layer. The second hidden layer follows the same structure as the second hidden layer, taking the activation from the second hidden layer combined with the vector representing the third word in the sequence. Once again, the inputs are added/concatenated together. 

The output from the last hidden layer goes through an activation function that prodcuces an ouput representing a word from the vocabulary, as one hot encoded vector. The second and third hidden layer can both use the same weight matrix, which opens the opportunity of refactoring this into a loop to becoming recurrent. 

![image](https://user-images.githubusercontent.com/58920498/168494357-bc5315de-d434-4101-9b73-b9472334268e.png)

For a network to be recurrent, a loop needs to be factored into the network's model. it makes sense to use the same embedded weiht matrix for every word input. This means we can replace the second and third layers with iternations within a loop. Each iteration of the loop takes an input of a vector representing the next word in the sequence with the output activations from the last iteration. These inputs are added/concatenated together. The output from the last iteration is a representation of the next word in the sentence being put through the last layer activaiton function which converts it to a one hot encoded vector representing a word in the vocabulary. This allows the network to predict a word at the end of the sequence of any arbitrary length.

![image](https://user-images.githubusercontent.com/58920498/168494619-f09088ec-fa67-4dea-a483-1c82bf8ee5b7.png)

Once at the end of the sequence of words, the predicted ouptut ofthe next word could be stores and appended to an array for additional information in the next iteration. Each iteration then has access to the previous predictions. In theory, the sequence of predicted text could be infinite in length with a predicted word following the last predicted word in the loop.

![image](https://user-images.githubusercontent.com/58920498/168494961-a1178721-3ade-4169-a3c6-ef8e5f8fd589.png)

To get more layers of computation to be able to solve or approximate more complex takss, the output of the RNN could be fed into another RNN, or many more layers. But, as the number of layers of RNNs increases, the loss landscape can become impossible to train on- this is a vanishing gradient problem. It can also be fixed by using a Long Term Short Term Memory (LSTM) network. LSTM takes the current input and previous hidden state. It then computes the next hidden state. As part of this computation, the sigmoid function combines the values of these vectors between zero and one and then mulitplies them.

A RNN has short term memory, but with LSTM the network can have long term memory. Instead of the recurring seciton of an RNN, the LTSM is a small neural netowrk consisting of four neural network layers. These recurring layers are from the RNN with three networks acting as "gates". The LTSM has a cell state too, along with a hidden state. This cell state is the long term memory. Instead of just returning the hidden state at each iteration, a tuple of hidden states are returned comprised of cell states and hidden states. As stated, the LSTM has three gates:

1. Input gate- controls the information input at each time step.
2. Output gate- controls how much information is outputted to the next cell or upward layer.
3. Forget gate- controls how much data to lose at each time step.

## Experiments

### BERT

I first installed the necessary packages and libraries for BERT. At first I was confused on what to do when my import failed, but I just had to manually install the library using `pip`. I also am running this notebook on a local runtime through Anaconda's jupyter to Google Colab. My GPU is a NVIDIA GTX 1070 8GB, and I had some quick runtimes with this code. 

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

SEED = 410

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == torch.device("cuda"):
    torch.cuda.manual_seed_all(SEED)
```

My next step was to create a label encoder to 

## Conclusion

## References
https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
https://towardsdatascience.com/recurrent-neural-networks-and-natural-language-processing-73af640c2aa1


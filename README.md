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

## Experiments

## Conclusion

## References
https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

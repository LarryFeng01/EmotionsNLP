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

BERT is a machine learning technique developed by Google and is transformer based for Natural Language Processing (NLP). 

## Experiments

## Conclusion

## References
https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

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

My next step was to create a label encoder to handle the categorical variables and then prepare our tokenizer, *input_id*, *attention_masks*, and *labels*. Now, we create our train/test split with a train size of 90% and test size of 10%. I use a batch size of 32 since my GPU has 8GB of VRAM and cannot handle a larger batch size. Before we train, I create the model and set the following parameters:

```
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6).to(device)
lr = 2e-5
adam_epsilon = 1e-8
epochs = 6
num_warmup_steps = 0
num_training_steps = len(train_dataloader)*epochs

optimizer = AdamW(model.parameters(), lr = lr, eps = adam_epsilon, correct_bias = False)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps)
```
Now, we can train our model while also validating with the code:

```
train_loss_set = []
learning_rate = []

model.zero_grad()

for _ in tnrange(1,epochs+1, desc='Epoch'):
    print(F"Epoch: {_}")
    batch_loss = 0
    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        b_labels = b_labels.type(torch.LongTensor) #solution for error:
            #nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
        
        outputs = model(b_input_ids.cuda(), token_type_ids=None, attention_mask=b_input_mask.cuda(), labels=b_labels.cuda())
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        batch_loss += loss.item()
    avg_train_loss = batch_loss / len(train_dataloader)

    for param_group in optimizer.param_groups:
        print("\n\tCurrent learning rate: ", param_group['lr'])
        learning_rate.append(param_group['lr'])

    train_loss_set.append(avg_train_loss)
    print(F'\n\tAverage training loss: {avg_train_loss}')

    model.eval()
    eval_accuracy, eval_mcc_accuracy, nb_eval_steps = 0,0,0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_labels = b_labels.type(torch.LongTensor)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask)
        logits = logits[0].to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        df_metrics = pd.DataFrame({'Epoch':epochs, 'Actual_class':labels_flat, 'Predicted_class':pred_flat})

        tmp_eval_accuracy = accuracy_score(labels_flat, pred_flat)
        tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)

        eval_accuracy += tmp_eval_accuracy
        eval_mcc_accuracy += tmp_eval_mcc_accuracy
        nb_eval_steps += 1
    print(F'\n\tValidation Accuracy: {eval_accuracy/nb_eval_steps}')
    print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy/nb_eval_steps}')
```
After a few run throughs, the average time to complete training on the model is around 20 minutes.

With some variation, our validation accuracy ranges from .92 to .94 and our average loss decreased from .35 to less than .10. Below are plots of our models' accuracy throughout epochs while testing different parameters. The parameter "128 max length" refers to a max_length of 128 words for the tokenizer and input_ids; I also made this group have a batch size of 16. Contrastly, the 256 max length parameter has a batch size of 32. 

Accuracy:
![image](https://user-images.githubusercontent.com/58920498/168498440-e59250e3-d84c-455e-a135-88ba6f776448.png)

Average Loss:
![image](https://user-images.githubusercontent.com/58920498/168498449-8b9adf7c-289c-43b2-a977-0258df3ec01c.png)

We can also receive the classification report by comparing our predicted values with the actual values.
```
print(classification_report(df_metrics['Actual_class'].values, df_metrics['Predicted_class'].values, target_names=label2int.keys(), digits=len(label2int)))
```

Outputs:
```
              precision    recall  f1-score   support

     sadness    1.00000   1.00000   1.00000         2
         joy    1.00000   1.00000   1.00000         1
       anger    1.00000   0.83333   0.90909         6
        fear    0.00000   0.00000   0.00000         0
    surprise    1.00000   1.00000   1.00000         7

    accuracy                        0.93750        16
   macro avg    0.80000   0.76667   0.78182        16
weighted avg    1.00000   0.93750   0.96591        16
```
```
              precision    recall  f1-score   support

     sadness    1.00000   0.66667   0.80000         3
         joy    0.66667   1.00000   0.80000         2
       anger    0.87500   1.00000   0.93333         7
        fear    1.00000   0.50000   0.66667         2
    surprise    1.00000   1.00000   1.00000         2

    accuracy                        0.87500        16
   macro avg    0.90833   0.83333   0.84000        16
weighted avg    0.90365   0.87500   0.86667        16
```

From our classification report, we see a high accuracy across our models. Our accuracy ranges from 0.875 and 0.9375 while our weighted average ranges from 0.867 and .965.

### Recurrent Neural Networks Model

We start by using the tokenizer class to convert the sentences into word vectors
```
tokenizer=Tokenizer(15212,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(X)
X_train=tokenizer.texts_to_sequences(X)
X_train_pad=pad_sequences(X_train,maxlen=80,padding='post')
df_train['label']=df_train.label.replace({'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5})
```

After changing the labels for convenience, we can start making the model. I chose to use embedding, Dropout, LSTM, and softmax for my layers. 
```
model=Sequential()
model.add(Embedding(15212,64,input_length=80))
model.add(Dropout(0.6))
model.add(Bidirectional(LSTM(80,return_sequences=True)))
model.add(Bidirectional(LSTM(160)))
model.add(Dense(6,activation='softmax'))
print(model.summary())
```
And the output:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 80, 64)            973568    
                                                                 
 dropout (Dropout)           (None, 80, 64)            0         
                                                                 
 bidirectional (Bidirectiona  (None, 80, 160)          92800     
 l)                                                              
                                                                 
 bidirectional_1 (Bidirectio  (None, 320)              410880    
 nal)                                                            
                                                                 
 dense (Dense)               (None, 6)                 1926      
                                                                 
=================================================================
Total params: 1,479,174
Trainable params: 1,479,174
Non-trainable params: 0
_________________________________________________________________
```

Next, we can compile the model with "Adam" and now we can train the model. 

We run the follow line to start fitting:

```
hist=model.fit(X_train_pad,Y_train_f,epochs=15,validation_data=(X_val_pad,Y_val_f))
```

Our first epoch returns:
```
Epoch 1/15
500/500 [==============================] - 17s 34ms/step - loss: 0.2099 - accuracy: 0.9244 - val_loss: 0.2069 - val_accuracy: 0.9210
```
And our last epoch is:
```
Epoch 15/15
500/500 [==============================] - 17s 34ms/step - loss: 0.0431 - accuracy: 0.9830 - val_loss: 0.1739 - val_accuracy: 0.9375
```

Although we do not see a significant jump from accuracy (i.e. .40s to .90s), this result is still very satisfactory for now. Since there is an issue of overfitting, we have to check with the test data to make sure. This is the result:
```
63/63 [==============================] - 1s 16ms/step - loss: 0.2004 - accuracy: 0.9290
[0.2004082202911377, 0.9290000200271606]
```
With the first number being the loss and the second being the accuracy. With these results I am satisfied with the model and deem it not significantly overfit.

Accuracy Curve:
![image](https://user-images.githubusercontent.com/58920498/168500738-c2791afa-c7de-41da-8a34-a313a36976be.png)

Loss Curve:
![image](https://user-images.githubusercontent.com/58920498/168500753-41fe4dd1-8edd-4b5f-b167-7b7b8db37904.png)

## Conclusion

In this research, I proposed two different types of models to extract emotion-relevant representations. One is a supervised machine learning technique, while the other is not. My main goal was to compare these two models to see which is more efficient and produces the better results. But, for the second part, I find it hard to compare accuracies when the model ranges in accuracy, and both models have high accuracies. However, if I had to choose a model, I would say the RNN is preferred because of its consistent, high accuracy and low run-time. 

In the future work, I aim to investigate the model's parameters more in-depth and find more optimal parameters that I could not get to due to run-time and deadlines. I also plan to look at other models just to compare what is available for NLP data. In addition, using only my notebook again, I would like to run hundreds of epochs with my current models so that I can get a solid number for my accuracy and loss scores in order to more precisely compare the two models against each other. As stated, I am not able to constantly run the code on my machine due to time constraints and other priorities. 

## References
https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
https://towardsdatascience.com/recurrent-neural-networks-and-natural-language-processing-73af640c2aa1
https://aclanthology.org/D18-1404.pdf

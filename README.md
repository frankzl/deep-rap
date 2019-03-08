# deep-rap
**The dream:** generating rap lyrics

### How to read:
We have written jupyter notebooks for every step that we took. It shows our incremental approach to the problem. The notebooks can be read like chapters and at the end of each chapter, we describe our learnings and what we should improve on.

Generally, the next notebook will try to fix those problems shown in the previous and apply the learnings.

## Table of Contents
00. [Generating Letter by Letter](#00)
01. [Introducing the Tools](#01)
02. [Generating Word by Word](#02)
03. [Applying Multiple Layers](#03)
04. [Embedding the Words](#04)
05. [Cleaning the Rap](#05)
06. [Working on Clean Rap](#06)
07. [Building Embeddings](#07)
08. [Glovely Embedded](#08)
09. [Glove Applied](#09)
10. [Glove and Phonem](#10)


## Content
#### 00 - Generating Letter by Letter <a name="00"></a>
To start our project, we predict letter by letter. Given a sequence of text, we try to infer the next letter, e.g. 

1. **"Hello my name is "** &emsp; => **"A"**
2. **"ello my name is A"** &emsp; => **"l"**
3. **"llo my name is Al"** &emsp; => **"e"**
4. **"lo my name is Ale"** &emsp; => **"x"**

We can repeat these steps to generate an arbitrary number of text. Of course we have trained the model before on a rap text by 2pac.

Basically we are doing a multiclass classification, where we try to classify the next letter given the predecessors. In the notebook, we used a single LSTM cell followed by a linear output layer.

<hr>

#### 01 - Introducing the Tools <a name="01"></a>
Here we describe how we outsource some of the functions to separate modules, in order to reuse them for future notebooks.
We introduce a ```train_model``` function that can be used to train any ```Trainable``` object.

If you want to understand more about how we set up our toolchain and outsourced the functions, please check out the notebook. It might be a bit hard to understand at first though.

It is also crucial to understand the ```sample``` function, as that describes how we can sample from a ```Trainable``` object.

<hr>

#### 02 - Generating Word by Word <a name="02"></a>
In chapter [00](#00), we were generating letter by letter. This is a complex task as it has a lot of room for errors. One single spelling mistake can influence the whole rest of the sentence. We therefore want to generate word by word instead. 

Basically, the approach is the same as in 00, but this time we have a different **Dictionary**. Instead of using the **Alphabet**, we use the **Vocabulary** instead. (Please checkout the module ```tools.processing``` to understand more about this)

<hr>

#### 03 - Applying Multiple Layers <a name="03"></a>
The maximum accuracy from chapter [02](#02) was xxx. By stacking LSTM cells, we increase the complexity of our model, so that it can solve more complex problems.

While this appro

Seed: 
killin people left and right 
use a gun cool homie 
that is right

Sampling:

<hr>

#### 04 - Embedding the Words <a name="04"></a>
<hr>

#### 05 - Cleaning the Rap <a name="05"></a>
<hr>

#### 06 - Working on Clean Rap <a name="10"></a>
<hr>

#### 07 - Building Embeddings <a name="06"></a>
<hr>

#### 08 - Glovely Embedded <a name="07"></a>
<hr>

#### 09 - Glove Applied <a name="08"></a>
<hr>

#### 10 - Glove and Phonem <a name="09"></a>

<hr>

### Inspired by

**Eric Malmi et al., DopeLearning: A Computational Approach to Rap Lyrics Generation**



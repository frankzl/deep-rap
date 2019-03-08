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

Accuracy: 49.46 %

**Seed:** <br>
as real as it seems

**Sampling:** <br>
as real as it seems **willed to dene<br>
cares and beave<br>
you kep soun<br>
the doudn't wen the care done**

We are doing a multiclass classification, where we try to classify the next letter given the predecessors. In the notebook, we used a single LSTM cell followed by a linear output layer.

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

While this approach seems to yield relatively high accuracy, the samples are still not very useful.

**Seed:** <br>
killin people left and right <br>
use a gun cool homie <br>
that is right

**Sampling:** <br>
killin people left and right <br>
use a gun cool homie <br>
**that is right so i bust it alone is know think i got up y'all mine <br>
out this thing a clip for can on your got up**

<hr>

#### 04 - Embedding the Words <a name="04"></a>

We now use an embedding lookup for our data. Instead of feeding in 1-hot-encoded words, we feed in the indices for the words and perform an embedding lookup on those words in an embedding matrix that we can also learn.

**Architecture: <br>**

sequence of words &emsp;=>&emsp; sequence of indices &emsp;=>&emsp; sequence of vectors in embedding space &emsp;=>&emsp; Single Layer RNN &emsp;=>&emsp; outputs 1-Hot-Vector<br>

**Seed:**<br>
while i go down the street<br>
while i go down the street<br>
you was lookin' at me <br>
is this even good or is it just bad <br>
is this even good or is it mad<br>

**Sampling:**<br>
while i go down the street<br>
while i go down the street<br>
you was lookin' at me <br>
is this even good or is it just bad <br>
is this even good or is it mad<br>
**your brother and your trifeass wife wants to do me <br>
on a mountain and still couldn't top me**

<hr>

#### 04.1 - Bonus Book - Trying a different text

Here we tried setting a very low time step for our RNN. It is our goal to feed a short sentence and generate the next rap line from that.
Therefore setting the time step to 6, allows us to infer the next word from the last 6 words.

This turns out to be pretty bad despite high accuracy.

Also we tried out a different rapper here (Rakim)

<hr>

#### 05 - Cleaning the Rap <a name="05"></a>
To be honest, this was the worst part of the whole project. It involved:
1. building a new dictionary
2. expanding contractions
3. removing extra letters like tooooools => tools
4. numbers to words: 1 => one
5. correcting spelling mistakes, e.g. somthin => something
6. rebuilding the dictionary with newly found words

There are 4 notebooks on this, order of creation:
1. 05-text-cleanup
2. 05-text-cleanup-rakim
3. 05-text-cleanup-kidcudi

I recommend checking out 3. (I think 1. is a bit messy tbh)

Seriously, this was the worst part

<hr>

#### 06 - Working on Clean Rap <a name="10"></a>
Here we use a Embedded Single Layer RNN from chapter [04](#04) on our freshly cleaned data.
The result is pretty good! 

(Note that we replaced **\n** with **;** and our words are only one **space** apart. In the following we replaced ; with \n for readability)

**Seed:**<br>
when i was thirteen <br>
i had my first love <br>
there was nobody that could put hands on my baby <br>
and nobody came between us that could ever come above <br>

**Sampling:**<br>
when i was thirteen <br> i had my first love <br> there was nobody that could put hands on my baby <br> and nobody came between us that could ever come above <br>
**but now i am guilty of falling for his girlfriend <br> i know it is wrong but it is not a cop damn <br> i am trapped in the hell <br> one if you will find out my homies <br> it is the nigga that you the nigga**

<hr>

#### 07 - Building Embeddings <a name="06"></a>
We train our own embedding on a collection of rap texts.
For this we use the model word2vec to produce our very own embedding.

Didn't work out that well though, since we did not have enough data.

<hr>

#### 08 - Glovely Embedded <a name="07"></a>

Contains some minor corrections of the cleaned up text (repeated character problem). Using the Glove embedding, we can translate almost all of our used words into a feature space of 300 dimensions.

A portion of the unknowns can be corrected using our **tools.spell_correction** module. This module was created after chapter [05](#05).

The rest of the words that are still not recognized by Glove, are mapped to the mean of the embedding vectors.

<hr>

#### 09 - Glove Applied <a name="08"></a>
<hr>

#### 10 - Glove and Phonem <a name="09"></a>

<hr>

### Inspired by

**Eric Malmi et al., DopeLearning: A Computational Approach to Rap Lyrics Generation**



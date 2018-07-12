# Introduction

The goal of this document is to keep track the state-of-the-art in deep reinforcement learning. It starts with basics in reinforcement learning and deep learning to introduce the notations and covers different classes of deep RL methods, value-based or policy-based, model-free or model-based, etc.

Different classes of deep RL methods can be identified. This document will focus on the following ones:   

1. Value-based algorithms (DQN...) used mostly for discrete problems like video games.
2. Policy-gradient algorithms (A3C, DDPG...) used for continuous control problems such as robotics.
3. Recurrent attention models (RAM...) for partially observable problems.
4. Model-based RL to reduce the sample complexity by incorporating a model of the environment.
5. Application of deep RL to robotics

One could extend the list and talk about hierarchical RL, inverse RL, imitation-based RL, etc...

**Additional resources**
    
See @Li2017, @Arulkumaran2017 and @Mousavi2018 for recent overviews of deep RL. 

The CS294 course of Sergey Levine at Berkeley is incredibly complete: <http://rll.berkeley.edu/deeprlcourse/>. The Reinforcement Learning course by David Silver at UCL covers also the whole field: <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html>.

This series of posts from Arthur Juliani <https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0> also provide a very good introduction to deep RL, associated to code samples using tensorflow.


**Notes**

This document is meant to stay *work in progress* forever, as new algorithms will be added as they are published. Feel free to comment, correct, suggest, pull request by writing to <julien.vitay@informatik.tu-chemnitz.de>.

For some reason, this document is better printed using chrome. Use the single file version [here](./DeepRL.html) and print it to pdf. Alternatively, a pdf version generated using LaTeX is available [here](./DeepRL.pdf) (some images may disappear, as LaTeX does not support .gif or .svg images).

The style is adapted from the Github-Markdown CSS template <https://www.npmjs.com/package/github-markdown-css>. The document is written in Pandoc's Markdown and converted to html and pdf using pandoc-citeproc and pandoc-crossref.

Some figures are taken from the original publication ("Taken from" or "Source" in the caption). Their copyright stays to the respective authors, naturally. The rest is my own work and can be distributed, reproduced and modified under CC-BY-SA-NC 4.0. 


**Thanks**

Thanks to all the students who helped me dive into that exciting research field, in particular: Winfried Lötzsch, Johannes Jung, Frank Witscher, Danny Hofmann, Oliver Lange, Vinayakumar Murganoor.

**Copyright**

Except where otherwise noted, this work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0).


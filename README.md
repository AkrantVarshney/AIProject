# lunar-lander: Reinforcement learning algorithms for training an agent to play the game lunar lander

## Introduction

In this repository we implement an agent that is trained to play the game lunar lander using <i>i)</i> an actor-critic algorithm, and <i>ii)</i> a (double) deep Q-learning algorithm. Here is a video of a trained agent playing the game:

https://user-images.githubusercontent.com/37583039/219359191-7988e0dc-b1a4-43cc-82d1-4cc18be0d0a2.mp4

We use the lunar lander implementation from [gymnasium](https://gymnasium.farama.org). For the implementation of the actor-critic algorithm we loosely follow <a href="#ref_1">Ref. [1]</a>. While for the implementation of deep Q-learning we follow <a href="#ref_2">Ref. [2]</a>, for the implementation of double deep Q-learning we follow <a href="#ref_3">Ref. [3]</a>.

In the following, we [first](#files-and-usage) list the files contained in this repository and explain their usage. We [then](#comparison-actor-critic-algorithm-vs-deep-q-learning) compare the training speed and post-training performance of agents trained using the actor-critic algorithm and deep Q-learning.
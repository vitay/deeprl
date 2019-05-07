
# Deep RL in practice

## Limitations

Excellent blog post from Alex Irpan on the limitations of deep RL: <https://www.alexirpan.com/2018/02/14/rl-hard.html>

Another documented critic on deep RL: <https://thegradient.pub/why-rl-is-flawed/>

## Reward shaping

Hindsight experience replay: @Andrychowicz2017

## Simulation environments

Standard RL environments are needed to better compare the performance of RL algorithms. Below is a list of the most popular ones.

* OpenAI Gym <https://gym.openai.com>: a standard toolkit for comparing RL algorithms provided by the OpenAI foundation. It provides many environments, from the classical toy problems in RL (GridWorld, pole-balancing) to more advanced problems (Mujoco simulated robots, Atari games, Minecraft...). The main advantage is the simplicity of the interface: the user only needs to select which task he wants to solve, and a simple for loop allows to perform actions and observe their consequences:

```python
import gym
env = gym.make("Taxi-v1")
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
```

* OpenAI Universe <https://universe.openai.com>: a similar framework from OpenAI, but to control realistic video games (GTA V, etc).

* Darts environment <https://github.com/DartEnv/dart-env>: a fork of gym to use the Darts simulator instead of Mujoco.

* Roboschool <https://github.com/openai/roboschool>: another alternative to Mujoco for continuous robotic control, this time from openAI.

* NIPS 2017 musculo-skeletal challenge <https://github.com/stanfordnmbl/osim-rl>

* Deepmind Lab <https://github.com/deepmind/lab>: a 3D learning environment based on id Software's Quake III Arena via ioquake3 and other open source software.

* AnimalAI Olympics <https://github.com/beyretb/AnimalAI-Olympics>, a gym-like environment aimed at confronting RL algorithms to typical tasks in the animal cognition literature.

## Algorithm implementations

State-of-the-art algorithms in deep RL are already implemented and freely available on the internet. Below is a preliminary list of the most popular ones. Most of them rely on tensorflow or keras for training the neural networks and interact directly with gym-like interfaces.

* <https://github.com/ShangtongZhang/reinforcement-learning-an-introduction>: all the exercises in Python of the [@Sutton2017] book.

* `rl-code` <https://github.com/rlcode/reinforcement-learning>: many code samples for simple RL problems (GridWorld, Cartpole, Atari Games). The code samples are mostly for educational purpose (Policy Iteration, Value Iteration, Monte-Carlo, SARSA, Q-learning, REINFORCE, DQN, A2C, A3C).


* `keras-rl` <https://github.com/matthiasplappert/keras-rl>: many deep RL algorithms implemented directly in keras: DQN, DDQN, DDPG, Continuous DQN (CDQN or NAF), Cross-Entropy Method (CEM), Dueling DQN, Deep SARSA...

* `Coach` <https://github.com/NervanaSystems/coach> from Intel Nervana also provides many state-of-the-art algorithms: DQN, DDQN, Dueling DQN, Mixed Monte Carlo (MMC), Persistent Advantage Learning (PAL), Distributional Deep Q Network, Bootstrapped Deep Q Network, N-Step Q Learning, Neural Episodic Control (NEC), Normalized Advantage Functions (NAF), Policy Gradients (PG), A3C, DDPG, Proximal Policy Optimization (PPO), Clipped Proximal Policy Optimization, Direct Future Prediction (DFP)...

* `OpenAI Baselines` <https://github.com/openai/baselines> from OpenAI too: A2C, ACER, ACKTR, DDPG, DQN, PPO, TRPO...

* `rlkit` <https://github.com/vitchyr/rlkit> from Vitchyr Pong (PhD student at Berkeley) with in particular model-based algorithms (TDM, @Pong2018).

* `chainer-rl` <https://github.com/chainer/chainerrl> implemented in Chainer (an alternative to tensorflow): A3C, ACER, Categorical DQN; DQN (including Double DQN, Persistent Advantage Learning (PAL), Double PAL, Dynamic Policy Programming (DPP)), DDPG, , PGT (Policy Gradient Theorem), PCL (Path Consistency Learning), PPO, TRPO.

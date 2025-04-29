from openai import OpenAI

prompt="The agent-environment interface of reinforcement learning, where the agent performs actions (noted $a_t$ in LaTeX) on the environment, which in turns changes the state $s_t$ of the agent and provides a reward $r_{t+1}$. The agent is represented by a brain and the environment by a robot enegaged in a task."

client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt=prompt,
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)

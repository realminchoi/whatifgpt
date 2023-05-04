<h1 align="center">
🤖 WhatIfGPT 🤖
</h1>

# Introduction
This project was inspired by [*Generative Agents: Interactive Simulacra of Human Behavior*](https://arxiv.org/abs/2304.03442) paper and evolved through use of LangChain's Experimental Generative Agents [*Generative Agents in LangChain*](https://python.langchain.com/en/latest/use_cases/agent_simulations/characters.html) and [*Multi-Player Dungeons & Dragons*](https://python.langchain.com/en/latest/use_cases/agent_simulations/multi_player_dnd.html).  In this simulation, there is Storyteller Agent, and multiple Generative Agents.  Storyteller Agent is just a narrator/moderator Agent with a set of memory that summarizes the story and Generative Agents' actions thus far.  Generative Agents are extension of LangChain's Experimental Genrative Agents using its time-weighted Memory object backed by LangChain Retriever, with the same memory that maintains the story and other Agents' actions.  The System and Human Messages were tweaked as it was losing focus and context hence why the summary memory was needed.  One of the key concept in these Generative Agents are that it has time-weighted vector store that contains the Agents' memories that impacts its thoughts and actions.  Though this memory is fairly short, it is experimental to see what is possible in this concept.

It starts with setting up the main theme of the story (i.e. Harry Potter, Pirates of the Caribbean, etc.), # of Agents (excluding the Storyteller), and checkbox for random generation or custom Agent names.  When the simulation runs, The Storyteller acts as the narrator and moderator having a dialogue with the Generative Agents.  Both the Storyteller and Generative Agents have a summary memory that keeps track of the main story, the Storyteller's observations, and each Agents' actions.  This was important to maintain as the agents would start to lose focus and start talking out of context.  

Each Generative Agents have the LangChain's Time-weighted Vector Store Retriever that contains certain memories that impacts their thoughts and actions.  This was the bit of the twist in the simulation which adds additional randomness to the story due to the way they act.

![whatifgpt](https://user-images.githubusercontent.com/67872688/236166451-24058391-7d02-4278-9fad-7c93ea94d5d5.svg)

Originally this memory was configurable but I removed it for simplicity for the UI.  It can be added back to be configurable from the UI if you want.  This was fun and great learning experience for me, I hope you enjoy it!

This project is built with
  * [LangChain Python](https://python.langchain.com/en/latest/index.html)
  * [OpenAI API for GPT and DALL-E](https://platform.openai.com/docs/api-reference)
  * [ElevenLabs for Voice](https://beta.elevenlabs.io/)
  * [Streamlit for UI](https://streamlit.io/)


# Demo
https://user-images.githubusercontent.com/67872688/236244415-a338481a-031b-4d30-acb9-a3f502672ba2.mp4
# Installation
Install the required packages:
````
pip install -r requirements.txt
````
## Usage

Run streamlit.
````
python -m streamlit run babyagi.py 
````

You can now view your Streamlit app in your browser (Streamlit also automatically launches your default browser to this location).

Local URL: http://localhost:8501

# Acknowledgments

I would like to express my gratitude to the developers whose code I referenced in creating this repo.

Special thanks go to 

@hwchase17 (https://github.com/hwchase17/langchain)

@mbchang (https://github.com/mbchang)



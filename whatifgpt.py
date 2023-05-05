import base64
import faiss
import json
import math
import os
import openai
import random
from datetime import datetime
from typing import List, Tuple
from elevenlabs import generate, play, clone, save, voices
from playsound import playsound

from PIL import Image
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import FAISS
from langchain.experimental import GenerativeAgent
from langchain.experimental.generative_agents.memory import GenerativeAgentMemory

class Message:
    def __init__(self, name: str, icon, layout: str = 'storyteller'):
        if layout == 'storyteller':
            message_col, icon_col = st.columns([10, 1], gap="medium")
        elif layout == 'agent':
            icon_col, message_col = st.columns([1, 10], gap="medium")
        else:
            raise ValueError("Invalid layout specified. Use 'storyteller' or 'agent'.")

        self.icon = icon
        icon_col.image(self.icon, caption=name)
        self.markdown = message_col.markdown

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def write(self, content):
        self.markdown(content)
        
class StorytellerAgent():

    def __init__(self, name, system_message: SystemMessage, summary_history, story_main_objective, llm: ChatOpenAI,):
        self.name = name
        self.llm = llm
        self.system_message = system_message
        self.summary_history = summary_history
        self.story_main_objective = story_main_objective
        self.prefix = f'\n{self.name}:'
        self.voice = None
        self.icon = "images/the_storyteller.png"
        
    def send(self) -> Tuple[str, bool]:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        summary = (f"Summary thus far: {self.summary_history}" )
        message = self.llm(
            [self.system_message, 
             HumanMessage(content=summary)]).content
        return message, self.is_objective_complete(message)
            
    def receive(self, name: str, message: str) -> None:
        self.summary_history = get_summary_content(self.summary_history, name, message)

    def is_objective_complete(self, message: str) -> bool:
        """
        Checks if objective has been completed
        """
        objective_check_prompt = [
            SystemMessage(content="Determine if objective has been achieved."),
            HumanMessage(content=
                f"""
                Story Objective: {self.story_main_objective}
                Story thus far: {message}
                Based on this "Summary thus far"" has the main "Story Objective" completed? If obtaining item is part of the "Story Objective", is the item(s) in the possession of the characters?
                Only answer with "Yes" or "No", do not add anything else.
                """
                )
        ]
        is_complete = ChatOpenAI(temperature=0.0)(objective_check_prompt).content
        return True if "yes" in is_complete.lower() else False

    def narrate(self, message: str):
        if not os.environ['ELEVEN_API_KEY']:
            return
        """Narrate the observation using to Voice Cloned Storyteller voice, need ElevenLabs"""
        if not self.voice:
            for voice in voices():
                if voice.name == "Storyteller":
                    self.voice = voice
                    break
            else:
                self.voice = clone(
                    name="Storyteller",
                    description="An old British male voice with a strong hoarseness in his throat.  Perfect for story narration",
                    files=["./voices/Storyteller_Narration_Voice.mp3"]
                )
        audio = generate(text=message, voice=self.voice)
        save(audio, "narration.mpeg")
        playsound("narration.mpeg")
        os.remove("narration.mpeg")

class WhatIfGenerativeAgent(GenerativeAgent):
    sex: str
    race: str
    age: int
    story: str
    traits: str
    system_message: SystemMessage = None
    summary_history: str = ""
    icon: str = None
    voice: voice = None
    
    def _compute_agent_summary(self) -> str:
        """"""
        prompt = PromptTemplate.from_template(
            "Please reply with a creative description of the character {name} in 50 words or less, "
            +f"also creatively include the character's traits with the description: {self.traits}."
            +"Also consider {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories}"
            + "Do not add anything else."
            + "\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        return (
            self.chain(prompt)
            .run(name=self.name, queries=[f"{self.name}'s core characteristics"])
            .strip()
        )

    def get_stats(self, force_refresh: bool = False) -> str:
        """Return the character stats of the agent."""
        current_time = datetime.now()
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        return (
            f"Age: {self.age}"
            f"\nSex: {self.sex}"
            f"\nRace: {self.race}"
            f"\nStatus: {self.status}"
            +f"\nInnate traits: {self.traits}\n"
        )

    def get_summary_description(self, force_refresh: bool = False) -> str:
        """Return a short summary of the agent."""
        current_time = datetime.now()
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        return (f"\n{self.summary}\n"
        )
        
    def _generate_reaction(self, observation: str, system_message: SystemMessage) -> str:
        """React to a given observation or dialogue act but with a Character Agent SystemMessage"""
        human_prompt = HumanMessagePromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
        )
        prompt = ChatPromptTemplate.from_messages([system_message, human_prompt])
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return self.chain(prompt=prompt).run(**kwargs).strip()

    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        story_summary_current = self.summary_history + "\n" + observation
        result = self._generate_reaction(story_summary_current, self.system_message)
        # Save Context to Agent's Memory
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result}"
            },
        )
        return result

    def setup_agent(self, system_message: SystemMessage, specified_story: str):
        """Sets the Agent post Story and Main Objective gets set"""
        self.system_message = system_message
        self.memory.add_memory(specified_story)
        self.summary_history = specified_story

    def receive(self, name: str, message: str) -> None:
        """Receives the current observation and summarize in to summary history"""
        self.summary_history = get_summary_content(self.summary_history, name, message)

    def narrate(self, message: str):
        """Narrate using ElevenLabs"""
        if not os.environ['ELEVEN_API_KEY']:
            return
        if not self.voice:
            for voice in voices():
                if voice.name == self.name:
                    self.voice = voice
                    break
            else:
                if self.name.lower() in ["harry potter", "hermione granger", "ron weasley"]:
                    self.voice = clone(
                        name=self.name,
                        description=f"voice clone of {self.name}-like voice",
                        files=[f"./voices/{self.name}_Narration_Voice.mp3"]
                    )
                else:
                    male_voices = ["Antoni", "Josh", "Arnold", "Adam", "Sam"]
                    female_voices = ["Rachel", "Bella", "Elli" ]
                    for voice in voices():
                        if self.sex.lower() == "male":
                            if voice.name == random.choice(male_voices):
                                self.voice = voice
                                break
                        else:
                            if voice.name == random.choice(female_voices):
                                self.voice = voice
                                break
        audio = generate(text=message, voice=self.voice)
        save(audio, f"{self.name}.mpeg")
        playsound(f"{self.name}.mpeg")
        os.remove(f"{self.name}.mpeg")

class WhatIfStorySimulator():
    def __init__(self, story, mood, num_agents, is_random, agent_names, story_setting_event):
        self.story = story
        self.mood = mood
        self.num_agents = num_agents
        self.is_random = is_random
        self.agent_names = agent_names
        self.story_setting_event = story_setting_event

    def generate_agent_character(self, agent_num, story: str, mood: str, **kwargs):
        """Generate a Character Agent."""
        name = kwargs["name"]
        age = kwargs["age"]
        sex = kwargs["sex"]
        race = kwargs["race"]
        st.markdown(f":blue[A wild **_{name}_** appeared.]")
        icon_prompt = (f"{age} years old {sex} {race} named {name} from {story}, portrait, 16-bit super nes")
        response = openai.Image.create(
            prompt=icon_prompt,
            n=1,
            size="256x256",
            response_format="b64_json"
        )
        binary_data = base64.b64decode(response["data"][0]["b64_json"])
        icon_file = f"images/agent{str(agent_num)}.png"
        with open(icon_file, "wb") as file:
            file.write(binary_data)
        gen_agent = WhatIfGenerativeAgent(
            icon=icon_file,
            name=name,
            age=kwargs["age"],
            race=kwargs["race"],
            sex=kwargs["sex"],
            story=story,
            traits=kwargs["traits"],
            status=kwargs["status"],
            memory=GenerativeAgentMemory(llm=ChatOpenAI(), memory_retriever=create_new_memory_retriever()),
            llm=ChatOpenAI(model_name=os.environ['OPENAI_API_MODEL'], temperature=float(os.environ['OPENAI_TEMPERATURE'])),
            daily_summaries=[str(x) for x in kwargs["daily_summaries"]],
        )
        portrait_area, stats_area = st.columns([1,3])
        with portrait_area:
            st.image(icon_file)
        with stats_area:
            st.markdown(f"Sex: :blue[{gen_agent.sex}]")
            st.markdown(f"Race: :blue[{gen_agent.race}]")
            st.markdown(f"Status: :blue[{gen_agent.status}]")
            st.markdown(f"traits: :blue[{gen_agent.traits}]")
        for memory in [str(x) for x in kwargs["memories"]]:
            gen_agent.memory.add_memory(memory)
        summary_description = gen_agent.get_summary_description(force_refresh=True)
        st.markdown(f"Summary: :green[{summary_description}]")

        return gen_agent

    def generate_random_character(self, story: str, mood: str, agent_names: list):
        """ Generate random character with properties """
        character_exclusion = f" that is not in [{', '.join(agent_names)}]" if agent_names else ""
        prompt = (
            f"Generate a random {story} character {character_exclusion}. "
            "Based on the character possessing some basic memories and events, "
            "provide the following properties in JSON format:\n"
            "name: Name of the character\n"
            "race: Race of the character\n"
            "sex: The character's sex\n"
            "age: The character's age\n"
            "traits: 3 to 8 traits that describe the character (comma-separated)\n"
            f"status: The character's current status in the perspective of {story}\n"
            f"daily_summaries: 5 to 10 {mood}-themed daily activities that the character completed today (array of strings)\n"
            f"memories: 5 to 10 {mood}-themed memories from the character's life (array of strings)\n"
        )
        return json.loads(
            ChatOpenAI(model_name=os.environ['OPENAI_API_MODEL'], temperature=1.0)(
                [HumanMessage(content=prompt)]
            ).content
        )

    def generate_random_props(self, story: str, mood: str, name: str):
        """ Generate random character properties """
        prompt = (
            f"Based on the {story} character {name} possessing some basic memories and events, "
            "provide the following properties in JSON format:\n"
            "name: Name of the character\n"
            "race: Race of the character\n"
            "sex: The character's sex\n"
            "age: The character's age\n"
            "traits: 3 to 8 traits that describe the character (comma-separated)\n"
            f"status: The character's current status in the perspective of {story}\n"
            f"daily_summaries: 5 to 10 {mood}-themed daily activities that the character completed today (array of strings)\n"
            f"memories: 5 to 10 {mood}-themed memories from the character's life (array of strings)\n"
        )
        return json.loads(
            ChatOpenAI(model_name=os.environ['OPENAI_API_MODEL'], temperature=1.0)(
                [HumanMessage(content=prompt)]
            ).content
        )

    def generate_character_system_message(self, story_description, character_name, character_description):
        """Generate System Message for Generative Agents"""
        return (SystemMessage(content=(
            f"""{story_description}
                Your name is {character_name}. 
                Your character description is as follows: {character_description}.
                You will speak what specific action you are taking next and try not to repeat any previous actions
                Speak in the first person from the perspective of {character_name}, in the tone that {character_name} would speak.
                Do not change roles!
                Do not speak from the perspective of anyone else.
                Remember you are {character_name}.
                Stop speaking the moment you finish speaking from your perspective.
                Never forget to keep your response to {word_limit} words!
                Do not add anything else.
            """)
        ))

    def generate_storyteller_system_message(self, story_description, storyteller_name):
        """Generate the System Message for Storyteller"""
        return (SystemMessage(content=(
            f"""{story_description}
                You are the storyteller, {storyteller_name}.
                Taking the character's actions into consideration you will narrate and explain what happens when they take those actions then narrate in details what must be done next.
                Narrate in a creative and captivating manner.  Do not repeat anything that has already happened.
                Do not change roles!
                Do not speak from the perspective of anyone else.
                Remember you are the storyteller, {storyteller_name}.
                Stop speaking the moment you finish speaking from your perspective.
                Never forget to keep your response to 50 words!
                Do not add anything else.
            """)
        ))

    def generate_agents(self, story, mood, num_agents, agent_names, is_random):
        """Generate Agents"""
        agents = []
        for i in range(num_agents):
            with st.spinner(f"Generating {story} Character Agent"):
                kwargs = self.generate_random_character(story, mood, agent_names) if is_random else self.generate_random_props(story, mood, agent_names[i])
                agent = self.generate_agent_character(i+1, story=story, mood=mood, **kwargs)
            agents.append(agent)
            agent_names.append(agent.name)

        return agents

    def define_story_details(self, story, agent_names, story_setting_event):
        """Define Story Details with Main Objective"""
        story_description = f"""This is based on {story}.
            The characters are: {', '.join(agent_names)}.
            Here is the story setting: {story_setting_event}"""
        story_specifier_prompt = [
            SystemMessage(content="You can make tasks more specific."),
            HumanMessage(content=
                f"""{story_description}
                Narrate a creative and thrilling background story that has never been told and sets the stage for the main objective of the story.
                The main objective must require series of tasks the characters must complete.
                If the main objective is item or person, narrate a creative and cool name for them.
                Narrate specific detail what is the next step to embark on this journey.
                No actions have been taken yet by {', '.join(agent_names)}, only provide the introduction and background of the story.
                Please reply with the specified quest in 100 words or less. 
                Speak directly to the characters: {', '.join(agent_names)}.
                Do not add anything else."""
                )
        ]
        with st.spinner(f"Generating Story"):
            specified_story = ChatOpenAI(model_name=os.environ['OPENAI_API_MODEL'], temperature=1.0)(story_specifier_prompt).content

        story_main_objective_prompt = [
            SystemMessage(content="Identify main objective"),
            HumanMessage(content=
                f"""Here is the story: {specified_story}
                What is the main objective of this story {', '.join(agent_names)}?  Narrate the response in one line, do not add anything else."""
                )
        ]
        with st.spinner(f"Extracting Objective"):
            story_main_objective = ChatOpenAI(model_name=os.environ['OPENAI_API_MODEL'], temperature=0.0)(story_main_objective_prompt).content
        return story_description, specified_story, story_main_objective

    def initialize_storyteller_and_agents(self, agent_names, story_description, specified_story, story_main_objective, agents):
        """Initialize Storyteller and Agents"""
        storyteller = StorytellerAgent(
            name=storyteller_name,
            llm=ChatOpenAI(model_name=os.environ['OPENAI_API_MODEL'], temperature=0.5),
            system_message=self.generate_storyteller_system_message(specified_story, storyteller_name),
            summary_history=specified_story,
            story_main_objective=story_main_objective
        )
        for agent in agents:
            agent.setup_agent(
                self.generate_character_system_message(story_description, agent.name, agent.get_summary_description()),
                specified_story
            )
        return storyteller, agents

    def generate_story_finale(self, story_main_objective, final_observation):
        """Generate a Cliffhanger Finale"""
        story_finale_prompt = [
            SystemMessage(content="Make the finale a cliffhanger"),
            HumanMessage(content=
                f"""
                Story Objective: {story_main_objective}
                Final Observation: {final_observation}
                Based on this "Story Objective" and "Final Observation", narrate a grand finale cliffhanger ending.
                Be creative and spectacular!
                """
                )
        ]
        story_finale = ChatOpenAI(model_name=os.environ['OPENAI_API_MODEL'], temperature=1.0)(story_finale_prompt).content
        return story_finale

    def run_story(self, storyteller: StorytellerAgent, agents: List[WhatIfGenerativeAgent], observation: str) -> Tuple[str, int]:
        """Runs the Story"""

        is_objective_complete = False
        turns = 0
        prev_agent = None
        while True:
            random.shuffle(agents)

            for chosen_agent in agents:
                while chosen_agent == prev_agent:
                    chosen_agent = random.choice(agents)
                prev_agent = chosen_agent

                with st.spinner(f"{chosen_agent.name} is reacting"):
                    reaction = chosen_agent.generate_reaction(observation)
                with Message(chosen_agent.name, chosen_agent.icon, layout='agent') as m:
                    m.write(f"{reaction}")
                chosen_agent.narrate(reaction)

                with st.spinner(f"Agents are observing"):
                    for recipient in agents + [storyteller]:
                        recipient.receive(chosen_agent.name, reaction)

                with st.spinner(f"{storyteller.name} is thinking"):
                    observation, is_objective_complete = storyteller.send()
                turns += 1
                if is_objective_complete:
                    return observation, turns
                with Message(storyteller.name, storyteller.icon, layout='storyteller') as m:
                    m.write(f":green[{observation}]")
                storyteller.narrate(observation)

    def run_simulation(self):
        self.agents = self.generate_agents(self.story, self.mood, self.num_agents, self.agent_names, self.is_random)
        story_description, specified_story, story_main_objective = self.define_story_details(self.story, self.agent_names, self.story_setting_event)
        self.storyteller, self.agents = self.initialize_storyteller_and_agents(self.agent_names, story_description, specified_story, story_main_objective, self.agents)
        with Message(self.storyteller.name, self.storyteller.icon, layout='storyteller') as m:
            m.write(f":green[{specified_story}]")
        self.storyteller.narrate(specified_story)
        final_observation, turns = self.run_story(self.storyteller, self.agents, specified_story)
        story_finale = self.generate_story_finale(story_main_objective, final_observation)

        with Message(self.storyteller.name, self.storyteller.icon, layout='storyteller') as m:
            m.write(f":green[{story_finale}]")
        self.storyteller.narrate(story_finale)
        st.success(f"Story Objective completed in {turns} turns!", icon="✅")

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)  

def get_summary_content(summary_history, name, message) -> str:
    """Summarize What has happened thus far"""
    summarizer_prompt = [
        SystemMessage(content="Make the summary concise."),
        HumanMessage(content=
            f"""Summarize the following into a concise summary with key details including the actions that {name} has taken and the results of that action
            {summary_history}
            {name} reacts {message}
            """
            )
    ]
    return ChatOpenAI(temperature=0.0)(summarizer_prompt).content

storyteller_name = "The Storyteller"
word_limit = 35

def main():
    st.set_page_config(
        initial_sidebar_state="expanded",
        page_title="WhatIfGPT",
        layout="centered",
    )

    with st.sidebar:
        openai_api_key = st.text_input("Your OpenAI API KEY", type="password")
        openai_api_model = st.selectbox("Model name", options=["gpt-3.5-turbo", "gpt-4"])
        openai_temperature = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.2,
        )
        eleven_api_key = st.text_input("Your Eleven Labs API Key", type="password")

    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['OPENAI_API_MODEL'] = openai_api_model
    os.environ['OPENAI_TEMPERATURE'] = str(openai_temperature)
    os.environ['ELEVEN_API_KEY'] = eleven_api_key
    st.title("WhatIfGPT")
    story = st.text_input("Enter the theme of the story", "Random Story")
    mood = "positive"
    num_agents = st.slider(
        label="Number of Agents",
        min_value=2,
        max_value=4,
        step=1,
        value=2,
    )
    is_random = st.checkbox("Do you want the event and agents to be created randomly?", value=True)
    agent_names = []
    story_setting_event = f"random entertaining story with a mission to complete in the theme of {story}"
    if not is_random:
        for i in range(num_agents):
            name = st.text_input(f"Enter Character {i + 1} name: ", "")
            agent_names.append(name)
        user_story_setting_event = st.text_input("Enter the story to have the agents participate in (or just leave blank for random): ")
        if user_story_setting_event:
            story_setting_event = user_story_setting_event
    button = st.button("Run")
    if button:
        try:
            whatifsim = WhatIfStorySimulator(
                story, 
                mood, 
                num_agents, 
                is_random, 
                agent_names, 
                story_setting_event
            )
            whatifsim.run_simulation()
        except Exception as e:
            st.error(e)
if __name__ == "__main__":
    main()

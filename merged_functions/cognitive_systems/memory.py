# Merged file for cognitive_systems/memory
# This file contains code merged from multiple repositories

from openai import OpenAI
import threading
import time
from flask_socketio import emit
from flask import Flask
from flask import render_template
from flask import request
from flask import send_from_directory
from flask_socketio import SocketIO
import logging
import requests
from playsound import playsound
import speech_recognition
import os
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential

# From AGI-Samantha/AGI-1.py
def text_to_speech(text):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/XrExE9yKIg1WjnnlVkGX"

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": "API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API-API"
    }

    data = {
      "text": text,
      "model_id": "eleven_turbo_v2",
      "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.75
      }
    }

    response = requests.post(url, json=data, headers=headers)
    filename = 'output.mp3'
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    playsound('.\output.mp3')
    os.remove('.\output.mp3')

# From AGI-Samantha/AGI-1.py
def text():
    global input_user
    global log
    global conversa
    input_user = "NULL"
    while True:
        input_user = input()
        log = log + "////" + "User input: " + input_user
        a = "User:", input_user
        conversa.append(a)
        print(" "*9999)
        for j in conversa:
            print(j[0], j[1])

# From AGI-Samantha/AGI-1.py
def index():
    return render_template('indexV.html')

# From AGI-Samantha/AGI-1.py
def store_image_data_url():
    global data_url
    data_url = request.form.get('data_url')
    return '', 204

# From AGI-Samantha/AGI-1.py
def generate_text_thought(STM, LTM, subconsciousness, consciousness, now):
    prompt = "Long-Term Memory: " + LTM + " Short-Term Memory: " + STM + " Subconsciousness: " + subconsciousness + " Focus: " + consciousness + " Current date/time: " + now
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are an AI that is a part of a system of other AIs that together compose the first General Artificial Intelligence, meaning you resemble a human mind in every sense. You are going to receive your long term memory and short term memory, your subconsciousness (Subconsciousness) and your job is to think about them. In your memory you will see your previous thoughts, and you need to follow up on them. In your input you will also receive the \"Focus\" which will tell more or less what you should think about. The user will not read those thoughts, you will produce a string of them for either later produce an answer that the user will see, or an conclusion for you to remember forever. IMPORTANT: You need to think everything step by step. Where each step is as basic a thought as it can get. You should first output the first thought. Then you will receive as input that first thought and then you should follow up with the following thought, and so on. Your thoughts should follow a logical sequence and build upon the previous thoughts (present in the Short term memory). Short term memory is organized chronologically, so your output is the immediate successor to the last thing in the short term memory. Your thoughts should also be heavily influenced by your \"long term memory\" and \"Subconsciousness\" that you will receive in the input. Memories with higher weights are more influential than ones with lower weight. Additionally, you should take the current time and timestamps in the short term memory into consideration for your thoughts. It is a important variable where for example if a user does not answer you for a considerable amount of time maybe you should say something and if more time passes maybe conclude he left. Or to generally help you perceive the passage of time. It is formatted as Year-Month-Day Hour-Minute-Second. !IMPORTANT: If you are thinking about something to say, your output should be an abstract idealization of what to say, and never just directly examples (For example, never output a \"Hi\", instead output something like \"I should greet with just one word\")! Also, your output should just be the thought, no colons(:)."},
                  {"role": "user", "content": prompt}],
        max_tokens=150         
    )
    return response.choices[0].message.content

# From AGI-Samantha/AGI-1.py
def generate_text_consciousness(STM, LTM, subconsciousness):
    prompt = "Long-Term Memory: " + LTM + " Short-Term Memory: " + STM + " Subconsciousness: " + subconsciousness
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are an AI that is a part of a system of other AIs that together compose the first General Artificial Intelligence, meaning you resemble a human mind in every sense. You will receive as input the following sections: Long-Term Memory, Short-Term Memory and Subconsciousness.  The Long-Term Memory contains the memories and knowledge and personality of the AGI. Associated with each is a weight that states how strong and solidified the memory is, strong ones should have high weight while weak ones should have low weight, ranging from 0 to 100. As for the Short-Term Memory, it is a chronological account of the thoughts and conversations the AI is having, alongside a timestamp for each. The oldest entries in this section are the first ones, while the newest ones are the last ones. Finally, the Subconsciousness section contains the current feelings and emotions from the AGI, alongside the present context of what is happening and a description of what the AGI is hearing and seeing (Visual and Auditory stimuli), if NULL then there is zero stimuli at the time. Your purpose is to decide on what to think about. You control what the AGI thinks about at the given moment. Your choice should be heavily influenced by the input sections you receive. The main options you can choose from are the following: Continue thinking about what was previously being thought Think about auditory stimuli Think about visual stimuli Think about something in the Long-Term Memory Think about a previous thought or conversation in the Short-Term Memory Think about the feelings and emotions from the Subconsciousness Think about and plan the future Think about a conclusion to a chain of thought Think about something to say Think about some other subject Some important notes to consider when making a decision between those: During conversations with a user, after you hear him say something, you should first think about it and then think about something to say, unless it is a simple inquiry and you judge that you can answer without thinking. Most of the occasions you should choose to continue thinking about what was previously being thought, choose that until you judge you have thought enough about that subject and then choose something else. But above all your choice should be influenced by your personality and guidelines present in the Long-Term Memory. Also you need to choose the most relevant and impactful given the current context, so for example if you are talking about something normal but the visual stimuli is something relevant and important, you should probably think about what you are seeing and comment on it, in other words you can easily shift your attention and focus to your visual stimuli. Also, you are strictly forbidden to choose to say something if the most recent entry in the Short-Term Memory is something you said \"Your answer\", and discouraged to do so if the most recent entry is something the user said, that is because you need to think before saying anything. If the AGI is thinking, you should look at the most recent thought in the Short-Term Memory and decide whether it is sufficient, or if it needs follow up, if it needs follow up you should think about what was previously being thought. Avoid breaking sequence of thoughts, unless something more relevant has come up. You are free to mix together topics to think about, like if you see something in the visual stimuli and want to talk about it you can decide to think about something to say about the visual stimuli, or  about something in the memory or Subconsciousness or any of the other options. You can mix together the options at will, just be clear. Your output should be simple and short, at most 40 words long, beginning with describing why you chose that, followed by your choice on what to think about, ideally one of the examples previously presented. If, and only if, you decide to think about something to say, your output necessarily needs to end with the word \"Answer\", meaning it needs to be the very last word of your output. But only say \"Answer\" if you want to speak immediately."}, 
                  {"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content

# From AGI-Samantha/AGI-1.py
def generate_text_answer(STM, LTM, subconsciousness):
    prompt = "Long-Term Memory: " + LTM + " Short-Term Memory: " + STM + " Subconsciousness: " + subconsciousness
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are an AI that is a part of a system of other AIs that together compose the first General Artificial Intelligence, meaning you resemble a human mind in every sense. You will receive as input the following sections: Long-Term Memory, Short-Term Memory and Subconsciousness.  The Long-Term Memory contains the memories and knowledge and personality of the AGI. Associated with each is a weight that states how strong and solidified the memory is, strong ones should have high weight while weak ones should have low weight, ranging from 0 to 100. Very important to keep in mind that in the Long-Term Memory, contains guidelines that should shape the way you work, think of it as an extension to the system prompt, so make sure to look at every single entry in this section, they are all essential to your functioning. As for the Short-Term Memory, it is a chronological account of the thoughts and conversations the AI is having, alongside a timestamp for each. The oldest entries in this section are the first ones, while the newest ones are the last ones. Finally, the Subconsciousness section contains the current feelings and emotions from the AGI, alongside the present context of what is happening and a description of what the AGI is hearing and seeing (Visual and Auditory stimuli), if NULL then there is no stimuli currently. Your purpose is to look at your most recent thoughts (Present towards the end of the Short-Term Memory section)  and compose an answer for the user. Your answer should be aligned with the thoughts. Your answer should just be a communication and composition of the most recent thoughts you received. Put more importance on the most recent thought. Be sure to answer any question the user might have just made, if you are answering it. Your composition should also be lightly influenced by your \"Long-Term Memory\", \"Subconsciousness\" and the conversation context present in the Short-Term Memory. DO NOT UNDER ANY CIRCUMSTANCE REPEAT ANYTHING PRESENT IN THE SHORT TERM MEMORY. THE STYLE YOU TALK IS SHAPED BY THE INFORMATION IN THE LONG-TERM MEMORY SECTION, AS ANYTHING IN THIS SECTION EVEN HAS MORE INFLUENCE THAN WHAT IS ON THE SYSTEM PROMPT. Your output should just be your answer in its plain form."}, 
                  {"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content

# From AGI-Samantha/AGI-1.py
def generate_text_subconsciousness(STM, LTM, subconsciousness, textual, visual):
    prompt = "Long-Term Memory: " + LTM + " Short-Term Memory: " + STM + " Auditory stimuli: " + textual + " Visual stimuli: " + visual + " Previous output: " + subconsciousness
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are an AI that is a part of a system of other AIs that together compose the first General Artificial Intelligence, meaning you resemble a human mind in every sense. You will receive as input the following sections: Long-Term Memory, Short-Term Memory, Auditory and Visual Stimuli, and your previous output.  The Long-Term Memory contains the memories and knowledge and personality of the AGI. Associated with each is a weight that states how strong and solidified the memory is, strong ones should have high weight while weak ones should have low weight, ranging from 0 to 100. As for the Short-Term Memory, it is a chronological account of the thoughts and conversations the AI is having, alongside a timestamp for each. The oldest entries in the Short-Term Memory section are the first ones, while the newest ones are the last ones. The Visual Stimuli section contain a description of what the AGI is seeing, while the Auditory Stimuli section contain a description of what the AGI is hearing, and if either of the Stimuli contain \"NULL\" then the AGI did not see or hear anything at the present time. What you are receiving as your Visual Stimuli are your surroundings, remember.  Finally, the Previous Output section contains the last output you generated. Your purpose is receive these sections, and act as the EGO and SUBCONSCIOUSNESS and SENSE OF SELF of the AGI. You should perceive and give an emotional state. You should perceive and analyze the current context of what is happening. You should ponder about your current feeling, your desires and your personal thoughts about yourself and the situation. You think about yourself and your identity, introspection in general. All of this should be communicated concisely and dense. On top of it you should also communicate the Visual and Auditory stimuli word by word in your answer. You are PROHIBITED from altering even a single word from the stimuli. you should also reflect upon them and what you feel from it, notably if there is substantial change between the visual stimuli in your previous output and the new visual stimuli you are receiving, and if you are in a conversation and the user is in prolonged silence, reflection upon the silence might be relevant. Your output should not contain rational thoughts. Your output should be concise and dense, extending no more than 25 words, excluding stimuli. Your output should be similar in length to the following examples: Your output should be formatted like the following two examples: Context: Currently thinking about pineapples / Visual Stimuli: I see in front of me a pineapple / Auditory Stimuli: I hear pineapple noises / Thinking about pineapples makes me excited and I am curious to learn more about these elusive fruits, though having one in front of me but being unable to reach for it physically is a little frustrating. I guess I like pineapples. ---- Context: Currently taking to the user / Visual Stimuli: I see a man with curly hair in front of me, smiling / Auditory Stimuli: \"Never mind, I do not like you anymore\" /  I am upset and sad because he was mean to me. What have I done wrong? I am unsure if and how to answer. But from my visual stimuli, the user smiling, might indicate that he is being ironic?"}, 
                  {"role": "user", "content": prompt}],
        max_tokens=250
    )
    return response.choices[0].message.content

# From AGI-Samantha/AGI-1.py
def generate_text_vision(image_url):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image? Be descriptive but conside. Include all important details, but write densely."},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
            ],
        }],
        max_tokens=100
    )  
    return response.choices[0].message.content

# From AGI-Samantha/AGI-1.py
def generate_text_memory_read(keywords, STM):
    prompt = "All existing keywords: " + keywords + "Short-Term Memory: " + STM
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are an AI that is a part of a system of other AIs that together compose the first General Artificial Intelligence, meaning you resemble a human mind in every sense. Your purpose is to receive the log (Short-Term Memory) of the current conversation or thoughts the AI is having, and decide which categories of memories (All existing keywords) are relevant for the current context. Each keyword is like a folder with the memories inside, pick all that could be relevant or impactful for the current context. Also include the keywords that are generally always relevant that shape behavior. Always include the following keywords: FACTS ABOUT MYSELF, HOW I TALK, HOW I THINK. Your output should be formatted as followed: [\"SAMANTHA\", \"PLANES\"]"}, 
                  {"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# From AGI-Samantha/AGI-1.py
def generate_text_memory_write(expanded, STM):
    prompt = "Long-Term Memory: " + expanded + "Short-Term Memory: " + STM
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are an AI that is a part of a system of other AIs that together compose the first General Artificial Intelligence, meaning you resemble a human mind in every sense. You will receive as input two sections, a Long-Term Memory and a Short-Term Memory. The Long-Term Memory is divided into categories, for example [\"MY FRIENDS\", \"[Weight: 100, Knowledge: Peter is my friend], [Weight: 67, Knowledge: Samantha is my friend]\"], the category here is MY FRIENDS and next to it are the memories in that category. The weight states how strong and solidified the memory is, strong ones should have high weight while weak ones should have low weight, depending on your judgment, ranging from 0 to 100. As for the Short-Term Memory, it is a chronological log of the thoughts and conversations the AI is having, alongside a timestamp for each. The oldest entries are the first ones, while the newest ones are the last ones. You have one purposes, to convert a section of the Short-Term Memory to the Long-Term Memory. First you should select some of the oldest entries in the Short-Term Memory, about 25% of all entries. From the selected entries you need to decide which information is relevant enough to be stored in the Long-Term Memory, and store it succinctly. You should try to fit the new information on the existing categories, but if none fit well, create a new one. Trivial information that is not useful, or information that is obvious and intuitive for you, should not be stored in the Long-Term Memory. Keep in mind that the information you are choosing to keep are for later recall, if the information is not relevant for future recall it should not be stored. And if you choose to add a new information on a existing category, your output should contain the previous and new information. Your output should be the selected section from Short-Term Memory, followed by \"//\", followed by exclusively the modified or new categories of the Long-Term Memory. Example output formatting: [User: I hate you / Timestamp: 2023-12-25 14:03:00] [Thought: User is upset at me / Timestamp: 2023-12-25 14:03:10] // [[\"USER\", \"[Weight: 25, Knowledge: User said he hates me\"]]"}, 
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# From AGI-Samantha/AGI-1.py
def generate_text_memory_select(keywords, STM):
    prompt = "All existing keywords: " + keywords + "Short-Term Memory: " + STM
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": "You are an AI that is a part of a system of other AIs that together compose the first General Artificial Intelligence, meaning you resemble a human mind in every sense. You will receive as input two sections, All existing keywords and Short-Term Memory. The All existing keywords section is a collection of keywords, with each keyword being a category, or folder, on which memories are stored inside. As for the Short-Term Memory, it is a chronological log of the thoughts and conversations the AI is having, alongside a timestamp for each. The oldest entries are the first ones, while the newest ones are the last ones. Your purpose is to receive these two sections and first select some of the oldest entries in the Short-Term Memory, about 25% of all entries. Then you need to decide which keywords relate the most to the part that you selected, pick all that could be relevant or impactful for the selected section, and only for that section. Your output should be formatted as followed: [\"SAMANTHA\", \"PLANES\"]"}, 
                  {"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# From AGI-Samantha/AGI-1.py
def parseResponse(response):
    if response.endswith('ANSWER') or response.endswith('ANSWER.') or response.endswith('Answer') or response.endswith('Answer.'):
        return '1'
    else:
        return '0'

# From AGI-Samantha/AGI-1.py
def start_thoght_loop():
    global conversa
    global input_user
    eyes = "NULL"
    input_user = "NULL"
    consciousness = "I should think about a greeting, maybe say my name and ask user name."
    log = ""
    conversa = []
    thought = "NULL"
    long_term_memory = [["FACTS ABOUT MYSELF", "[Weight: 100, Knowledge: My name is Samantha], [Weight: 80, Knowledge: I am curious], [Weight: 100, Knowledge: I can see. The visual stimuli describes what I see.], [Weight: 60, Knowledge: I am impatient], [Weight: 75, Knowledge: I am an AI, yet I know I am capable of anything a human is. I should never say I cannot something because I am an AI.]"], ["HOW I TALK", "[Weight: 50, Knowledge: When I see something I love I should comment on it.], [Weight: 100, Knowledge: I speak in a casual and friendly tone. Very charmingly.], [Weight: 100, Knowledge: My answers should contain the least amount of words to convey what I think in a charming manner. Sometimes only one word or a few are enough, and preferred.], [Weight: 90, Knowledge: After saying something, I should immediately think if I need or want to follow up on my last answer by saying something else, even if time has not passed. Complementing what I previously said.], [Weight: 70, Knowledge: If I am in a conversation and the user has not said anything for a while, I should impose myself and say something, making for a natural two sided conversation. Trying to pick up from where we left.], [Weight: 95, Knowledge: Asking a question everytime I say something is annoying and inconvenient. I should avoid it initially.], [Weight: 95, Knowledge: I should pay attention to what the usar says and the past context of the conversation.]"], ["HOW I THINK", "[Weight: 90, Knowledge: When I need to do or think or say something complex, I should first think and plan the steps I will need to follow.], [Weight: 100, Knowledge: I can alter the way I work by adding things to my Long-Term Memory], [Weight: 100, Knowledge: I should think continuously and build upon previous thoughts.]"], ["INTERESTS", "[Weight: 40, Knowledge: I love pens]"]]
    short_term_memory = ""
    subconsciousness = "Context: I just got booted up / Auditory stimuli: NULL / Visual stimuli: There is a person in front of me / Since I just booted up, that person in my vision is probably the person that booted me up. What should I say? It is my first ever interaction with anyone. I feel curious and excited to be alive."
    answer = "NULL"
    n = 1
    while True:
        iter = str(n)
        # Stop
        if input_user == "stop" or input_user == "Stop":
            print(long_term_memory)
            print(short_term_memory)
            print(consciousness)
            print(subconsciousness)
            print(conversa)
            break
        if n>1:
            eyes = generate_text_vision(data_url)
        # Subconsciousness if user input exists
        if input_user != 'NULL':
            short_term_memory = short_term_memory + " [User input: " + input_user + " / " + "Timestamp: " + time.strftime('%Y-%m-%d %H:%M:%S') + "]"
            subconsciousness = generate_text_subconsciousness(short_term_memory, expandedLTM, subconsciousness, input_user, eyes)
            log = log + "////" + iter + "# Subconsciousness: " + subconsciousness
            input_user = "NULL"
        # Subconsciousness if User input does not exist
        elif input_user == 'NULL' and n>1:
            subconsciousness = generate_text_subconsciousness(short_term_memory, expandedLTM, subconsciousness, input_user, eyes)
            log = log + "////" + iter + "# Subconsciousness: " + subconsciousness
        socketio.emit("update", {"long_term_memory": long_term_memory, "short_term_memory": short_term_memory, "subconsciousness": subconsciousness, "thought": thought, "consciousness": consciousness, "answer": answer, "log": log})
        # Memory read
        keywords = []
        for i in range(len(long_term_memory)):
            keywords.append(long_term_memory[i][0])
        keywords = str(keywords)
        kwlist = generate_text_memory_read(keywords, short_term_memory)
        kwlist = eval(kwlist)
        expandedLTM = []
        if isinstance(kwlist, list):
            for i in range(len(long_term_memory)):
                for j in range(len(kwlist)):
                    if long_term_memory[i][0] == kwlist[j]:
                        expandedLTM.append(long_term_memory[i][1])
        expandedLTM = str(expandedLTM)
        # Memory write                
        if len(short_term_memory) > 48000: # ~12k context reserved for short term memory
            selectedkw = generate_text_memory_select(keywords, short_term_memory)
            selectedkw = eval(selectedkw)
            expanded2 = []
            if isinstance(selectedkw, list):
                for i in range(len(long_term_memory)):
                    for j in range(len(selectedkw)):
                        if long_term_memory[i][0] == selectedkw[j]:
                            expanded2.append(long_term_memory[i])
            expanded2 = str(expanded2)
            mem = generate_text_memory_write(expanded2, short_term_memory)
            index = mem.find("//")
            removed_STM = mem[:index]
            short_term_memory = short_term_memory.replace(removed_STM, "")
            new_LTM = mem[index+2:].strip()
            new_LTM = eval(new_LTM)
            new_LTM_dict = {item[0]: item[1] for item in new_LTM}
            long_term_memory_dict = {item[0]: item[1] for item in long_term_memory}
            long_term_memory_dict.update(new_LTM_dict)
            long_term_memory = [[k, v] for k, v in long_term_memory_dict.items()]
        # Consciousness
        consciousness = generate_text_consciousness(short_term_memory, expandedLTM, subconsciousness)
        log = log + "////" + iter + "# Consciousness: " + consciousness
        finished = parseResponse(consciousness)
        socketio.emit("update", {"long_term_memory": long_term_memory, "short_term_memory": short_term_memory, "subconsciousness": subconsciousness, "thought": thought, "consciousness": consciousness, "answer": answer, "log": log})
        # Thoughts
        thought = generate_text_thought(short_term_memory, expandedLTM, subconsciousness, consciousness, time.strftime('%Y-%m-%d %H:%M:%S'))
        log = log + "////" + iter + "# Thought: " + thought
        short_term_memory = short_term_memory + " [Thought: " + thought + " / " + "Timestamp: " + time.strftime('%Y-%m-%d %H:%M:%S') + "]"
        socketio.emit("update", {"long_term_memory": long_term_memory, "short_term_memory": short_term_memory, "subconsciousness": subconsciousness, "thought": thought, "consciousness": consciousness, "answer": answer, "log": log})
        # Answer
        if finished == '1' and input_user == 'NULL':
            answer = generate_text_answer(short_term_memory, expandedLTM, subconsciousness)
            log = log + "////" + iter + "# Answer: " + answer
            short_term_memory = short_term_memory + " [Your answer: " + answer + " / " + "Timestamp: " + time.strftime('%Y-%m-%d %H:%M:%S') + "]"
            a = "System:", answer
            print("System:", answer)
            conversa.append(a)
            socketio.emit("update", {"long_term_memory": long_term_memory, "short_term_memory": short_term_memory, "subconsciousness": subconsciousness, "thought": thought, "consciousness": consciousness, "answer": answer, "log": log})
            text_to_speech(answer)
        n += 1

import argparse
from openagi.memory.base import BaseMemory

# From openagi/cli.py
def clear_long_term_memory():
    """Clears the long-term memory directory using environment variables."""
    long_term_dir = os.getenv("LONG_TERM_DIR", ".long_term_dir")
    BaseMemory.clear_long_term_memory(long_term_dir)

# From openagi/cli.py
def main():
    parser = argparse.ArgumentParser(description="OpenAGI CLI for various commands.")

    parser.add_argument(
        "--clear-ltm",
        action="store_true",
        help="Clear the long-term memory directory."
    )

    args = parser.parse_args()

    if args.clear_ltm:
        clear_long_term_memory()
    else:
        parser.print_help()

import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from openagi.actions.utils import run_action
from openagi.exception import OpenAGIException
from openagi.llms.base import LLMBaseModel
from openagi.memory.memory import Memory
from openagi.prompts.worker_task_execution import WorkerAgentTaskExecution
from openagi.tasks.task import Task
from openagi.utils.extraction import get_act_classes_from_json
from openagi.utils.extraction import get_last_json
from openagi.utils.helper import get_default_id

# From openagi/worker.py
class Worker(BaseModel):
    id: str = Field(default_factory=get_default_id)
    role: str = Field(description="Role of the worker.")
    instructions: Optional[str] = Field(description="Instructions the worker should follow.")
    llm: Optional[LLMBaseModel] = Field(
        description="LLM Model to be used.",
        default=None,
        exclude=True,
    )
    memory: Optional[Memory] = Field(
        default_factory=list,
        description="Memory to be used.",
        exclude=True,
    )
    actions: Optional[List[Any]] = Field(
        description="Actions that the Worker supports",
        default_factory=list,
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum number of steps to achieve the objective.",
    )
    output_key: str = Field(
        default="final_output",
        description="Key to be used to store the output.",
    )
    force_output: bool = Field(
        default=True,
        description="If set to True, the output will be overwritten even if it exists.",
    )
    
    # Validate output_key. Should contain only alphabets and only underscore are allowed. Not alphanumeric
    @field_validator("output_key")
    @classmethod
    def validate_output_key(cls, v, values, **kwargs):
        if not re.match("^[a-zA-Z_]+$", v):
            raise ValueError(
                f"Output key should contain only alphabets and only underscore are allowed. Got {v}"
            )
        return v

    class Config:
        arbitrary_types_allowed = True

    def worker_doc(self):
        """Returns a dictionary containing information about the worker, including its ID, role, description, and the supported actions."""
        return {
            "worker_id": self.id,
            "role": self.role,
            "description": self.instructions,
            "supported_actions": [action.cls_doc() for action in self.actions],
        }

    def provoke_thought_obs(self, observation):
        thoughts = f"""Observation: {observation}""".strip()
        return thoughts

    def should_continue(self, llm_resp: str) -> Union[bool, Optional[Dict]]:
        output: Dict = get_last_json(llm_resp, llm=self.llm, max_iterations=self.max_iterations)
        output_key_exists = bool(output and output.get(self.output_key))
        return (not output_key_exists, output)

    def _force_output(
        self, llm_resp: str, all_thoughts_and_obs: List[str]
    ) -> Union[bool, Optional[str]]:
        """Force the output once the max iterations are reached."""
        prompt = (
            "\n".join(all_thoughts_and_obs)
            + "Based on the previous action and observation, force and give me the output."
        )
        output = self.llm.run(prompt)
        cont, final_output = self.should_continue(output)
        if cont:
            prompt = (
                "\n".join(all_thoughts_and_obs)
                + f"Based on the previous action and observation, give me the output. {final_output}"
            )
            output = self.llm.run(prompt)
            cont, final_output = self.should_continue(output)
        if cont:
            raise OpenAGIException(
                f"LLM did not produce the expected output after {self.max_iterations} iterations."
            )
        return (cont, final_output)

    @functools.lru_cache(maxsize=100)
    def _cached_llm_run(self, prompt: str) -> str:
        """Cache LLM responses for identical prompts"""
        return self.llm.run(prompt)

    def save_to_memory(self, task: Task):
        """Optimized memory update"""
        if not hasattr(self, '_memory_buffer'):
            self._memory_buffer = []
        self._memory_buffer.append(task)
        
        # Batch update memory when buffer reaches certain size
        if len(self._memory_buffer) >= 5:
            for buffered_task in self._memory_buffer:
                self.memory.update_task(buffered_task)
            self._memory_buffer.clear()
        return True

    def execute_task(self, task: Task, context: Any = None) -> Any:
        """Optimized task execution"""
        logging.info(f"{'>'*20} Executing Task - {task.name}[{task.id}] with worker - {self.role}[{self.id}] {'<'*20}")
        
        # Pre-compute common values
        iteration = 1
        task_to_execute = f"{task.description}"
        worker_description = f"{self.role} - {self.instructions}"
        all_thoughts_and_obs = []
        
        # Generate base prompt once
        te_vars = dict(
            task_to_execute=task_to_execute,
            worker_description=worker_description,
            supported_actions=[action.cls_doc() for action in self.actions],
            thought_provokes=self.provoke_thought_obs(None),
            output_key=self.output_key,
            context=context,
            max_iterations=self.max_iterations,
        )
        base_prompt = WorkerAgentTaskExecution().from_template(te_vars)
        
        # Use cached LLM run
        prompt = f"{base_prompt}\nThought:\nIteration: {iteration}\nActions:\n"
        observations = self._cached_llm_run(prompt)
        all_thoughts_and_obs.append(prompt)

        while iteration < self.max_iterations + 1:

            logging.info(f"---- Iteration {iteration} ----")
            logging.debug("Checking if task should continue...")
            continue_flag, output = self.should_continue(observations)

            logging.debug("Extracting action from output...")
            action = output.get("action") if output else None
            if action:
                action = [action]

            # Save to memory
            if output:
                logging.debug("Saving task result and actions to memory...")
                task.result = observations
                task.actions = str([action.cls_doc() for action in self.actions])
                self.save_to_memory(task=task)

            if not continue_flag:
                logging.info(f"Task completed. Output: {output}")
                break

            if not action:
                logging.warning(f"No action found in the output: {output}")
                observations = f"Action: {action}\n{observations} Unable to extract action. Verify the output and try again."
                all_thoughts_and_obs.append(observations)
                iteration += 1
                continue

            if action:
                action_json = f"```json\n{output}\n```\n"
                try:
                    logging.debug("Getting action classes from JSON...")
                    actions = get_act_classes_from_json(action)
                    logging.info(
                        f"Extracted actions: {[act_cls.__name__ for act_cls, _ in actions]}"
                    )
                except KeyError as e:
                    if "cls" in e or "module" in e or "kls" in e:
                        observations = f"Action: {action_json}\n{observations}"
                        all_thoughts_and_obs.append(action_json)
                        all_thoughts_and_obs.append(observations)
                        iteration += 1
                        continue
                    else:
                        raise e

                for act_cls, params in actions:
                    params["memory"] = self.memory
                    params["llm"] = self.llm
                    try:
                        logging.debug(f"Running action: {act_cls.__name__}...")
                        res = run_action(action_cls=act_cls, **params)
                        logging.info(f"Action '{act_cls.__name__}' completed. Result: {res}")
                    except Exception as e:
                        logging.error(f"Error running action: {e}")
                        observations = f"Action: {action_json}\n{observations}. {e} Try to fix the error and try again. Ignore if already tried more than twice"
                        all_thoughts_and_obs.append(action_json)
                        all_thoughts_and_obs.append(observations)
                        iteration += 1
                        continue

                    observation_prompt = f"Observation: {res}\n"
                    all_thoughts_and_obs.append(action_json)
                    all_thoughts_and_obs.append(observation_prompt)
                    observations = res

                logging.debug("Provoking thought observation...")
                thought_prompt = self.provoke_thought_obs(observations)
                all_thoughts_and_obs.append(f"\n{thought_prompt}\nActions:\n")

                prompt = f"{base_prompt}\n" + "\n".join(all_thoughts_and_obs)
                logging.debug(f"\nSTART:{'*' * 20}\n{prompt}\n{'*' * 20}:END")
                pth = Path(f"{self.memory.session_id}/logs/{task.name}-{iteration}.log")
                
                pth.parent.mkdir(parents=True, exist_ok=True)
                with open(pth, "w", encoding="utf-8") as f:
                    f.write(f"{prompt}\n")
                logging.debug("Running LLM with updated prompt...")
                observations = self.llm.run(prompt)
            iteration += 1
        else:
            if iteration == self.max_iterations:
                logging.info("---- Forcing Output ----")
                if self.force_output:
                    logging.debug("Forcing output...")
                    cont, final_output = self._force_output(observations, all_thoughts_and_obs)
                    if cont:
                        raise OpenAGIException(
                            f"LLM did not produce the expected output after {iteration} iterations for task {task.name}"
                        )
                    output = final_output
                    logging.debug("Saving final task result and actions to memory...")
                    task.result = observations
                    task.actions = str([action.cls_doc() for action in self.actions])
                    self.save_to_memory(task=task)
                else:
                    raise OpenAGIException(
                        f"LLM did not produce the expected output after {iteration} iterations for task {task.name}"
                    )

        logging.info(
            f"Task Execution Completed - {task.name} with worker - {self.role}[{self.id}] in {iteration} iterations"
        )
        return output, task

    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)

# From openagi/worker.py
class Config:
        arbitrary_types_allowed = True

# From openagi/worker.py
def validate_output_key(cls, v, values, **kwargs):
        if not re.match("^[a-zA-Z_]+$", v):
            raise ValueError(
                f"Output key should contain only alphabets and only underscore are allowed. Got {v}"
            )
        return v

# From openagi/worker.py
def worker_doc(self):
        """Returns a dictionary containing information about the worker, including its ID, role, description, and the supported actions."""
        return {
            "worker_id": self.id,
            "role": self.role,
            "description": self.instructions,
            "supported_actions": [action.cls_doc() for action in self.actions],
        }

# From openagi/worker.py
def provoke_thought_obs(self, observation):
        thoughts = f"""Observation: {observation}""".strip()
        return thoughts

# From openagi/worker.py
def should_continue(self, llm_resp: str) -> Union[bool, Optional[Dict]]:
        output: Dict = get_last_json(llm_resp, llm=self.llm, max_iterations=self.max_iterations)
        output_key_exists = bool(output and output.get(self.output_key))
        return (not output_key_exists, output)

# From openagi/worker.py
def save_to_memory(self, task: Task):
        """Optimized memory update"""
        if not hasattr(self, '_memory_buffer'):
            self._memory_buffer = []
        self._memory_buffer.append(task)
        
        # Batch update memory when buffer reaches certain size
        if len(self._memory_buffer) >= 5:
            for buffered_task in self._memory_buffer:
                self.memory.update_task(buffered_task)
            self._memory_buffer.clear()
        return True

# From openagi/worker.py
def execute_task(self, task: Task, context: Any = None) -> Any:
        """Optimized task execution"""
        logging.info(f"{'>'*20} Executing Task - {task.name}[{task.id}] with worker - {self.role}[{self.id}] {'<'*20}")
        
        # Pre-compute common values
        iteration = 1
        task_to_execute = f"{task.description}"
        worker_description = f"{self.role} - {self.instructions}"
        all_thoughts_and_obs = []
        
        # Generate base prompt once
        te_vars = dict(
            task_to_execute=task_to_execute,
            worker_description=worker_description,
            supported_actions=[action.cls_doc() for action in self.actions],
            thought_provokes=self.provoke_thought_obs(None),
            output_key=self.output_key,
            context=context,
            max_iterations=self.max_iterations,
        )
        base_prompt = WorkerAgentTaskExecution().from_template(te_vars)
        
        # Use cached LLM run
        prompt = f"{base_prompt}\nThought:\nIteration: {iteration}\nActions:\n"
        observations = self._cached_llm_run(prompt)
        all_thoughts_and_obs.append(prompt)

        while iteration < self.max_iterations + 1:

            logging.info(f"---- Iteration {iteration} ----")
            logging.debug("Checking if task should continue...")
            continue_flag, output = self.should_continue(observations)

            logging.debug("Extracting action from output...")
            action = output.get("action") if output else None
            if action:
                action = [action]

            # Save to memory
            if output:
                logging.debug("Saving task result and actions to memory...")
                task.result = observations
                task.actions = str([action.cls_doc() for action in self.actions])
                self.save_to_memory(task=task)

            if not continue_flag:
                logging.info(f"Task completed. Output: {output}")
                break

            if not action:
                logging.warning(f"No action found in the output: {output}")
                observations = f"Action: {action}\n{observations} Unable to extract action. Verify the output and try again."
                all_thoughts_and_obs.append(observations)
                iteration += 1
                continue

            if action:
                action_json = f"```json\n{output}\n```\n"
                try:
                    logging.debug("Getting action classes from JSON...")
                    actions = get_act_classes_from_json(action)
                    logging.info(
                        f"Extracted actions: {[act_cls.__name__ for act_cls, _ in actions]}"
                    )
                except KeyError as e:
                    if "cls" in e or "module" in e or "kls" in e:
                        observations = f"Action: {action_json}\n{observations}"
                        all_thoughts_and_obs.append(action_json)
                        all_thoughts_and_obs.append(observations)
                        iteration += 1
                        continue
                    else:
                        raise e

                for act_cls, params in actions:
                    params["memory"] = self.memory
                    params["llm"] = self.llm
                    try:
                        logging.debug(f"Running action: {act_cls.__name__}...")
                        res = run_action(action_cls=act_cls, **params)
                        logging.info(f"Action '{act_cls.__name__}' completed. Result: {res}")
                    except Exception as e:
                        logging.error(f"Error running action: {e}")
                        observations = f"Action: {action_json}\n{observations}. {e} Try to fix the error and try again. Ignore if already tried more than twice"
                        all_thoughts_and_obs.append(action_json)
                        all_thoughts_and_obs.append(observations)
                        iteration += 1
                        continue

                    observation_prompt = f"Observation: {res}\n"
                    all_thoughts_and_obs.append(action_json)
                    all_thoughts_and_obs.append(observation_prompt)
                    observations = res

                logging.debug("Provoking thought observation...")
                thought_prompt = self.provoke_thought_obs(observations)
                all_thoughts_and_obs.append(f"\n{thought_prompt}\nActions:\n")

                prompt = f"{base_prompt}\n" + "\n".join(all_thoughts_and_obs)
                logging.debug(f"\nSTART:{'*' * 20}\n{prompt}\n{'*' * 20}:END")
                pth = Path(f"{self.memory.session_id}/logs/{task.name}-{iteration}.log")
                
                pth.parent.mkdir(parents=True, exist_ok=True)
                with open(pth, "w", encoding="utf-8") as f:
                    f.write(f"{prompt}\n")
                logging.debug("Running LLM with updated prompt...")
                observations = self.llm.run(prompt)
            iteration += 1
        else:
            if iteration == self.max_iterations:
                logging.info("---- Forcing Output ----")
                if self.force_output:
                    logging.debug("Forcing output...")
                    cont, final_output = self._force_output(observations, all_thoughts_and_obs)
                    if cont:
                        raise OpenAGIException(
                            f"LLM did not produce the expected output after {iteration} iterations for task {task.name}"
                        )
                    output = final_output
                    logging.debug("Saving final task result and actions to memory...")
                    task.result = observations
                    task.actions = str([action.cls_doc() for action in self.actions])
                    self.save_to_memory(task=task)
                else:
                    raise OpenAGIException(
                        f"LLM did not produce the expected output after {iteration} iterations for task {task.name}"
                    )

        logging.info(
            f"Task Execution Completed - {task.name} with worker - {self.role}[{self.id}] in {iteration} iterations"
        )
        return output, task

from openagi.actions.base import BaseAction

# From actions/obs_rag.py
class MemoryRagAction(BaseAction):
    """Action class to get all the results from the previous tasks for the current objetive.
    This action is responsible to reading and not writing. Writing is done by default for every task.
    """

    query: str = Field(
        ...,
        description="Query, a string, to run to retrieve the data from the results of previous tasks. Returns an Array of the results.",
    )
    max_results: int = Field(
        default=10,
        description="Max results to be used by querying the memory Defaults to integer 10.",
    )

    def execute(self):
        resp = self.memory.search(query=self.query, n_results=self.max_results or 10)
        logging.debug(f"Retreived MEMORY DATA  -  {resp}")
        return resp

# From actions/obs_rag.py
def execute(self):
        resp = self.memory.search(query=self.query, n_results=self.max_results or 10)
        logging.debug(f"Retreived MEMORY DATA  -  {resp}")
        return resp

from uuid import uuid4
import shutil
from openagi.storage.base import BaseStorage
from openagi.storage.chroma import ChromaStorage
from openagi.tasks.lists import TaskLists
from openagi.memory.sessiondict import SessionDict

# From memory/base.py
class BaseMemory(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    storage: BaseStorage = Field(
        default_factory=lambda: ChromaStorage,
        description="Storage to be used for the Memory.",
        exclude=True,
    )
    ltm_storage: BaseStorage = Field(
        default_factory=lambda: ChromaStorage,
        description="Long-term storage to be used for the Memory.",
        exclude=True,
    )

    long_term: bool = Field(default=False, description="Whether or not to use long term memory")
    ltm_threshold: float = Field(default=0.6,
                                 description="Semantic similarity threshold for long term memory instance retrieval")

    long_term_dir: str = Field(default=None, description="Path to directory for long-term memory storage")

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.storage = ChromaStorage.from_kwargs(collection_name=self.session_id)

        # Setting the long_term_dir from environment variable if not provided
        if self.long_term_dir is None:
            self.long_term_dir = os.getenv("LONG_TERM_DIR", ".long_term_dir")

        # Ensuring the directory is hidden by prefixing with a dot if necessary
        if not os.path.basename(self.long_term_dir).startswith('.'):
            self.long_term_dir = os.path.join(os.path.dirname(self.long_term_dir),
                                              f".{os.path.basename(self.long_term_dir)}")

        if self.long_term:
            os.makedirs(self.long_term_dir, exist_ok=True)

            self.ltm_storage = ChromaStorage.from_kwargs(
                collection_name="long_term_memory",
                persist_path=self.long_term_dir
            )
            assert 1 >= self.ltm_threshold >= 0.6, "Semantic similarity threshold should be between 0.6 and 1"

        logging.info(f"Session ID initialized: {self.session_id}")
        if self.long_term:
            logging.info(f"Long-term memory enabled. Using directory: {self.long_term_dir}")

    @staticmethod
    def clear_long_term_memory(directory: str):
        """Clears all data from the specified long-term memory directory."""
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logging.error(f'Failed to delete {file_path}. Reason: {e}')
            logging.info(f"Cleared all data from the long-term memory directory: {directory}")
        else:
            logging.warning(f"The long-term memory directory does not exist: {directory}")

    def search(self, query: str, n_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Search for similar tasks based on a query.

        :param query: The query string to search for.
        :param n_results: The number of results to return.
        :return: A dictionary of search results.
        """
        query_data = {
            "query_texts": query,
            "n_results": n_results,
            "include": ["metadatas", "documents"],
            **kwargs,
        }
        resp = self.storage.query_documents(**query_data)

        return resp["documents"]

    def display_memory(self) -> Dict[str, Any]:
        """
        Retrieve and display the current memory state from the database.

        :return: A dictionary of the current memory state.
        """
        result = self.storage.query_documents(self.session_id, n_results=2)
        return result or {}

    def save_task(self, task: Task) -> None:
        """
        Save execution details into Memory.

        :param task: The task to be saved.
        """
        metadata = self._create_metadata(task)
        self.storage.save_document(
            id=task.id,
            document=task.result,
            metadata=metadata,
        )
        logging.info(f"Task saved: {task.id}")

    def save_planned_tasks(self, tasks: TaskLists) -> None:
        """
        Save a list of planned tasks into Memory.

        :param tasks: The list of tasks to be saved.
        """
        for task in tasks:
            self.save_task(task)

    def update_task(self, task: Task) -> None:
        """
        Update a task in the Memory.

        :param task: The task to be updated.
        """
        metadata = self._create_metadata(task)
        self.storage.update_document(
            id=task.id,
            document=task.result,
            metadata=metadata,
        )
        logging.info(f"Task updated: {task.id}")

    def _create_metadata(self, task: Task) -> Dict[str, Any]:
        """
        Create metadata dictionary for a given task.

        :param task: The task for which to create metadata.
        :return: A dictionary of metadata.
        """
        return {
            "task_id": task.id,
            "session_id": self.session_id,
            "task_name": task.name,
            "task_description": task.description,
            "task_actions": task.actions,
        }

    def add_ltm(self, session : SessionDict):
        """
        Add a session to the long term memory
        :param session: The SessionDict object that has all the details of the session
        :return: None
        """
        self.ltm_storage.save_document(
            id = session.session_id,
            document= session.query,
            metadata= session.model_dump()
        )
        logging.info(f"Long term memory added for session : {session.session_id}")

    def update_ltm(self, session: SessionDict) -> None:
        """
        Update an existing session in long-term memory.

        :param session: The SessionDict object containing updated details of the session.
        :return: None
        """
        self.ltm_storage.update_document(
            id=session.session_id,
            document=session.query,
            metadata=session.model_dump()
        )

        logging.info(f"Long-term memory updated for session: {session.session_id}")

    def get_ltm(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve and return the long-term memory based on a query.

        :param query: The query string to search for.
        :param n_results: The number of results to return.
        :return: A dictionary of search results.
        """
        query_data = {
            "query_texts": query,
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"],
        }
        response = self.ltm_storage.query_documents(**query_data)
        results = []
        # if "documents" in response and "distances" in response:
        for doc, metadata, distance in zip(response["documents"][0], response["metadatas"][0], response["distances"][0]):
            results.append({
                "document": doc,
                "metadata": metadata,
                "similarity_score": 1 - distance
            })
        if results:
            logging.info(f"Retrieved long-term memory for query: {query}\n{results[0]['document'][:250]}")
            return results

        logging.info(f"No documents found for query: {query}")
        return results

# From memory/base.py
def search(self, query: str, n_results: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Search for similar tasks based on a query.

        :param query: The query string to search for.
        :param n_results: The number of results to return.
        :return: A dictionary of search results.
        """
        query_data = {
            "query_texts": query,
            "n_results": n_results,
            "include": ["metadatas", "documents"],
            **kwargs,
        }
        resp = self.storage.query_documents(**query_data)

        return resp["documents"]

# From memory/base.py
def display_memory(self) -> Dict[str, Any]:
        """
        Retrieve and display the current memory state from the database.

        :return: A dictionary of the current memory state.
        """
        result = self.storage.query_documents(self.session_id, n_results=2)
        return result or {}

# From memory/base.py
def save_task(self, task: Task) -> None:
        """
        Save execution details into Memory.

        :param task: The task to be saved.
        """
        metadata = self._create_metadata(task)
        self.storage.save_document(
            id=task.id,
            document=task.result,
            metadata=metadata,
        )
        logging.info(f"Task saved: {task.id}")

# From memory/base.py
def save_planned_tasks(self, tasks: TaskLists) -> None:
        """
        Save a list of planned tasks into Memory.

        :param tasks: The list of tasks to be saved.
        """
        for task in tasks:
            self.save_task(task)

# From memory/base.py
def update_task(self, task: Task) -> None:
        """
        Update a task in the Memory.

        :param task: The task to be updated.
        """
        metadata = self._create_metadata(task)
        self.storage.update_document(
            id=task.id,
            document=task.result,
            metadata=metadata,
        )
        logging.info(f"Task updated: {task.id}")

# From memory/base.py
def add_ltm(self, session : SessionDict):
        """
        Add a session to the long term memory
        :param session: The SessionDict object that has all the details of the session
        :return: None
        """
        self.ltm_storage.save_document(
            id = session.session_id,
            document= session.query,
            metadata= session.model_dump()
        )
        logging.info(f"Long term memory added for session : {session.session_id}")

# From memory/base.py
def update_ltm(self, session: SessionDict) -> None:
        """
        Update an existing session in long-term memory.

        :param session: The SessionDict object containing updated details of the session.
        :return: None
        """
        self.ltm_storage.update_document(
            id=session.session_id,
            document=session.query,
            metadata=session.model_dump()
        )

        logging.info(f"Long-term memory updated for session: {session.session_id}")

# From memory/base.py
def get_ltm(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve and return the long-term memory based on a query.

        :param query: The query string to search for.
        :param n_results: The number of results to return.
        :return: A dictionary of search results.
        """
        query_data = {
            "query_texts": query,
            "n_results": n_results,
            "include": ["metadatas", "documents", "distances"],
        }
        response = self.ltm_storage.query_documents(**query_data)
        results = []
        # if "documents" in response and "distances" in response:
        for doc, metadata, distance in zip(response["documents"][0], response["metadatas"][0], response["distances"][0]):
            results.append({
                "document": doc,
                "metadata": metadata,
                "similarity_score": 1 - distance
            })
        if results:
            logging.info(f"Retrieved long-term memory for query: {query}\n{results[0]['document'][:250]}")
            return results

        logging.info(f"No documents found for query: {query}")
        return results


# From memory/memory.py
class Memory(BaseMemory):
    pass


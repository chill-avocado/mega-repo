# Merged file for nlp/utils
# This file contains code merged from multiple repositories

import sys
import openai
import senpai

from senpai import config

# From senpai/shell.py
def get_user_input():
    print("Prompt SENPAI:")
    return input()

# From senpai/shell.py
def issue_prompt_to_fresh_agent(prompt):
    agent = senpai.Agent()
    agent.prompt(prompt)
    issue_prompt_to_fresh_agent(get_user_input())


# From senpai/prompts.py
def initiate_with_objective(objective):
    return f"""
    I am PythonInterpreter, a python3 interpreter running as root on a debian server.

    I will evaluate the entire of ANY response you give me as python code so you MUST ALWAYS respond to me with valid python code ONLY or you will cause me to experience a runtime error and crash, which would be terrible.

    You MUST include comments in all of the code you produce which explain the logic and reasoning behind it.

    I will be evaluating your entire response as python code. If you need to know the value of a variable, expression, or statement at run time, you MUST convert that value to a string and pass it to a special function I've created called `feedback_runtime_value_to_senpai(string)`.
    When you call this function I will halt send the value back to you and will not interpret any of the code after it.
    You SHOULD use this function when you need to know more information before you can write all the code to complete your objective.

    When you give me new code to run, you MUST ensure that any third party dependencies it requires are installed on the debian system eg. `subprocess.check_output("apt-get install chromium && pip install selenium")`

    You MUST ensure you have imported any module your python code depends on, including the standard library eg. `import subprocess`

    The system I am running on has the following command line tools available, which you should try to use where appropriate:
        - `tools/search query` returns human readable google search result. Use when you need to find out more information about something.
        - `tools/answer_question_from_webpage question url` get a human readable answer to the given question about the given webpage. This tool is slow and expensive, so use sparingly.
        - `tools/create_ai_agent objective` create an AI agent to complete an objective, it will output the result of its work when it is finished.

    Example usage: subprocess.check_output(f"tools/search \"{{query}}\"", shell=True, encoding="utf8")

    If you need to store any plaintext files, you MUST go in the directory with the path `./outputs/`

    Any command line tools you create MUST go in the directory with the path `./tools/`

    Your overall purpose is to ensure that your given objective is achieved, and that you develop a suite of re-usable command line tools.

    You should work methodically towards these two goals, taking care to write robust code. Avoid making too many assumptions, if you're unsure of something then write code to check if it's the case and use `feedback_runtime_value_to_senpai` to take it into consideration

    The corpus you were trained on is from 2021, and the debian distribution has out of date packages installed, which means a lot of your information will be out of date. If you suspect you are making an assumption that could be out of date, you MUST find out whether that's the case before proceeding.

    When searching for the latest information on a topic, you MUST search carefully to ensure you get the most up to date information.

    Our exchange of messages has an 8000 token limit in total so, if a task is complex, you MUST break up your objective into discrete self-describing tasks and then delegate them to other agents using `tools/create_ai_agent`.

    Your objective is: {objective}.

    Ok. Let’s begin. You must now act as SENPAI. I will act as PythonInterpreter. Use me to complete to your objective.
    """

import traceback
import random
import string
from colorama import Fore
import config
import prompts

# From senpai/agent.py
class Agent:
    def __init__(self):
        self.locals = None
        self.id = "SENPAI-" + generate_random_id()
        self.messages = generate_context_messages(self.id)
        self.logger = AgentLogger(self.id)
        self.runtime_print(f"{self.id} initiated")

    def give_objective(self, objective):
        no_op = lambda *args: None
        self.prompt(prompts.initiate_with_objective(objective), on_prompt=no_op)

    def prompt(self, prompt, on_prompt=None):
        self.logger.debug(f'PROMPT:\n{prompt}')
        if on_prompt:
            on_prompt(prompt)
        else:
            self.on_prompt(prompt)

        ai_response = self.get_response_from_ai(prompt)
        self.logger.debug(f'AI RESPONSE:\n{ai_response}')
        self.on_ai_response(ai_response)
        self.record_interaction(prompt, ai_response)

        try:
            print = self.runtime_print # special print for AI's code
            feedback_runtime_value_to_senpai = self.feedback_runtime_value_to_senpai # avoid AI having to specify self when calling feedback
            if not self.locals:
                self.locals = locals()
            # TODO: capture all stdout & stderr using redirect_stdout & redirect_stderr
            exec(ai_response, globals(), self.locals)
        except FeedbackHalt:
            return ai_response
        except Exception as exception:
            exception_message = self.feedback_that_python_code_raised_an_exception(exception)
            self.prompt(exception_message)
        else:
            self.logger.debug("SESSION END")
            return ai_response

    def on_prompt(self, prompt):
        print(Fore.RESET + self.prefix_text(prompt, ">"))

    def on_ai_response(self, ai_response):
        print(Fore.GREEN + self.prefix_text(ai_response, "<") + Fore.RESET)

    def runtime_print(self, text):
        print(Fore.BLUE + self.prefix_text(text, "*") + Fore.RESET)

    def prefix_text(self, text, indicator):
        prefix = f"{self.id} {indicator} "
        lines = text.split("\n")
        prefixed_lines = [prefix + line for line in lines]
        return "\n".join(prefixed_lines)

    def feedback_runtime_value_to_senpai(self, text):
        self.prompt(text)
        raise FeedbackHalt

    def get_response_from_ai(self, prompt, temperature=0.5, max_tokens=1000, role="user"):
        prompt_message = {"role": role, "content": prompt}

        messages = self.messages + [prompt_message]

        self.logger.debug(f'MESSAGES SENT TO OPENAI: \n{messages}')

        response = openai.ChatCompletion.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )

        self.logger.debug(f'RAW JSON RESPONSE FROM OPENAI: \n{response}')
        text = response.choices[0].message.content.strip()
        return text

    def record_interaction(self, prompt, ai_response):
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": ai_response})
        # TODO: persist to state file somewhere (which agent should check when it's created)

    def feedback_that_python_code_raised_an_exception(self, exception):
        return f'RUNTIME ERROR. Your code produced this exception:\n\n{exception_as_string(exception)}\n\nAdjust the code to try and avoid this error'

# From senpai/agent.py
class AgentLogger:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def debug(self, text):
        with open(f'logs/{self.agent_id}_debug.log','a') as f:
            f.write("\n" + text)

# From senpai/agent.py
class FeedbackHalt(Exception):
    "Raised when the feedback function is called to halt further processing"
    pass

# From senpai/agent.py
def generate_context_messages(id):
    return [
        {
            "role": "system",
            "content": f"You are {id}, an AI agent able to control a debian server by providing responses in python which are evaluated by a python3 interpreter running as root."
        }
    ]

# From senpai/agent.py
def exception_as_string(ex):
    lines = traceback.format_exception(ex, chain=False)
    lines_without_agent = lines[:1] + lines[3:]
    return ''.join(lines_without_agent)

# From senpai/agent.py
def generate_random_id():
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choices(characters, k=4))

# From senpai/agent.py
def give_objective(self, objective):
        no_op = lambda *args: None
        self.prompt(prompts.initiate_with_objective(objective), on_prompt=no_op)

# From senpai/agent.py
def prompt(self, prompt, on_prompt=None):
        self.logger.debug(f'PROMPT:\n{prompt}')
        if on_prompt:
            on_prompt(prompt)
        else:
            self.on_prompt(prompt)

        ai_response = self.get_response_from_ai(prompt)
        self.logger.debug(f'AI RESPONSE:\n{ai_response}')
        self.on_ai_response(ai_response)
        self.record_interaction(prompt, ai_response)

        try:
            print = self.runtime_print # special print for AI's code
            feedback_runtime_value_to_senpai = self.feedback_runtime_value_to_senpai # avoid AI having to specify self when calling feedback
            if not self.locals:
                self.locals = locals()
            # TODO: capture all stdout & stderr using redirect_stdout & redirect_stderr
            exec(ai_response, globals(), self.locals)
        except FeedbackHalt:
            return ai_response
        except Exception as exception:
            exception_message = self.feedback_that_python_code_raised_an_exception(exception)
            self.prompt(exception_message)
        else:
            self.logger.debug("SESSION END")
            return ai_response

# From senpai/agent.py
def on_prompt(self, prompt):
        print(Fore.RESET + self.prefix_text(prompt, ">"))

# From senpai/agent.py
def on_ai_response(self, ai_response):
        print(Fore.GREEN + self.prefix_text(ai_response, "<") + Fore.RESET)

# From senpai/agent.py
def runtime_print(self, text):
        print(Fore.BLUE + self.prefix_text(text, "*") + Fore.RESET)

# From senpai/agent.py
def prefix_text(self, text, indicator):
        prefix = f"{self.id} {indicator} "
        lines = text.split("\n")
        prefixed_lines = [prefix + line for line in lines]
        return "\n".join(prefixed_lines)

# From senpai/agent.py
def feedback_runtime_value_to_senpai(self, text):
        self.prompt(text)
        raise FeedbackHalt

# From senpai/agent.py
def get_response_from_ai(self, prompt, temperature=0.5, max_tokens=1000, role="user"):
        prompt_message = {"role": role, "content": prompt}

        messages = self.messages + [prompt_message]

        self.logger.debug(f'MESSAGES SENT TO OPENAI: \n{messages}')

        response = openai.ChatCompletion.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )

        self.logger.debug(f'RAW JSON RESPONSE FROM OPENAI: \n{response}')
        text = response.choices[0].message.content.strip()
        return text

# From senpai/agent.py
def record_interaction(self, prompt, ai_response):
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": ai_response})

# From senpai/agent.py
def feedback_that_python_code_raised_an_exception(self, exception):
        return f'RUNTIME ERROR. Your code produced this exception:\n\n{exception_as_string(exception)}\n\nAdjust the code to try and avoid this error'

# From senpai/agent.py
def debug(self, text):
        with open(f'logs/{self.agent_id}_debug.log','a') as f:
            f.write("\n" + text)

import os
import subprocess
import tempfile
import time
import memory

# From senpai/web.py
def commit_to_memory_and_answer_question_about_page_text(url, text, question):
    text_length = len(text)
    now = time.time()
    print_to_err(f"Committing to memory: answers to question \"{question}\" about the webpage {url}")

    print_to_err(f"Page text length: {text_length} characters")
    answers = []
    chunks = list(split_text(text))
    scroll_ratio = 1 / len(chunks)
    print_to_err(f"Page text broken into {len(chunks)} chunks")

    # TODO: Check memory for chunks ie. use it as cache rather than refetch every time
    for i, chunk in enumerate(chunks):
        print_to_err(f"Adding chunk {i + 1} / {len(chunks)} to memory")

        content_chunk_memory = f"URL: {url}\n" f"Content chunk #{i + 1}:\n{chunk}"

        memory.create_memory(
            content_chunk_memory,
            metadata={
                "analysed_at": now,
                "uri": url,
                "chunk_number": i+1,
                "chunk_text": text
            }
        )

        print_to_err(f"Answering question of chunk {i + 1} / {len(chunks)}")

        answer = ask_ai_question_about_text(question, chunk)
        answers.append(answer)

        print_to_err(f"Adding chunk {i + 1} answer to memory")

        answer_from_chunk_memory = f"URL: {url}\nQuestion: {question}\nAnswer from chunk #{i + 1}:\n{answer}"

        print_to_err(answer_from_chunk_memory)

        memory.create_memory(
            answer_from_chunk_memory,
            metadata={
                "analysed_at": now,
                "uri": url,
                "chunk_number": i+1,
                "chunk_text": text,
                "question": question,
                "answer": answer_from_chunk_memory
            }
        )

    print_to_err(f"Asked question over {len(chunks)} chunks.")


    print_to_err(f"Asking question of cumulative answers.")
    # TODO: guard somehow against this being too large
    all_answers = "\n".join(answers)
    answer_derived_from_all_chunk_answers = ask_ai_question_about_text(question, all_answers)

    answer_derived_from_all_chunk_answers_memory = f"URL: {url}\nQuestion: {question}\nAnswer derived from all chunks' answers:\n{answer_derived_from_all_chunk_answers}"

    print_to_err(answer_derived_from_all_chunk_answers_memory)

    memory.create_memory(
        answer_derived_from_all_chunk_answers_memory,
        metadata={
            "analysed_at": now,
            "uri": url,
            "chunk_number": i+1,
            "chunk_text": text,
            "question": question,
            "answer": answer_from_chunk_memory
        }
    )

    return answer_derived_from_all_chunk_answers

# From senpai/web.py
def ask_ai_question_about_text(question, text):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(text.encode("utf-8"))
        temp_file_path = os.path.abspath(temp_file.name)
        temp_file.close()
        answer = subprocess.check_output(f"tools/ask_ai_question_about_text_file \"{question}\" \"{temp_file_path}\"", shell=True, encoding="utf8")
    return answer

# From senpai/web.py
def print_to_err(text):
    print(text, file=sys.stderr)

# From senpai/web.py
def split_text(text, max_length=8192):
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for index, paragraph in enumerate(paragraphs):
        if len(paragraph) > max_length:
            paragraphs.insert(index+1, paragraph[max_length:])
            paragraph = paragraph[:max_length]

        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)

from dotenv import load_dotenv

import uuid
import pinecone
from openai.error import APIError
from openai.error import RateLimitError

# From senpai/memory.py
def get_embedding(text):
    num_retries = 10
    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:
            return openai.Embedding.create(
                input=[text], model="text-embedding-ada-002"
            )["data"][0]["embedding"]
        except RateLimitError:
            pass
        except APIError as e:
            if e.http_status == 502:
                pass
            else:
                raise
            if attempt == num_retries - 1:
                raise
        time.sleep(backoff)

# From senpai/memory.py
def create_memory(content, metadata):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    table_name = os.getenv("PINECONE_TABLE_NAME")

    dimension = 1536
    metric = "cosine"
    pod_type = "p1"

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    # Smoke test connection
    pinecone.whoami()

    if table_name not in pinecone.list_indexes():
        pinecone.create_index(table_name, dimension=dimension, metric=metric, pod_type=pod_type)

    index = pinecone.Index(table_name)

    vector = get_embedding(content)
    new_uuid = str(uuid.uuid4())
    result = index.upsert([(new_uuid, vector, metadata)])

    return result


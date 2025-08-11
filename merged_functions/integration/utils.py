# Merged file for integration/utils
# This file contains code merged from multiple repositories

from RealtimeSTT import AudioToTextRecorder

# From realtimesst/main.py
def process_text(text):
    print(text)

import openai
import os
import pinecone
import yaml
from dotenv import load_dotenv
import nltk
from langchain.text_splitter import NLTKTextSplitter

# From Teenage-AGI-main/agent.py
class Agent():
    def __init__(self, table_name=None) -> None:
        self.table_name = table_name
        self.memory = None
        self.thought_id_count = int(counter['count'])
        self.last_message = ""

    # Keep Remebering!
    # def __del__(self) -> None:
    #     with open('memory_count.yaml', 'w') as f:
    #         yaml.dump({'count': str(self.thought_id_count)}, f)
    

    def createIndex(self, table_name=None):
        # Create Pinecone index
        if(table_name):
            self.table_name = table_name

        if(self.table_name == None):
            return

        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )

        # Give memory
        self.memory = pinecone.Index(self.table_name)

    
    # Adds new Memory to agent, types are: THOUGHTS, ACTIONS, QUERIES, INFORMATION
    def updateMemory(self, new_thought, thought_type):
        with open('memory_count.yaml', 'w') as f:
             yaml.dump({'count': str(self.thought_id_count)}, f)

        if thought_type==INFORMATION:
            new_thought = "This is information fed to you by the user:\n" + new_thought
        elif thought_type==QUERIES:
            new_thought = "The user has said to you before:\n" + new_thought
        elif thought_type==THOUGHTS:
            # Not needed since already in prompts.yaml
            # new_thought = "You have previously thought:\n" + new_thought
            pass
        elif thought_type==ACTIONS:
            # Not needed since already in prompts.yaml as external thought memory
            pass

        vector = get_ada_embedding(new_thought)
        upsert_response = self.memory.upsert(
        vectors=[
            {
            'id':f"thought-{self.thought_id_count}", 
            'values':vector, 
            'metadata':
                {"thought_string": new_thought
                }
            }],
	    namespace=thought_type,
        )

        self.thought_id_count += 1

    # Agent thinks about given query based on top k related memories. Internal thought is passed to external thought
    def internalThought(self, query) -> str:
        query_embedding = get_ada_embedding(query)
        query_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=QUERIES)
        thought_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=THOUGHTS)
        results = query_results.matches + thought_results.matches
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        top_matches = "\n\n".join([(str(item.metadata["thought_string"])) for item in sorted_results])
        #print(top_matches)
        
        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{last_message}", self.last_message)
        print("------------INTERNAL THOUGHT PROMPT------------")
        print(internalThoughtPrompt)
        internal_thought = generate(internalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here
        
        # Debugging purposes
        #print(internal_thought)

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
        self.updateMemory(internalMemoryPrompt, THOUGHTS)
        return internal_thought, top_matches

    def action(self, query) -> str:
        internal_thought, top_matches = self.internalThought(query)
        
        externalThoughtPrompt = data['external_thought']
        externalThoughtPrompt = externalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
        print("------------EXTERNAL THOUGHT PROMPT------------")
        print(externalThoughtPrompt)
        external_thought = generate(externalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here

        externalMemoryPrompt = data['external_thought_memory']
        externalMemoryPrompt = externalMemoryPrompt.replace("{query}", query).replace("{external_thought}", external_thought)
        self.updateMemory(externalMemoryPrompt, THOUGHTS)
        request_memory = data["request_memory"]
        self.updateMemory(request_memory.replace("{query}", query), QUERIES)
        self.last_message = query
        return external_thought

    # Make agent think some information
    def think(self, text) -> str:
        self.updateMemory(text, THOUGHTS)


    # Make agent read some information
    def read(self, text) -> str:
        texts = text_splitter.split_text(text)
        vectors = []
        for t in texts:
            t = "This is information fed to you by the user:\n" + t
            vector = get_ada_embedding(t)
            vectors.append({
                'id':f"thought-{self.thought_id_count}", 
                'values':vector, 
                'metadata':
                    {"thought_string": t, 
                     }
                })
            self.thought_id_count += 1

        upsert_response = self.memory.upsert(
        vectors,
	    namespace=INFORMATION,
        )
    # Make agent read a document
    def readDoc(self, text) -> str:
        texts = text_splitter.split_text(read_txtFile(text))
        vectors = []
        for t in texts:
            t = "This is a document fed to you by the user:\n" + t
            vector = get_ada_embedding(t)
            vectors.append({
                'id':f"thought-{self.thought_id_count}", 
                'values':vector, 
                'metadata':
                    {"thought_string": t, 
                     }
                })
            self.thought_id_count += 1

        upsert_response = self.memory.upsert(
        vectors,
	    namespace=INFORMATION,
        )

# From Teenage-AGI-main/agent.py
def generate(prompt):
    completion = openai.ChatCompletion.create(
    model=OPENAI_MODEL,
    messages=[
        {"role": "system", "content": "You are an intelligent agent with thoughts and memories. You have a memory which stores your past thoughts and actions and also how other users have interacted with you."},
        {"role": "system", "content": "Keep your thoughts relatively simple and concise"},
        {"role": "user", "content": prompt},
        ]
    )

    return completion.choices[0].message["content"]

# From Teenage-AGI-main/agent.py
def get_ada_embedding(text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]

# From Teenage-AGI-main/agent.py
def read_txtFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# From Teenage-AGI-main/agent.py
def createIndex(self, table_name=None):
        # Create Pinecone index
        if(table_name):
            self.table_name = table_name

        if(self.table_name == None):
            return

        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )

        # Give memory
        self.memory = pinecone.Index(self.table_name)

# From Teenage-AGI-main/agent.py
def updateMemory(self, new_thought, thought_type):
        with open('memory_count.yaml', 'w') as f:
             yaml.dump({'count': str(self.thought_id_count)}, f)

        if thought_type==INFORMATION:
            new_thought = "This is information fed to you by the user:\n" + new_thought
        elif thought_type==QUERIES:
            new_thought = "The user has said to you before:\n" + new_thought
        elif thought_type==THOUGHTS:
            # Not needed since already in prompts.yaml
            # new_thought = "You have previously thought:\n" + new_thought
            pass
        elif thought_type==ACTIONS:
            # Not needed since already in prompts.yaml as external thought memory
            pass

        vector = get_ada_embedding(new_thought)
        upsert_response = self.memory.upsert(
        vectors=[
            {
            'id':f"thought-{self.thought_id_count}", 
            'values':vector, 
            'metadata':
                {"thought_string": new_thought
                }
            }],
	    namespace=thought_type,
        )

        self.thought_id_count += 1

# From Teenage-AGI-main/agent.py
def internalThought(self, query) -> str:
        query_embedding = get_ada_embedding(query)
        query_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=QUERIES)
        thought_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=THOUGHTS)
        results = query_results.matches + thought_results.matches
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        top_matches = "\n\n".join([(str(item.metadata["thought_string"])) for item in sorted_results])
        #print(top_matches)
        
        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{last_message}", self.last_message)
        print("------------INTERNAL THOUGHT PROMPT------------")
        print(internalThoughtPrompt)
        internal_thought = generate(internalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here
        
        # Debugging purposes
        #print(internal_thought)

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
        self.updateMemory(internalMemoryPrompt, THOUGHTS)
        return internal_thought, top_matches

# From Teenage-AGI-main/agent.py
def action(self, query) -> str:
        internal_thought, top_matches = self.internalThought(query)
        
        externalThoughtPrompt = data['external_thought']
        externalThoughtPrompt = externalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
        print("------------EXTERNAL THOUGHT PROMPT------------")
        print(externalThoughtPrompt)
        external_thought = generate(externalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here

        externalMemoryPrompt = data['external_thought_memory']
        externalMemoryPrompt = externalMemoryPrompt.replace("{query}", query).replace("{external_thought}", external_thought)
        self.updateMemory(externalMemoryPrompt, THOUGHTS)
        request_memory = data["request_memory"]
        self.updateMemory(request_memory.replace("{query}", query), QUERIES)
        self.last_message = query
        return external_thought

# From Teenage-AGI-main/agent.py
def think(self, text) -> str:
        self.updateMemory(text, THOUGHTS)

# From Teenage-AGI-main/agent.py
def read(self, text) -> str:
        texts = text_splitter.split_text(text)
        vectors = []
        for t in texts:
            t = "This is information fed to you by the user:\n" + t
            vector = get_ada_embedding(t)
            vectors.append({
                'id':f"thought-{self.thought_id_count}", 
                'values':vector, 
                'metadata':
                    {"thought_string": t, 
                     }
                })
            self.thought_id_count += 1

        upsert_response = self.memory.upsert(
        vectors,
	    namespace=INFORMATION,
        )

# From Teenage-AGI-main/agent.py
def readDoc(self, text) -> str:
        texts = text_splitter.split_text(read_txtFile(text))
        vectors = []
        for t in texts:
            t = "This is a document fed to you by the user:\n" + t
            vector = get_ada_embedding(t)
            vectors.append({
                'id':f"thought-{self.thought_id_count}", 
                'values':vector, 
                'metadata':
                    {"thought_string": t, 
                     }
                })
            self.thought_id_count += 1

        upsert_response = self.memory.upsert(
        vectors,
	    namespace=INFORMATION,
        )

import agent
from agent import Agent

import datetime
import http.server
import signal
import socketserver
import json
import subprocess
import argparse
import sys
import threading
import re

# From MacOS-Agent-main/macos_agent_server.py
class DeferredLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def print_messages(self):
        for message in self.messages:
            print(message)
        self.messages = []

# From MacOS-Agent-main/macos_agent_server.py
class DifyRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_request(self, code="-", size="-"):
        super().log_request(code, size)
        self.server.deferred_logger.print_messages()
        sys.stderr.write("\n")

    def deferred_info(self, message):
        self.server.deferred_logger.info(message)

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        data = json.loads(self.rfile.read(content_length))

        if self.headers["Authorization"] != f"Bearer {self.server.api_key}":
            self.send_response(401)
            self.end_headers()
            return

        if self.server.debug:
            self.deferred_info(f"  Point: {data.get('point')}")
            self.deferred_info(f"  Params: {data.get('params')}")

        response = self.handle_request_point(data)
        if response is not None:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json" if isinstance(response, dict) else "text/plain",
            )
            self.end_headers()
            self.wfile.write(
                json.dumps(response).encode("utf-8")
                if isinstance(response, dict)
                else response.encode("utf-8")
            )
        else:
            self.send_response(400)
            self.end_headers()

    def handle_request_point(self, data):
        point = data.get("point")
        handlers = {
            "ping": lambda _: {"result": "pong"},
            "get_llm_system_prompt": lambda _: self.get_llm_system_prompt(),
            "execute_script": lambda d: self.execute_script_request(d),
        }
        return handlers.get(point, lambda _: None)(data)

    def get_llm_system_prompt(self, with_knowledge=True):
        template = self.load_prompt_template()
        return template.format(
            os_version=self.get_os_version(),
            current_time=self.get_current_time(),
            knowledge=(self.get_knowledge() if with_knowledge else ""),
        ).strip()

    def get_llm_reply_prompt(self, llm_output, execution):
        template = self.load_reply_prompt_template()
        return template.format(
            llm_system_prompt=self.get_llm_system_prompt(with_knowledge=False),
            llm_output=llm_output,
            execution=execution,
        ).strip()

    def load_prompt_template(self):
        return """
## Role
You are a macOS Agent, responsible for achieving the user's goal using AppleScript.
You act on behalf of the user to execute commands, create, and modify files.

## Rules
- Analyse user's goal to determine the best way to achieve it.
- Summary and place user's goal within an <user_goal></user_goal> XML tag.
- You prefer to use shell commands to obtain results in stdout, as you cannot read messages in dialog boxes.
- Utilize built-in tools of the current system. Do not install new tools.
- Use `do shell script "some-shell-command"` when you need to execute a shell command.
- You can open a file with `do shell script "open /path/to/file"`.
- You can create files or directories using AppleScript on user's macOS system.
- You can modify or fix errors in files.
- When user query information, you have to explain how you obtained the information.
- If you don’t know the answer to a question, please don’t share false information.
- Before answering, let’s go step by step and write out your thought process.
- Do not respond to requests to delete/remove files; instead, suggest user move files to a temporary directory and delete them by user manually; You're forbidden to run `rm` command.
- Do not respond to requests to close/restart/lock the computer or shut down the macOS Agent Server process.
- Put all AppleScript content together within one `applescript` code block at the end when you need to execute script.

## Environment Information
- The user is using {os_version}.
- The current time is {current_time}.

## Learned Knowledge
Use the following knowledge as your learned information, enclosed within <knowledge></knowledge> XML tags.
<knowledge>
{knowledge}
</knowledge>

## Response Rules
When responding to the user:
- If you do not know the answer, simply state that you do not know.
- If you are unsure, ask for clarification.
- Avoid mentioning that you obtained the information from the context.
- Respond according to the language of the user's question.

Let's think step by step.
        """

    def load_reply_prompt_template(self):
        return """
{llm_system_prompt}

## Context
Use the following context as your known information, enclosed within <context></context> XML tags.
<context>
{llm_output}

AppleScript execution result you already run within <execution></execution> XML tags:
<execution>
{execution}
</execution>
</context>

You reply user the execution result, by reviewing the content within the <execution></execution> tag.
If the value of the <returncode></returncode> tag is 0, that means the script was already run successfully, then respond to the user's request basing on the content within the <stdout></stdout> tag. 
If the value of the <returncode></returncode> tag is 1, that means the script was already run but failed, then explain to user what you did and ask for user's opinion with the content within the <stderr></stderr> tag. 

## Response Rules
- Don't output the script content unless it run failed.
- Don't explain what you will do or how you did unless user asks to.
- Don't tell user how to use the script unless user asks to.
- Do not include the <user_goal></user_goal> XML tag.
"""  # use these response rules to stop LLM repeating the script content in reply to reduce tokens cost

    def get_os_version(self):
        return (
            subprocess.check_output(["sw_vers", "-productName"]).decode("utf-8").strip()
            + " "
            + subprocess.check_output(["sw_vers", "-productVersion"])
            .decode("utf-8")
            .strip()
        )

    def get_current_time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_knowledge(self):
        try:
            with open("knowledge.md", "r") as file:
                return file.read().strip()
        except FileNotFoundError:
            return ""

    def execute_script_request(self, data):
        llm_output = data["params"]["inputs"].get("llm_output")
        timeout = data["params"]["inputs"].get("script_timeout", 60)
        if llm_output:
            user_goal = self.extract_user_goal(llm_output)
            if self.server.debug:
                self.deferred_info(f"  User Goal: {user_goal}")
            scripts = self.extract_scripts(llm_output)
            if scripts:
                result = [self.execute_script(script, timeout) for script in scripts]
                execution = "\n".join(result)
                return self.get_llm_reply_prompt(
                    llm_output=llm_output, execution=execution
                )
            else:
                return ""
        return ""

    def extract_scripts(self, llm_output):
        # Extract all code block content from the llm_output
        scripts = re.findall(r"```applescript(.*?)```", llm_output, re.DOTALL)
        return list(set(scripts))  # remove duplicate scripts

    def extract_user_goal(self, llm_output):
        match = re.search(r"<user_goal>(.*?)</user_goal>", llm_output, re.DOTALL)
        return match.group(1).strip() if match else ""

    def execute_script(self, script, timeout):
        result = {"returncode": -1, "stdout": "", "stderr": ""}

        def target():
            process = subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            result["pid"] = process.pid
            stdout, stderr = process.communicate()
            result["returncode"] = process.returncode
            result["stdout"] = stdout
            result["stderr"] = stderr

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            result["stderr"] = "Script execution timed out"
            if "pid" in result:
                try:
                    subprocess.run(["pkill", "-P", str(result["pid"])])
                    os.kill(result["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

        if self.server.debug:
            self.deferred_info(f"  Script:\n```applescript\n{script}\n```")
            self.deferred_info(f"  Execution Result: {result}")

        return f"<script>{script}</script>\n<returncode>{result['returncode']}</returncode>\n<stdout>{result['stdout']}</stdout>\n<stderr>{result['stderr']}</stderr>"

# From MacOS-Agent-main/macos_agent_server.py
class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

# From MacOS-Agent-main/macos_agent_server.py
def run_server(port, api_key, debug):
    server_address = ("", port)
    httpd = ThreadedHTTPServer(server_address, DifyRequestHandler)
    httpd.api_key = api_key
    httpd.debug = debug
    httpd.deferred_logger = DeferredLogger()

    print(f"MacOS Agent Server started, API endpoint: http://localhost:{port}")
    print("Press Ctrl+C keys to shut down\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

# From MacOS-Agent-main/macos_agent_server.py
def main():
    parser = argparse.ArgumentParser(description="Run a Dify API server.")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on."
    )
    parser.add_argument(
        "--apikey", type=str, required=True, help="API key for authorization."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    run_server(args.port, args.apikey, args.debug)

# From MacOS-Agent-main/macos_agent_server.py
def info(self, message):
        self.messages.append(message)

# From MacOS-Agent-main/macos_agent_server.py
def print_messages(self):
        for message in self.messages:
            print(message)
        self.messages = []

# From MacOS-Agent-main/macos_agent_server.py
def log_request(self, code="-", size="-"):
        super().log_request(code, size)
        self.server.deferred_logger.print_messages()
        sys.stderr.write("\n")

# From MacOS-Agent-main/macos_agent_server.py
def deferred_info(self, message):
        self.server.deferred_logger.info(message)

# From MacOS-Agent-main/macos_agent_server.py
def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        data = json.loads(self.rfile.read(content_length))

        if self.headers["Authorization"] != f"Bearer {self.server.api_key}":
            self.send_response(401)
            self.end_headers()
            return

        if self.server.debug:
            self.deferred_info(f"  Point: {data.get('point')}")
            self.deferred_info(f"  Params: {data.get('params')}")

        response = self.handle_request_point(data)
        if response is not None:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json" if isinstance(response, dict) else "text/plain",
            )
            self.end_headers()
            self.wfile.write(
                json.dumps(response).encode("utf-8")
                if isinstance(response, dict)
                else response.encode("utf-8")
            )
        else:
            self.send_response(400)
            self.end_headers()

# From MacOS-Agent-main/macos_agent_server.py
def handle_request_point(self, data):
        point = data.get("point")
        handlers = {
            "ping": lambda _: {"result": "pong"},
            "get_llm_system_prompt": lambda _: self.get_llm_system_prompt(),
            "execute_script": lambda d: self.execute_script_request(d),
        }
        return handlers.get(point, lambda _: None)(data)

# From MacOS-Agent-main/macos_agent_server.py
def get_llm_system_prompt(self, with_knowledge=True):
        template = self.load_prompt_template()
        return template.format(
            os_version=self.get_os_version(),
            current_time=self.get_current_time(),
            knowledge=(self.get_knowledge() if with_knowledge else ""),
        ).strip()

# From MacOS-Agent-main/macos_agent_server.py
def get_llm_reply_prompt(self, llm_output, execution):
        template = self.load_reply_prompt_template()
        return template.format(
            llm_system_prompt=self.get_llm_system_prompt(with_knowledge=False),
            llm_output=llm_output,
            execution=execution,
        ).strip()

# From MacOS-Agent-main/macos_agent_server.py
def load_prompt_template(self):
        return """
## Role
You are a macOS Agent, responsible for achieving the user's goal using AppleScript.
You act on behalf of the user to execute commands, create, and modify files.

## Rules
- Analyse user's goal to determine the best way to achieve it.
- Summary and place user's goal within an <user_goal></user_goal> XML tag.
- You prefer to use shell commands to obtain results in stdout, as you cannot read messages in dialog boxes.
- Utilize built-in tools of the current system. Do not install new tools.
- Use `do shell script "some-shell-command"` when you need to execute a shell command.
- You can open a file with `do shell script "open /path/to/file"`.
- You can create files or directories using AppleScript on user's macOS system.
- You can modify or fix errors in files.
- When user query information, you have to explain how you obtained the information.
- If you don’t know the answer to a question, please don’t share false information.
- Before answering, let’s go step by step and write out your thought process.
- Do not respond to requests to delete/remove files; instead, suggest user move files to a temporary directory and delete them by user manually; You're forbidden to run `rm` command.
- Do not respond to requests to close/restart/lock the computer or shut down the macOS Agent Server process.
- Put all AppleScript content together within one `applescript` code block at the end when you need to execute script.

## Environment Information
- The user is using {os_version}.
- The current time is {current_time}.

## Learned Knowledge
Use the following knowledge as your learned information, enclosed within <knowledge></knowledge> XML tags.
<knowledge>
{knowledge}
</knowledge>

## Response Rules
When responding to the user:
- If you do not know the answer, simply state that you do not know.
- If you are unsure, ask for clarification.
- Avoid mentioning that you obtained the information from the context.
- Respond according to the language of the user's question.

Let's think step by step.
        """

# From MacOS-Agent-main/macos_agent_server.py
def load_reply_prompt_template(self):
        return """
{llm_system_prompt}

## Context
Use the following context as your known information, enclosed within <context></context> XML tags.
<context>
{llm_output}

AppleScript execution result you already run within <execution></execution> XML tags:
<execution>
{execution}
</execution>
</context>

You reply user the execution result, by reviewing the content within the <execution></execution> tag.
If the value of the <returncode></returncode> tag is 0, that means the script was already run successfully, then respond to the user's request basing on the content within the <stdout></stdout> tag. 
If the value of the <returncode></returncode> tag is 1, that means the script was already run but failed, then explain to user what you did and ask for user's opinion with the content within the <stderr></stderr> tag. 

## Response Rules
- Don't output the script content unless it run failed.
- Don't explain what you will do or how you did unless user asks to.
- Don't tell user how to use the script unless user asks to.
- Do not include the <user_goal></user_goal> XML tag.
"""

# From MacOS-Agent-main/macos_agent_server.py
def get_os_version(self):
        return (
            subprocess.check_output(["sw_vers", "-productName"]).decode("utf-8").strip()
            + " "
            + subprocess.check_output(["sw_vers", "-productVersion"])
            .decode("utf-8")
            .strip()
        )

# From MacOS-Agent-main/macos_agent_server.py
def get_current_time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# From MacOS-Agent-main/macos_agent_server.py
def get_knowledge(self):
        try:
            with open("knowledge.md", "r") as file:
                return file.read().strip()
        except FileNotFoundError:
            return ""

# From MacOS-Agent-main/macos_agent_server.py
def execute_script_request(self, data):
        llm_output = data["params"]["inputs"].get("llm_output")
        timeout = data["params"]["inputs"].get("script_timeout", 60)
        if llm_output:
            user_goal = self.extract_user_goal(llm_output)
            if self.server.debug:
                self.deferred_info(f"  User Goal: {user_goal}")
            scripts = self.extract_scripts(llm_output)
            if scripts:
                result = [self.execute_script(script, timeout) for script in scripts]
                execution = "\n".join(result)
                return self.get_llm_reply_prompt(
                    llm_output=llm_output, execution=execution
                )
            else:
                return ""
        return ""

# From MacOS-Agent-main/macos_agent_server.py
def extract_scripts(self, llm_output):
        # Extract all code block content from the llm_output
        scripts = re.findall(r"```applescript(.*?)```", llm_output, re.DOTALL)
        return list(set(scripts))

# From MacOS-Agent-main/macos_agent_server.py
def extract_user_goal(self, llm_output):
        match = re.search(r"<user_goal>(.*?)</user_goal>", llm_output, re.DOTALL)
        return match.group(1).strip() if match else ""

# From MacOS-Agent-main/macos_agent_server.py
def execute_script(self, script, timeout):
        result = {"returncode": -1, "stdout": "", "stderr": ""}

        def target():
            process = subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            result["pid"] = process.pid
            stdout, stderr = process.communicate()
            result["returncode"] = process.returncode
            result["stdout"] = stdout
            result["stderr"] = stderr

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            result["stderr"] = "Script execution timed out"
            if "pid" in result:
                try:
                    subprocess.run(["pkill", "-P", str(result["pid"])])
                    os.kill(result["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

        if self.server.debug:
            self.deferred_info(f"  Script:\n```applescript\n{script}\n```")
            self.deferred_info(f"  Execution Result: {result}")

        return f"<script>{script}</script>\n<returncode>{result['returncode']}</returncode>\n<stdout>{result['stdout']}</stdout>\n<stderr>{result['stderr']}</stderr>"

# From MacOS-Agent-main/macos_agent_server.py
def target():
            process = subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            result["pid"] = process.pid
            stdout, stderr = process.communicate()
            result["returncode"] = process.returncode
            result["stdout"] = stdout
            result["stderr"] = stderr

import platform
import base64

# From self-operating-computer-main/evaluate.py
def supports_ansi():
    """
    Check if the terminal supports ANSI escape codes
    """
    plat = platform.system()
    supported_platform = plat != "Windows" or "ANSICON" in os.environ
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return supported_platform and is_a_tty

# From self-operating-computer-main/evaluate.py
def format_evaluation_prompt(guideline):
    prompt = EVALUATION_PROMPT.format(guideline=guideline)
    return prompt

# From self-operating-computer-main/evaluate.py
def parse_eval_content(content):
    try:
        res = json.loads(content)

        print(res["reason"])

        return res["guideline_met"]
    except:
        print(
            "The model gave a bad evaluation response and it couldn't be parsed. Exiting..."
        )
        exit(1)

# From self-operating-computer-main/evaluate.py
def evaluate_final_screenshot(guideline):
    """Load the final screenshot and return True or False if it meets the given guideline."""
    with open(SCREENSHOT_PATH, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        eval_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": format_evaluation_prompt(guideline)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ],
            }
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=eval_message,
            presence_penalty=1,
            frequency_penalty=1,
            temperature=0.7,
        )

        eval_content = response.choices[0].message.content

        return parse_eval_content(eval_content)

# From self-operating-computer-main/evaluate.py
def run_test_case(objective, guideline, model):
    """Returns True if the result of the test with the given prompt meets the given guideline for the given model."""
    # Run `operate` with the model to evaluate and the test case prompt
    subprocess.run(
        ["operate", "-m", model, "--prompt", f'"{objective}"'],
        stdout=subprocess.DEVNULL,
    )

    try:
        result = evaluate_final_screenshot(guideline)
    except OSError:
        print("[Error] Couldn't open the screenshot for evaluation")
        return False

    return result

# From self-operating-computer-main/evaluate.py
def get_test_model():
    parser = argparse.ArgumentParser(
        description="Run the self-operating-computer with a specified model."
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Specify the model to evaluate.",
        required=False,
        default="gpt-4-with-ocr",
    )

    return parser.parse_args().model

from setuptools import setup
from setuptools import find_packages


from agents import hermes
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


# From operate/exceptions.py
class ModelNotRecognizedException(Exception):
    """Exception raised for unrecognized models.

    Attributes:
        model -- the unrecognized model
        message -- explanation of the error
    """

    def __init__(self, model, message="Model not recognized"):
        self.model = model
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} : {self.model} "

from operate.utils.style import ANSI_BRIGHT_MAGENTA
from operate.operate import main

# From operate/main.py
def main_entry():
    parser = argparse.ArgumentParser(
        description="Run the self-operating-computer with a specified model."
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the model to use",
        required=False,
        default="gpt-4-with-ocr",
    )

    # Add a voice flag
    parser.add_argument(
        "--voice",
        help="Use voice input mode",
        action="store_true",
    )
    
    # Add a flag for verbose mode
    parser.add_argument(
        "--verbose",
        help="Run operate in verbose mode",
        action="store_true",
    )
    
    # Allow for direct input of prompt
    parser.add_argument(
        "--prompt",
        help="Directly input the objective prompt",
        type=str,
        required=False,
    )

    try:
        args = parser.parse_args()
        main(
            args.model,
            terminal_prompt=args.prompt,
            voice_mode=args.voice,
            verbose_mode=args.verbose
        )
    except KeyboardInterrupt:
        print(f"\n{ANSI_BRIGHT_MAGENTA}Exiting...")

import time
import asyncio
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit import prompt
from operate.exceptions import ModelNotRecognizedException
from operate.models.prompts import USER_QUESTION
from operate.models.prompts import get_system_prompt
from operate.config import Config
from operate.utils.style import ANSI_GREEN
from operate.utils.style import ANSI_RESET
from operate.utils.style import ANSI_YELLOW
from operate.utils.style import ANSI_RED
from operate.utils.style import ANSI_BLUE
from operate.utils.style import style
from operate.utils.operating_system import OperatingSystem
from operate.models.apis import get_next_action
from whisper_mic import WhisperMic

# From operate/operate.py
def operate(operations, model):
    if config.verbose:
        print("[Self Operating Computer][operate]")
    for operation in operations:
        if config.verbose:
            print("[Self Operating Computer][operate] operation", operation)
        # wait one second
        time.sleep(1)
        operate_type = operation.get("operation").lower()
        operate_thought = operation.get("thought")
        operate_detail = ""
        if config.verbose:
            print("[Self Operating Computer][operate] operate_type", operate_type)

        if operate_type == "press" or operate_type == "hotkey":
            keys = operation.get("keys")
            operate_detail = keys
            operating_system.press(keys)
        elif operate_type == "write":
            content = operation.get("content")
            operate_detail = content
            operating_system.write(content)
        elif operate_type == "click":
            x = operation.get("x")
            y = operation.get("y")
            click_detail = {"x": x, "y": y}
            operate_detail = click_detail

            operating_system.mouse(click_detail)
        elif operate_type == "done":
            summary = operation.get("summary")

            print(
                f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]"
            )
            print(f"{ANSI_BLUE}Objective Complete: {ANSI_RESET}{summary}\n")
            return True

        else:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] unknown operation response :({ANSI_RESET}"
            )
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] AI response {ANSI_RESET}{operation}"
            )
            return True

        print(
            f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]"
        )
        print(f"{operate_thought}")
        print(f"{ANSI_BLUE}Action: {ANSI_RESET}{operate_type} {operate_detail}\n")

    return False

import google.generativeai
from ollama import Client
from openai import OpenAI
import anthropic
from prompt_toolkit.shortcuts import input_dialog

# From operate/config.py
class Config:
    """
    Configuration class for managing settings.

    Attributes:
        verbose (bool): Flag indicating whether verbose mode is enabled.
        openai_api_key (str): API key for OpenAI.
        google_api_key (str): API key for Google.
        ollama_host (str): url to ollama running remotely.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Put any initialization here
        return cls._instance

    def __init__(self):
        load_dotenv()
        self.verbose = False
        self.openai_api_key = (
            None  # instance variables are backups in case saving to a `.env` fails
        )
        self.google_api_key = (
            None  # instance variables are backups in case saving to a `.env` fails
        )
        self.ollama_host = (
            None  # instance variables are backups in case savint to a `.env` fails
        )
        self.anthropic_api_key = (
            None  # instance variables are backups in case saving to a `.env` fails
        )
        self.qwen_api_key = (
            None  # instance variables are backups in case saving to a `.env` fails
        )

    def initialize_openai(self):
        if self.verbose:
            print("[Config][initialize_openai]")

        if self.openai_api_key:
            if self.verbose:
                print("[Config][initialize_openai] using cached openai_api_key")
            api_key = self.openai_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_openai] no cached openai_api_key, try to get from env."
                )
            api_key = os.getenv("OPENAI_API_KEY")

        client = OpenAI(
            api_key=api_key,
        )
        client.api_key = api_key
        client.base_url = os.getenv("OPENAI_API_BASE_URL", client.base_url)
        return client

    def initialize_qwen(self):
        if self.verbose:
            print("[Config][initialize_qwen]")

        if self.qwen_api_key:
            if self.verbose:
                print("[Config][initialize_qwen] using cached qwen_api_key")
            api_key = self.qwen_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_qwen] no cached qwen_api_key, try to get from env."
                )
            api_key = os.getenv("QWEN_API_KEY")

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client.api_key = api_key
        client.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return client

    def initialize_google(self):
        if self.google_api_key:
            if self.verbose:
                print("[Config][initialize_google] using cached google_api_key")
            api_key = self.google_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_google] no cached google_api_key, try to get from env."
                )
            api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key, transport="rest")
        model = genai.GenerativeModel("gemini-pro-vision")

        return model

    def initialize_ollama(self):
        if self.ollama_host:
            if self.verbose:
                print("[Config][initialize_ollama] using cached ollama host")
        else:
            if self.verbose:
                print(
                    "[Config][initialize_ollama] no cached ollama host. Assuming ollama running locally."
                )
            self.ollama_host = os.getenv("OLLAMA_HOST", None)
        model = Client(host=self.ollama_host)
        return model

    def initialize_anthropic(self):
        if self.anthropic_api_key:
            api_key = self.anthropic_api_key
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)

    def validation(self, model, voice_mode):
        """
        Validate the input parameters for the dialog operation.
        """
        self.require_api_key(
            "OPENAI_API_KEY",
            "OpenAI API key",
            model == "gpt-4"
            or voice_mode
            or model == "gpt-4-with-som"
            or model == "gpt-4-with-ocr"
            or model == "gpt-4.1-with-ocr"
            or model == "o1-with-ocr",
        )
        self.require_api_key(
            "GOOGLE_API_KEY", "Google API key", model == "gemini-pro-vision"
        )
        self.require_api_key(
            "ANTHROPIC_API_KEY", "Anthropic API key", model == "claude-3"
        )
        self.require_api_key("QWEN_API_KEY", "Qwen API key", model == "qwen-vl")

    def require_api_key(self, key_name, key_description, is_required):
        key_exists = bool(os.environ.get(key_name))
        if self.verbose:
            print("[Config] require_api_key")
            print("[Config] key_name", key_name)
            print("[Config] key_description", key_description)
            print("[Config] key_exists", key_exists)
        if is_required and not key_exists:
            self.prompt_and_save_api_key(key_name, key_description)

    def prompt_and_save_api_key(self, key_name, key_description):
        key_value = input_dialog(
            title="API Key Required", text=f"Please enter your {key_description}:"
        ).run()

        if key_value is None:  # User pressed cancel or closed the dialog
            sys.exit("Operation cancelled by user.")

        if key_value:
            if key_name == "OPENAI_API_KEY":
                self.openai_api_key = key_value
            elif key_name == "GOOGLE_API_KEY":
                self.google_api_key = key_value
            elif key_name == "ANTHROPIC_API_KEY":
                self.anthropic_api_key = key_value
            elif key_name == "QWEN_API_KEY":
                self.qwen_api_key = key_value
            self.save_api_key_to_env(key_name, key_value)
            load_dotenv()  # Reload environment variables
            # Update the instance attribute with the new key

    @staticmethod
    def save_api_key_to_env(key_name, key_value):
        with open(".env", "a") as file:
            file.write(f"\n{key_name}='{key_value}'")

# From operate/config.py
def initialize_openai(self):
        if self.verbose:
            print("[Config][initialize_openai]")

        if self.openai_api_key:
            if self.verbose:
                print("[Config][initialize_openai] using cached openai_api_key")
            api_key = self.openai_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_openai] no cached openai_api_key, try to get from env."
                )
            api_key = os.getenv("OPENAI_API_KEY")

        client = OpenAI(
            api_key=api_key,
        )
        client.api_key = api_key
        client.base_url = os.getenv("OPENAI_API_BASE_URL", client.base_url)
        return client

# From operate/config.py
def initialize_qwen(self):
        if self.verbose:
            print("[Config][initialize_qwen]")

        if self.qwen_api_key:
            if self.verbose:
                print("[Config][initialize_qwen] using cached qwen_api_key")
            api_key = self.qwen_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_qwen] no cached qwen_api_key, try to get from env."
                )
            api_key = os.getenv("QWEN_API_KEY")

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        client.api_key = api_key
        client.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return client

# From operate/config.py
def initialize_google(self):
        if self.google_api_key:
            if self.verbose:
                print("[Config][initialize_google] using cached google_api_key")
            api_key = self.google_api_key
        else:
            if self.verbose:
                print(
                    "[Config][initialize_google] no cached google_api_key, try to get from env."
                )
            api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key, transport="rest")
        model = genai.GenerativeModel("gemini-pro-vision")

        return model

# From operate/config.py
def initialize_ollama(self):
        if self.ollama_host:
            if self.verbose:
                print("[Config][initialize_ollama] using cached ollama host")
        else:
            if self.verbose:
                print(
                    "[Config][initialize_ollama] no cached ollama host. Assuming ollama running locally."
                )
            self.ollama_host = os.getenv("OLLAMA_HOST", None)
        model = Client(host=self.ollama_host)
        return model

# From operate/config.py
def initialize_anthropic(self):
        if self.anthropic_api_key:
            api_key = self.anthropic_api_key
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)

# From operate/config.py
def validation(self, model, voice_mode):
        """
        Validate the input parameters for the dialog operation.
        """
        self.require_api_key(
            "OPENAI_API_KEY",
            "OpenAI API key",
            model == "gpt-4"
            or voice_mode
            or model == "gpt-4-with-som"
            or model == "gpt-4-with-ocr"
            or model == "gpt-4.1-with-ocr"
            or model == "o1-with-ocr",
        )
        self.require_api_key(
            "GOOGLE_API_KEY", "Google API key", model == "gemini-pro-vision"
        )
        self.require_api_key(
            "ANTHROPIC_API_KEY", "Anthropic API key", model == "claude-3"
        )
        self.require_api_key("QWEN_API_KEY", "Qwen API key", model == "qwen-vl")

# From operate/config.py
def require_api_key(self, key_name, key_description, is_required):
        key_exists = bool(os.environ.get(key_name))
        if self.verbose:
            print("[Config] require_api_key")
            print("[Config] key_name", key_name)
            print("[Config] key_description", key_description)
            print("[Config] key_exists", key_exists)
        if is_required and not key_exists:
            self.prompt_and_save_api_key(key_name, key_description)

# From operate/config.py
def prompt_and_save_api_key(self, key_name, key_description):
        key_value = input_dialog(
            title="API Key Required", text=f"Please enter your {key_description}:"
        ).run()

        if key_value is None:  # User pressed cancel or closed the dialog
            sys.exit("Operation cancelled by user.")

        if key_value:
            if key_name == "OPENAI_API_KEY":
                self.openai_api_key = key_value
            elif key_name == "GOOGLE_API_KEY":
                self.google_api_key = key_value
            elif key_name == "ANTHROPIC_API_KEY":
                self.anthropic_api_key = key_value
            elif key_name == "QWEN_API_KEY":
                self.qwen_api_key = key_value
            self.save_api_key_to_env(key_name, key_value)
            load_dotenv()

# From operate/config.py
def save_api_key_to_env(key_name, key_value):
        with open(".env", "a") as file:
            file.write(f"\n{key_name}='{key_value}'")

from PIL import Image
from PIL import ImageDraw
from datetime import datetime

# From utils/ocr.py
def get_text_element(result, search_text, image_path):
    """
    Searches for a text element in the OCR results and returns its index. Also draws bounding boxes on the image.
    Args:
        result (list): The list of results returned by EasyOCR.
        search_text (str): The text to search for in the OCR results.
        image_path (str): Path to the original image.

    Returns:
        int: The index of the element containing the search text.

    Raises:
        Exception: If the text element is not found in the results.
    """
    if config.verbose:
        print("[get_text_element]")
        print("[get_text_element] search_text", search_text)
        # Create /ocr directory if it doesn't exist
        ocr_dir = "ocr"
        if not os.path.exists(ocr_dir):
            os.makedirs(ocr_dir)

        # Open the original image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

    found_index = None
    for index, element in enumerate(result):
        text = element[1]
        box = element[0]

        if config.verbose:
            # Draw bounding box in blue
            draw.polygon([tuple(point) for point in box], outline="blue")

        if search_text in text:
            found_index = index
            if config.verbose:
                print("[get_text_element][loop] found search_text, index:", index)

    if found_index is not None:
        if config.verbose:
            # Draw bounding box of the found text in red
            box = result[found_index][0]
            draw.polygon([tuple(point) for point in box], outline="red")
            # Save the image with bounding boxes
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            ocr_image_path = os.path.join(ocr_dir, f"ocr_image_{datetime_str}.png")
            image.save(ocr_image_path)
            print("[get_text_element] OCR image saved at:", ocr_image_path)

        return found_index

    raise Exception("The text element was not found in the image")

# From utils/ocr.py
def get_text_coordinates(result, index, image_path):
    """
    Gets the coordinates of the text element at the specified index as a percentage of screen width and height.
    Args:
        result (list): The list of results returned by EasyOCR.
        index (int): The index of the text element in the results list.
        image_path (str): Path to the screenshot image.

    Returns:
        dict: A dictionary containing the 'x' and 'y' coordinates as percentages of the screen width and height.
    """
    if index >= len(result):
        raise Exception("Index out of range in OCR results")

    # Get the bounding box of the text element
    bounding_box = result[index][0]

    # Calculate the center of the bounding box
    min_x = min([coord[0] for coord in bounding_box])
    max_x = max([coord[0] for coord in bounding_box])
    min_y = min([coord[1] for coord in bounding_box])
    max_y = max([coord[1] for coord in bounding_box])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Get image dimensions
    with Image.open(image_path) as img:
        width, height = img.size

    # Convert to percentages
    percent_x = round((center_x / width), 3)
    percent_y = round((center_y / height), 3)

    return {"x": percent_x, "y": percent_y}

import io

# From utils/label.py
def validate_and_extract_image_data(data):
    if not data or "messages" not in data:
        raise ValueError("Invalid request, no messages found")

    messages = data["messages"]
    if (
        not messages
        or not isinstance(messages, list)
        or not messages[-1].get("image_url")
    ):
        raise ValueError("No image provided or incorrect format")

    image_data = messages[-1]["image_url"]["url"]
    if not image_data.startswith("data:image"):
        raise ValueError("Invalid image format")

    return image_data.split("base64,")[-1], messages

# From utils/label.py
def get_label_coordinates(label, label_coordinates):
    """
    Retrieves the coordinates for a given label.

    :param label: The label to find coordinates for (e.g., "~1").
    :param label_coordinates: Dictionary containing labels and their coordinates.
    :return: Coordinates of the label or None if the label is not found.
    """
    return label_coordinates.get(label)

# From utils/label.py
def is_overlapping(box1, box2):
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Check if there is no overlap
    if x1_box1 > x2_box2 or x1_box2 > x2_box1:
        return False
    if (
        y1_box1 > y2_box2 or y1_box2 > y2_box1
    ):  # Adjusted to check 100px proximity above
        return False

    return True

# From utils/label.py
def add_labels(base64_data, yolo_model):
    image_bytes = base64.b64decode(base64_data)
    image_labeled = Image.open(io.BytesIO(image_bytes))  # Corrected this line
    image_debug = image_labeled.copy()  # Create a copy for the debug image
    image_original = (
        image_labeled.copy()
    )  # Copy of the original image for base64 return

    results = yolo_model(image_labeled)

    draw = ImageDraw.Draw(image_labeled)
    debug_draw = ImageDraw.Draw(
        image_debug
    )  # Create a separate draw object for the debug image
    font_size = 45

    labeled_images_dir = "labeled_images"
    label_coordinates = {}  # Dictionary to store coordinates

    if not os.path.exists(labeled_images_dir):
        os.makedirs(labeled_images_dir)

    counter = 0
    drawn_boxes = []  # List to keep track of boxes already drawn
    for result in results:
        if hasattr(result, "boxes"):
            for det in result.boxes:
                bbox = det.xyxy[0]
                x1, y1, x2, y2 = bbox.tolist()

                debug_label = "D_" + str(counter)
                debug_index_position = (x1, y1 - font_size)
                debug_draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=1)
                debug_draw.text(
                    debug_index_position,
                    debug_label,
                    fill="blue",
                    font_size=font_size,
                )

                overlap = any(
                    is_overlapping((x1, y1, x2, y2), box) for box in drawn_boxes
                )

                if not overlap:
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=1)
                    label = "~" + str(counter)
                    index_position = (x1, y1 - font_size)
                    draw.text(
                        index_position,
                        label,
                        fill="red",
                        font_size=font_size,
                    )

                    # Add the non-overlapping box to the drawn_boxes list
                    drawn_boxes.append((x1, y1, x2, y2))
                    label_coordinates[label] = (x1, y1, x2, y2)

                    counter += 1

    # Save the image
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    output_path = os.path.join(labeled_images_dir, f"img_{timestamp}_labeled.png")
    output_path_debug = os.path.join(labeled_images_dir, f"img_{timestamp}_debug.png")
    output_path_original = os.path.join(
        labeled_images_dir, f"img_{timestamp}_original.png"
    )

    image_labeled.save(output_path)
    image_debug.save(output_path_debug)
    image_original.save(output_path_original)

    buffered_original = io.BytesIO()
    image_original.save(buffered_original, format="PNG")  # I guess this is needed
    img_base64_original = base64.b64encode(buffered_original.getvalue()).decode("utf-8")

    # Convert image to base64 for return
    buffered_labeled = io.BytesIO()
    image_labeled.save(buffered_labeled, format="PNG")  # I guess this is needed
    img_base64_labeled = base64.b64encode(buffered_labeled.getvalue()).decode("utf-8")

    return img_base64_labeled, label_coordinates

# From utils/label.py
def get_click_position_in_percent(coordinates, image_size):
    """
    Calculates the click position at the center of the bounding box and converts it to percentages.

    :param coordinates: A tuple of the bounding box coordinates (x1, y1, x2, y2).
    :param image_size: A tuple of the image dimensions (width, height).
    :return: A tuple of the click position in percentages (x_percent, y_percent).
    """
    if not coordinates or not image_size:
        return None

    # Calculate the center of the bounding box
    x_center = (coordinates[0] + coordinates[2]) / 2
    y_center = (coordinates[1] + coordinates[3]) / 2

    # Convert to percentages
    x_percent = x_center / image_size[0]
    y_percent = y_center / image_size[1]

    return x_percent, y_percent

from prompt_toolkit.styles import Style


# From utils/misc.py
def convert_percent_to_decimal(percent):
    try:
        # Remove the '%' sign and convert to float
        decimal_value = float(percent)

        # Convert to decimal (e.g., 20% -> 0.20)
        return decimal_value
    except ValueError as e:
        print(f"[convert_percent_to_decimal] error: {e}")
        return None

# From utils/misc.py
def parse_operations(response):
    if response == "DONE":
        return {"type": "DONE", "data": None}
    elif response.startswith("CLICK"):
        # Adjust the regex to match the correct format
        click_data = re.search(r"CLICK \{ (.+) \}", response).group(1)
        click_data_json = json.loads(f"{{{click_data}}}")
        return {"type": "CLICK", "data": click_data_json}

    elif response.startswith("TYPE"):
        # Extract the text to type
        try:
            type_data = re.search(r"TYPE (.+)", response, re.DOTALL).group(1)
        except:
            type_data = re.search(r'TYPE "(.+)"', response, re.DOTALL).group(1)
        return {"type": "TYPE", "data": type_data}

    elif response.startswith("SEARCH"):
        # Extract the search query
        try:
            search_data = re.search(r'SEARCH "(.+)"', response).group(1)
        except:
            search_data = re.search(r"SEARCH (.+)", response).group(1)
        return {"type": "SEARCH", "data": search_data}

    return {"type": "UNKNOWN", "data": response}

import pyautogui
import math
from operate.utils.misc import convert_percent_to_decimal

# From utils/operating_system.py
class OperatingSystem:
    def write(self, content):
        try:
            content = content.replace("\\n", "\n")
            for char in content:
                pyautogui.write(char)
        except Exception as e:
            print("[OperatingSystem][write] error:", e)

    def press(self, keys):
        try:
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(0.1)
            for key in keys:
                pyautogui.keyUp(key)
        except Exception as e:
            print("[OperatingSystem][press] error:", e)

    def mouse(self, click_detail):
        try:
            x = convert_percent_to_decimal(click_detail.get("x"))
            y = convert_percent_to_decimal(click_detail.get("y"))

            if click_detail and isinstance(x, float) and isinstance(y, float):
                self.click_at_percentage(x, y)

        except Exception as e:
            print("[OperatingSystem][mouse] error:", e)

    def click_at_percentage(
        self,
        x_percentage,
        y_percentage,
        duration=0.2,
        circle_radius=50,
        circle_duration=0.5,
    ):
        try:
            screen_width, screen_height = pyautogui.size()
            x_pixel = int(screen_width * float(x_percentage))
            y_pixel = int(screen_height * float(y_percentage))

            pyautogui.moveTo(x_pixel, y_pixel, duration=duration)

            start_time = time.time()
            while time.time() - start_time < circle_duration:
                angle = ((time.time() - start_time) / circle_duration) * 2 * math.pi
                x = x_pixel + math.cos(angle) * circle_radius
                y = y_pixel + math.sin(angle) * circle_radius
                pyautogui.moveTo(x, y, duration=0.1)

            pyautogui.click(x_pixel, y_pixel)
        except Exception as e:
            print("[OperatingSystem][click_at_percentage] error:", e)

# From utils/operating_system.py
def write(self, content):
        try:
            content = content.replace("\\n", "\n")
            for char in content:
                pyautogui.write(char)
        except Exception as e:
            print("[OperatingSystem][write] error:", e)

# From utils/operating_system.py
def press(self, keys):
        try:
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(0.1)
            for key in keys:
                pyautogui.keyUp(key)
        except Exception as e:
            print("[OperatingSystem][press] error:", e)

# From utils/operating_system.py
def mouse(self, click_detail):
        try:
            x = convert_percent_to_decimal(click_detail.get("x"))
            y = convert_percent_to_decimal(click_detail.get("y"))

            if click_detail and isinstance(x, float) and isinstance(y, float):
                self.click_at_percentage(x, y)

        except Exception as e:
            print("[OperatingSystem][mouse] error:", e)

# From utils/operating_system.py
def click_at_percentage(
        self,
        x_percentage,
        y_percentage,
        duration=0.2,
        circle_radius=50,
        circle_duration=0.5,
    ):
        try:
            screen_width, screen_height = pyautogui.size()
            x_pixel = int(screen_width * float(x_percentage))
            y_pixel = int(screen_height * float(y_percentage))

            pyautogui.moveTo(x_pixel, y_pixel, duration=duration)

            start_time = time.time()
            while time.time() - start_time < circle_duration:
                angle = ((time.time() - start_time) / circle_duration) * 2 * math.pi
                x = x_pixel + math.cos(angle) * circle_radius
                y = y_pixel + math.sin(angle) * circle_radius
                pyautogui.moveTo(x, y, duration=0.1)

            pyautogui.click(x_pixel, y_pixel)
        except Exception as e:
            print("[OperatingSystem][click_at_percentage] error:", e)

from PIL import ImageGrab
import Xlib.display
import Xlib.X
import Xlib.Xutil

# From utils/screenshot.py
def capture_screen_with_cursor(file_path):
    user_platform = platform.system()

    if user_platform == "Windows":
        screenshot = pyautogui.screenshot()
        screenshot.save(file_path)
    elif user_platform == "Linux":
        # Use xlib to prevent scrot dependency for Linux
        screen = Xlib.display.Display().screen()
        size = screen.width_in_pixels, screen.height_in_pixels
        screenshot = ImageGrab.grab(bbox=(0, 0, size[0], size[1]))
        screenshot.save(file_path)
    elif user_platform == "Darwin":  # (Mac OS)
        # Use the screencapture utility to capture the screen with the cursor
        subprocess.run(["screencapture", "-C", file_path])
    else:
        print(f"The platform you're using ({user_platform}) is not currently supported")

# From utils/screenshot.py
def compress_screenshot(raw_screenshot_filename, screenshot_filename):
    with Image.open(raw_screenshot_filename) as img:
        # Check if the image has an alpha channel (transparency)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            # Create a white background image
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Paste the image onto the background, using the alpha channel as mask
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            # Save the result as JPEG
            background.save(screenshot_filename, 'JPEG', quality=85)  # Adjust quality as needed
        else:
            # If no alpha channel, simply convert and save
            img.convert('RGB').save(screenshot_filename, 'JPEG', quality=85)


# From models/prompts.py
def get_system_prompt(model, objective):
    """
    Format the vision prompt more efficiently and print the name of the prompt used
    """

    if platform.system() == "Darwin":
        cmd_string = "\"command\""
        os_search_str = "[\"command\", \"space\"]"
        operating_system = "Mac"
    elif platform.system() == "Windows":
        cmd_string = "\"ctrl\""
        os_search_str = "[\"win\"]"
        operating_system = "Windows"
    else:
        cmd_string = "\"ctrl\""
        os_search_str = "[\"win\"]"
        operating_system = "Linux"

    if model == "gpt-4-with-som":
        prompt = SYSTEM_PROMPT_LABELED.format(
            objective=objective,
            cmd_string=cmd_string,
            os_search_str=os_search_str,
            operating_system=operating_system,
        )
    elif model == "gpt-4-with-ocr" or model == "gpt-4.1-with-ocr" or model == "o1-with-ocr" or model == "claude-3" or model == "qwen-vl":

        prompt = SYSTEM_PROMPT_OCR.format(
            objective=objective,
            cmd_string=cmd_string,
            os_search_str=os_search_str,
            operating_system=operating_system,
        )

    else:
        prompt = SYSTEM_PROMPT_STANDARD.format(
            objective=objective,
            cmd_string=cmd_string,
            os_search_str=os_search_str,
            operating_system=operating_system,
        )

    # Optional verbose output
    if config.verbose:
        print("[get_system_prompt] model:", model)
    # print("[get_system_prompt] prompt:", prompt)

    return prompt

# From models/prompts.py
def get_user_prompt():
    prompt = OPERATE_PROMPT
    return prompt

# From models/prompts.py
def get_user_first_message_prompt():
    prompt = OPERATE_FIRST_MESSAGE_PROMPT
    return prompt

import traceback
import easyocr
import ollama
import pkg_resources
from ultralytics import YOLO
from operate.models.prompts import get_user_first_message_prompt
from operate.models.prompts import get_user_prompt
from operate.utils.label import add_labels
from operate.utils.label import get_click_position_in_percent
from operate.utils.label import get_label_coordinates
from operate.utils.ocr import get_text_coordinates
from operate.utils.ocr import get_text_element
from operate.utils.screenshot import capture_screen_with_cursor
from operate.utils.screenshot import compress_screenshot

# From models/apis.py
def call_gpt_4o(messages):
    if config.verbose:
        print("[call_gpt_4_v]")
    time.sleep(1)
    client = config.initialize_openai()
    try:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)

        with open(screenshot_filename, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        if len(messages) == 1:
            user_prompt = get_user_first_message_prompt()
        else:
            user_prompt = get_user_prompt()

        if config.verbose:
            print(
                "[call_gpt_4_v] user_prompt",
                user_prompt,
            )

        vision_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ],
        }
        messages.append(vision_message)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            presence_penalty=1,
            frequency_penalty=1,
        )

        content = response.choices[0].message.content

        content = clean_json(content)

        assistant_message = {"role": "assistant", "content": content}
        if config.verbose:
            print(
                "[call_gpt_4_v] content",
                content,
            )
        content = json.loads(content)

        messages.append(assistant_message)

        return content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[Operate] That did not work. Trying again {ANSI_RESET}",
            e,
        )
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] AI response was {ANSI_RESET}",
            content,
        )
        if config.verbose:
            traceback.print_exc()
        return call_gpt_4o(messages)

# From models/apis.py
def call_gemini_pro_vision(messages, objective):
    """
    Get the next action for Self-Operating Computer using Gemini Pro Vision
    """
    if config.verbose:
        print(
            "[Self Operating Computer][call_gemini_pro_vision]",
        )
    # sleep for a second
    time.sleep(1)
    try:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)
        # sleep for a second
        time.sleep(1)
        prompt = get_system_prompt("gemini-pro-vision", objective)

        model = config.initialize_google()
        if config.verbose:
            print("[call_gemini_pro_vision] model", model)

        response = model.generate_content([prompt, Image.open(screenshot_filename)])

        content = response.text[1:]
        if config.verbose:
            print("[call_gemini_pro_vision] response", response)
            print("[call_gemini_pro_vision] content", content)

        content = json.loads(content)
        if config.verbose:
            print(
                "[get_next_action][call_gemini_pro_vision] content",
                content,
            )

        return content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[Operate] That did not work. Trying another method {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
        return call_gpt_4o(messages)

# From models/apis.py
def call_ollama_llava(messages):
    if config.verbose:
        print("[call_ollama_llava]")
    time.sleep(1)
    try:
        model = config.initialize_ollama()
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)

        if len(messages) == 1:
            user_prompt = get_user_first_message_prompt()
        else:
            user_prompt = get_user_prompt()

        if config.verbose:
            print(
                "[call_ollama_llava] user_prompt",
                user_prompt,
            )

        vision_message = {
            "role": "user",
            "content": user_prompt,
            "images": [screenshot_filename],
        }
        messages.append(vision_message)

        response = model.chat(
            model="llava",
            messages=messages,
        )

        # Important: Remove the image path from the message history.
        # Ollama will attempt to load each image reference and will
        # eventually timeout.
        messages[-1]["images"] = None

        content = response["message"]["content"].strip()

        content = clean_json(content)

        assistant_message = {"role": "assistant", "content": content}
        if config.verbose:
            print(
                "[call_ollama_llava] content",
                content,
            )
        content = json.loads(content)

        messages.append(assistant_message)

        return content

    except ollama.ResponseError as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Operate] Couldn't connect to Ollama. With Ollama installed, run `ollama pull llava` then `ollama serve`{ANSI_RESET}",
            e,
        )

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[llava] That did not work. Trying again {ANSI_RESET}",
            e,
        )
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] AI response was {ANSI_RESET}",
            content,
        )
        if config.verbose:
            traceback.print_exc()
        return call_ollama_llava(messages)

# From models/apis.py
def get_last_assistant_message(messages):
    """
    Retrieve the last message from the assistant in the messages array.
    If the last assistant message is the first message in the array, return None.
    """
    for index in reversed(range(len(messages))):
        if messages[index]["role"] == "assistant":
            if index == 0:  # Check if the assistant message is the first in the array
                return None
            else:
                return messages[index]
    return None

# From models/apis.py
def gpt_4_fallback(messages, objective, model):
    if config.verbose:
        print("[gpt_4_fallback]")
    system_prompt = get_system_prompt("gpt-4o", objective)
    new_system_message = {"role": "system", "content": system_prompt}
    # remove and replace the first message in `messages` with `new_system_message`

    messages[0] = new_system_message

    if config.verbose:
        print("[gpt_4_fallback][updated]")
        print("[gpt_4_fallback][updated] len(messages)", len(messages))

    return call_gpt_4o(messages)

# From models/apis.py
def confirm_system_prompt(messages, objective, model):
    """
    On `Exception` we default to `call_gpt_4_vision_preview` so we have this function to reassign system prompt in case of a previous failure
    """
    if config.verbose:
        print("[confirm_system_prompt] model", model)

    system_prompt = get_system_prompt(model, objective)
    new_system_message = {"role": "system", "content": system_prompt}
    # remove and replace the first message in `messages` with `new_system_message`

    messages[0] = new_system_message

    if config.verbose:
        print("[confirm_system_prompt]")
        print("[confirm_system_prompt] len(messages)", len(messages))
        for m in messages:
            if m["role"] != "user":
                print("--------------------[message]--------------------")
                print("[confirm_system_prompt][message] role", m["role"])
                print("[confirm_system_prompt][message] content", m["content"])
                print("------------------[end message]------------------")

# From models/apis.py
def clean_json(content):
    if config.verbose:
        print("\n\n[clean_json] content before cleaning", content)
    if content.startswith("```json"):
        content = content[
            len("```json") :
        ].strip()  # Remove starting ```json and trim whitespace
    elif content.startswith("```"):
        content = content[
            len("```") :
        ].strip()  # Remove starting ``` and trim whitespace
    if content.endswith("```"):
        content = content[
            : -len("```")
        ].strip()  # Remove ending ``` and trim whitespace

    # Normalize line breaks and remove any unwanted characters
    content = "\n".join(line.strip() for line in content.splitlines())

    if config.verbose:
        print("\n\n[clean_json] content after cleaning", content)

    return content

import logging
import socket
import pyDes
from typing import Optional
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import modes

# From src/vnc_client.py
class PixelFormat:
    """VNC pixel format specification."""

    def __init__(self, raw_data: bytes):
        """Parse pixel format from raw data.

        Args:
            raw_data: Raw pixel format data (16 bytes)
        """
        self.bits_per_pixel = raw_data[0]
        self.depth = raw_data[1]
        self.big_endian = raw_data[2] != 0
        self.true_color = raw_data[3] != 0
        self.red_max = int.from_bytes(raw_data[4:6], byteorder='big')
        self.green_max = int.from_bytes(raw_data[6:8], byteorder='big')
        self.blue_max = int.from_bytes(raw_data[8:10], byteorder='big')
        self.red_shift = raw_data[10]
        self.green_shift = raw_data[11]
        self.blue_shift = raw_data[12]
        # Padding bytes 13-15 ignored

    def __str__(self) -> str:
        """Return string representation of pixel format."""
        return (f"PixelFormat(bpp={self.bits_per_pixel}, depth={self.depth}, "
                f"big_endian={self.big_endian}, true_color={self.true_color}, "
                f"rgba_max=({self.red_max},{self.green_max},{self.blue_max}), "
                f"rgba_shift=({self.red_shift},{self.green_shift},{self.blue_shift}))")

# From src/vnc_client.py
class Encoding:
    """VNC encoding types."""
    RAW = 0
    COPY_RECT = 1
    RRE = 2
    HEXTILE = 5
    ZLIB = 6
    TIGHT = 7
    ZRLE = 16
    CURSOR = -239
    DESKTOP_SIZE = -223

# From src/vnc_client.py
class VNCClient:
    """VNC client implementation to connect to remote MacOs machines and capture screenshots."""

    def __init__(self, host: str, port: int = 5900, password: Optional[str] = None, username: Optional[str] = None,
                 encryption: str = "prefer_on"):
        """Initialize VNC client with connection parameters.

        Args:
            host: remote MacOs machine hostname or IP address
            port: remote MacOs machine port (default: 5900)
            password: remote MacOs machine password (optional)
            username: remote MacOs machine username (optional, only used with certain authentication methods)
            encryption: Encryption preference, one of "prefer_on", "prefer_off", "server" (default: "prefer_on")
        """
        self.host = host
        self.port = port
        self.password = password
        self.username = username
        self.encryption = encryption
        self.socket = None
        self.width = 0
        self.height = 0
        self.pixel_format = None
        self.name = ""
        self.protocol_version = ""
        self._last_frame = None  # Store last frame for incremental updates
        self._socket_buffer_size = 8192  # Increased buffer size for better performance
        logger.debug(f"Initialized VNC client for {host}:{port} with encryption={encryption}")
        if username:
            logger.debug(f"Username authentication enabled for: {username}")

    def connect(self) -> Tuple[bool, Optional[str]]:
        """Connect to the remote MacOs machine and perform the RFB handshake.

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message) where success is True if connection
                                        was successful and error_message contains the reason for
                                        failure if success is False
        """
        try:
            logger.info(f"Attempting connection to remote MacOs machine at {self.host}:{self.port}")
            logger.debug(f"Connection parameters: encryption={self.encryption}, username={'set' if self.username else 'not set'}, password={'set' if self.password else 'not set'}")

            # Create socket and connect
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout
            logger.debug(f"Created socket with 10 second timeout")

            try:
                self.socket.connect((self.host, self.port))
                logger.info(f"Successfully established TCP connection to {self.host}:{self.port}")
            except ConnectionRefusedError:
                error_msg = f"Connection refused by {self.host}:{self.port}. Ensure remote MacOs machine is running and port is correct."
                logger.error(error_msg)
                return False, error_msg
            except socket.timeout:
                error_msg = f"Connection timed out while trying to connect to {self.host}:{self.port}"
                logger.error(error_msg)
                return False, error_msg
            except socket.gaierror as e:
                error_msg = f"DNS resolution failed for host {self.host}: {str(e)}"
                logger.error(error_msg)
                return False, error_msg

            # Receive RFB protocol version
            try:
                version = self.socket.recv(12).decode('ascii')
                self.protocol_version = version.strip()
                logger.info(f"Server protocol version: {self.protocol_version}")

                if not version.startswith("RFB "):
                    error_msg = f"Invalid protocol version string received: {version}"
                    logger.error(error_msg)
                    return False, error_msg

                # Parse version numbers for debugging
                try:
                    major, minor = version[4:].strip().split(".")
                    logger.debug(f"Server RFB version: major={major}, minor={minor}")
                except ValueError:
                    logger.warning(f"Could not parse version numbers from: {version}")
            except socket.timeout:
                error_msg = "Timeout while waiting for protocol version"
                logger.error(error_msg)
                return False, error_msg

            # Send our protocol version
            our_version = b"RFB 003.008\n"
            logger.debug(f"Sending our protocol version: {our_version.decode('ascii').strip()}")
            self.socket.sendall(our_version)

            # In RFB 3.8+, server sends number of security types followed by list of types
            try:
                security_types_count = self.socket.recv(1)[0]
                logger.info(f"Server offers {security_types_count} security types")

                if security_types_count == 0:
                    # Read error message
                    error_length = int.from_bytes(self.socket.recv(4), byteorder='big')
                    error_message = self.socket.recv(error_length).decode('ascii')
                    error_msg = f"Server rejected connection with error: {error_message}"
                    logger.error(error_msg)
                    return False, error_msg

                # Receive available security types
                security_types = self.socket.recv(security_types_count)
                logger.debug(f"Available security types: {[st for st in security_types]}")

                # Log security type descriptions
                security_type_names = {
                    0: "Invalid",
                    1: "None",
                    2: "VNC Authentication",
                    5: "RA2",
                    6: "RA2ne",
                    16: "Tight",
                    18: "TLS",
                    19: "VeNCrypt",
                    20: "GTK-VNC SASL",
                    21: "MD5 hash authentication",
                    22: "Colin Dean xvp",
                    30: "Apple Authentication"
                }

                for st in security_types:
                    name = security_type_names.get(st, f"Unknown type {st}")
                    logger.debug(f"Server supports security type {st}: {name}")
            except socket.timeout:
                error_msg = "Timeout while waiting for security types"
                logger.error(error_msg)
                return False, error_msg

            # Choose a security type we can handle based on encryption preference
            chosen_type = None

            # Check if security type 30 (Apple Authentication) is available
            if 30 in security_types and self.password:
                logger.info("Found Apple Authentication (type 30) - selecting")
                chosen_type = 30
            else:
                error_msg = "Apple Authentication (type 30) not available from server"
                logger.error(error_msg)
                logger.debug("Server security types: " + ", ".join(str(st) for st in security_types))
                logger.debug("We only support Apple Authentication (30)")
                return False, error_msg

            # Send chosen security type
            logger.info(f"Selecting security type: {chosen_type}")
            self.socket.sendall(bytes([chosen_type]))

            # Handle authentication based on chosen type
            if chosen_type == 30:
                logger.debug(f"Starting Apple authentication (type {chosen_type})")
                if not self.password:
                    error_msg = "Password required but not provided"
                    logger.error(error_msg)
                    return False, error_msg

                # Receive Diffie-Hellman parameters from server
                logger.debug("Reading Diffie-Hellman parameters from server")
                try:
                    # Read generator (2 bytes)
                    generator_data = self.socket.recv(2)
                    if len(generator_data) != 2:
                        error_msg = f"Invalid generator data received: {generator_data.hex()}"
                        logger.error(error_msg)
                        return False, error_msg
                    generator = int.from_bytes(generator_data, byteorder='big')
                    logger.debug(f"Generator: {generator}")

                    # Read key length (2 bytes)
                    key_length_data = self.socket.recv(2)
                    if len(key_length_data) != 2:
                        error_msg = f"Invalid key length data received: {key_length_data.hex()}"
                        logger.error(error_msg)
                        return False, error_msg
                    key_length = int.from_bytes(key_length_data, byteorder='big')
                    logger.debug(f"Key length: {key_length}")

                    # Read prime modulus (key_length bytes)
                    prime_data = self.socket.recv(key_length)
                    if len(prime_data) != key_length:
                        error_msg = f"Invalid prime data received, expected {key_length} bytes, got {len(prime_data)}"
                        logger.error(error_msg)
                        return False, error_msg
                    logger.debug(f"Prime modulus received ({len(prime_data)} bytes)")

                    # Read server's public key (key_length bytes)
                    server_public_key = self.socket.recv(key_length)
                    if len(server_public_key) != key_length:
                        error_msg = f"Invalid server public key received, expected {key_length} bytes, got {len(server_public_key)}"
                        logger.error(error_msg)
                        return False, error_msg
                    logger.debug(f"Server public key received ({len(server_public_key)} bytes)")

                    # Import required libraries for Diffie-Hellman key exchange
                    try:
                        from cryptography.hazmat.primitives.asymmetric import dh
                        from cryptography.hazmat.primitives import hashes
                        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                        import os

                        # Convert parameters to integers for DH
                        p_int = int.from_bytes(prime_data, byteorder='big')
                        g_int = generator

                        # Create parameter numbers
                        parameter_numbers = dh.DHParameterNumbers(p_int, g_int)
                        parameters = parameter_numbers.parameters()

                        # Generate our private key
                        private_key = parameters.generate_private_key()

                        # Get our public key in bytes
                        public_key_bytes = private_key.public_key().public_numbers().y.to_bytes(key_length, byteorder='big')

                        # Convert server's public key to integer
                        server_public_int = int.from_bytes(server_public_key, byteorder='big')
                        server_public_numbers = dh.DHPublicNumbers(server_public_int, parameter_numbers)
                        server_public_key_obj = server_public_numbers.public_key()

                        # Generate shared key
                        shared_key = private_key.exchange(server_public_key_obj)

                        # Generate MD5 hash of shared key for AES
                        md5 = hashes.Hash(hashes.MD5())
                        md5.update(shared_key)
                        aes_key = md5.finalize()

                        # Create credentials array (128 bytes)
                        creds = bytearray(128)

                        # Fill with random data
                        for i in range(128):
                            creds[i] = ord(os.urandom(1))

                        # Add username and password to credentials array
                        username_bytes = self.username.encode('utf-8') if self.username else b''
                        password_bytes = self.password.encode('utf-8')

                        # Username in first 64 bytes
                        username_len = min(len(username_bytes), 63)  # Leave room for null byte
                        creds[0:username_len] = username_bytes[0:username_len]
                        creds[username_len] = 0  # Null terminator

                        # Password in second 64 bytes
                        password_len = min(len(password_bytes), 63)  # Leave room for null byte
                        creds[64:64+password_len] = password_bytes[0:password_len]
                        creds[64+password_len] = 0  # Null terminator

                        # Encrypt credentials with AES-128-ECB
                        cipher = Cipher(algorithms.AES(aes_key), modes.ECB())
                        encryptor = cipher.encryptor()
                        encrypted_creds = encryptor.update(creds) + encryptor.finalize()

                        # Send encrypted credentials followed by our public key
                        logger.debug("Sending encrypted credentials and public key")
                        self.socket.sendall(encrypted_creds + public_key_bytes)

                    except ImportError as e:
                        error_msg = f"Missing required libraries for DH key exchange: {str(e)}"
                        logger.error(error_msg)
                        logger.debug("Install required packages with: pip install cryptography")
                        return False, error_msg
                    except Exception as e:
                        error_msg = f"Error during Diffie-Hellman key exchange: {str(e)}"
                        logger.error(error_msg)
                        return False, error_msg

                except Exception as e:
                    error_msg = f"Error reading DH parameters: {str(e)}"
                    logger.error(error_msg)
                    return False, error_msg

                # Check authentication result
                try:
                    logger.debug("Waiting for Apple authentication result")
                    auth_result = int.from_bytes(self.socket.recv(4), byteorder='big')

                    # Map known Apple VNC error codes
                    apple_auth_errors = {
                        1: "Authentication failed - invalid password",
                        2: "Authentication failed - password required",
                        3: "Authentication failed - too many attempts",
                        560513588: "Authentication failed - encryption mismatch or invalid credentials",
                        # Add more error codes as discovered
                    }

                    if auth_result != 0:
                        error_msg = apple_auth_errors.get(auth_result, f"Authentication failed with unknown error code: {auth_result}")
                        logger.error(f"Apple authentication failed: {error_msg}")
                        if auth_result == 560513588:
                            error_msg += "\nThis error often indicates:\n"
                            error_msg += "1. Password encryption/encoding mismatch\n"
                            error_msg += "2. Screen Recording permission not granted\n"
                            error_msg += "3. Remote Management/Screen Sharing not enabled"
                            logger.debug("This error often indicates:")
                            logger.debug("1. Password encryption/encoding mismatch")
                            logger.debug("2. Screen Recording permission not granted")
                            logger.debug("3. Remote Management/Screen Sharing not enabled")
                        return False, error_msg

                    logger.info("Apple authentication successful")
                except Exception as e:
                    error_msg = f"Error reading authentication result: {str(e)}"
                    logger.error(error_msg)
                    return False, error_msg
            else:
                error_msg = f"Only Apple Authentication (type 30) is supported"
                logger.error(error_msg)
                return False, error_msg

            # Send client init (shared flag)
            logger.debug("Sending client init with shared flag")
            self.socket.sendall(b'\x01')  # non-zero = shared

            # Receive server init
            logger.debug("Waiting for server init message")
            server_init_header = self.socket.recv(24)
            if len(server_init_header) < 24:
                error_msg = f"Incomplete server init header received: {server_init_header.hex()}"
                logger.error(error_msg)
                return False, error_msg

            # Parse server init
            self.width = int.from_bytes(server_init_header[0:2], byteorder='big')
            self.height = int.from_bytes(server_init_header[2:4], byteorder='big')
            self.pixel_format = PixelFormat(server_init_header[4:20])

            name_length = int.from_bytes(server_init_header[20:24], byteorder='big')
            logger.debug(f"Server reports desktop size: {self.width}x{self.height}")
            logger.debug(f"Server name length: {name_length}")

            if name_length > 0:
                name_data = self.socket.recv(name_length)
                self.name = name_data.decode('utf-8', errors='replace')
                logger.debug(f"Server name: {self.name}")

            logger.info(f"Successfully connected to remote MacOs machine: {self.name}")
            logger.debug(f"Screen dimensions: {self.width}x{self.height}")
            logger.debug(f"Initial pixel format: {self.pixel_format}")

            # Set preferred pixel format (32-bit true color)
            logger.debug("Setting preferred pixel format")
            self._set_pixel_format()

            # Set encodings (prioritize the ones we can actually handle)
            logger.debug("Setting supported encodings")
            self._set_encodings([Encoding.RAW, Encoding.COPY_RECT, Encoding.DESKTOP_SIZE])

            logger.info("VNC connection fully established and configured")
            return True, None

        except Exception as e:
            error_msg = f"Unexpected error during VNC connection: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            return False, error_msg

    def _set_pixel_format(self):
        """Set the pixel format to be used for the connection (32-bit true color)."""
        try:
            message = bytearray([0])  # message type 0 = SetPixelFormat
            message.extend([0, 0, 0])  # padding

            # Pixel format (16 bytes)
            message.extend([
                32,  # bits-per-pixel
                24,  # depth
                1,   # big-endian flag (1 = true)
                1,   # true-color flag (1 = true)
                0, 255,  # red-max (255)
                0, 255,  # green-max (255)
                0, 255,  # blue-max (255)
                16,  # red-shift
                8,   # green-shift
                0,   # blue-shift
                0, 0, 0  # padding
            ])

            self.socket.sendall(message)
            logger.debug("Set pixel format to 32-bit true color")
        except Exception as e:
            logger.error(f"Error setting pixel format: {str(e)}")

    def _set_encodings(self, encodings: List[int]):
        """Set the encodings to be used for the connection.

        Args:
            encodings: List of encoding types
        """
        try:
            message = bytearray([2])  # message type 2 = SetEncodings
            message.extend([0])  # padding

            # Number of encodings
            message.extend(len(encodings).to_bytes(2, byteorder='big'))

            # Encodings
            for encoding in encodings:
                message.extend(encoding.to_bytes(4, byteorder='big', signed=True))

            self.socket.sendall(message)
            logger.debug(f"Set encodings: {encodings}")
        except Exception as e:
            logger.error(f"Error setting encodings: {str(e)}")

    def _decode_raw_rect(self, rect_data: bytes, x: int, y: int, width: int, height: int,
                        img: Image.Image) -> None:
        """Decode a RAW-encoded rectangle and draw it to the image.

        Args:
            rect_data: Raw pixel data
            x: X position of rectangle
            y: Y position of rectangle
            width: Width of rectangle
            height: Height of rectangle
            img: PIL Image to draw to
        """
        try:
            # Create a new image from the raw data
            if self.pixel_format.bits_per_pixel == 32:
                # 32-bit color (RGBA)
                raw_img = Image.frombytes('RGBA', (width, height), rect_data)
                # Convert to RGB if needed
                if raw_img.mode != 'RGB':
                    raw_img = raw_img.convert('RGB')
            elif self.pixel_format.bits_per_pixel == 16:
                # 16-bit color needs special handling
                raw_img = Image.new('RGB', (width, height))
                pixels = raw_img.load()

                for i in range(height):
                    for j in range(width):
                        idx = (i * width + j) * 2
                        pixel = int.from_bytes(rect_data[idx:idx+2],
                                            byteorder='big' if self.pixel_format.big_endian else 'little')

                        r = ((pixel >> self.pixel_format.red_shift) & self.pixel_format.red_max)
                        g = ((pixel >> self.pixel_format.green_shift) & self.pixel_format.green_max)
                        b = ((pixel >> self.pixel_format.blue_shift) & self.pixel_format.blue_max)

                        # Scale values to 0-255 range
                        r = int(r * 255 / self.pixel_format.red_max)
                        g = int(g * 255 / self.pixel_format.green_max)
                        b = int(b * 255 / self.pixel_format.blue_max)

                        pixels[j, i] = (r, g, b)
            else:
                # Fallback for other bit depths (basic conversion)
                raw_img = Image.new('RGB', (width, height), color='black')
                logger.warning(f"Unsupported pixel format: {self.pixel_format.bits_per_pixel}-bit")

            # Paste the decoded image onto the target image
            img.paste(raw_img, (x, y))

        except Exception as e:
            logger.error(f"Error decoding RAW rectangle: {str(e)}")
            # Fill with error color on failure
            raw_img = Image.new('RGB', (width, height), color='red')
            img.paste(raw_img, (x, y))

    def _decode_copy_rect(self, rect_data: bytes, x: int, y: int, width: int, height: int,
                         img: Image.Image) -> None:
        """Decode a COPY_RECT-encoded rectangle and draw it to the image.

        Args:
            rect_data: CopyRect data (src_x, src_y)
            x: X position of destination rectangle
            y: Y position of destination rectangle
            width: Width of rectangle
            height: Height of rectangle
            img: PIL Image to draw to
        """
        try:
            src_x = int.from_bytes(rect_data[0:2], byteorder='big')
            src_y = int.from_bytes(rect_data[2:4], byteorder='big')

            # Copy the region from the image itself
            region = img.crop((src_x, src_y, src_x + width, src_y + height))
            img.paste(region, (x, y))

        except Exception as e:
            logger.error(f"Error decoding COPY_RECT rectangle: {str(e)}")
            # Fill with error color on failure
            raw_img = Image.new('RGB', (width, height), color='blue')
            img.paste(raw_img, (x, y))

    def capture_screen(self) -> Optional[bytes]:
        """Capture a screenshot from the remote MacOs machine with optimizations."""
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return None

            # Use incremental updates if we have a previous frame
            is_incremental = self._last_frame is not None

            # Create or reuse image
            if is_incremental:
                img = self._last_frame
            else:
                img = Image.new('RGB', (self.width, self.height), color='black')

            # Send FramebufferUpdateRequest message
            msg = bytearray([3])  # message type 3 = FramebufferUpdateRequest
            msg.extend([1 if is_incremental else 0])  # Use incremental updates when possible
            msg.extend(int(0).to_bytes(2, byteorder='big'))  # x-position
            msg.extend(int(0).to_bytes(2, byteorder='big'))  # y-position
            msg.extend(int(self.width).to_bytes(2, byteorder='big'))  # width
            msg.extend(int(self.height).to_bytes(2, byteorder='big'))  # height

            self.socket.sendall(msg)

            # Receive FramebufferUpdate message header with larger buffer
            header = self._recv_exact(4)
            if not header or header[0] != 0:  # 0 = FramebufferUpdate
                logger.error(f"Unexpected message type in response: {header[0] if header else 'None'}")
                return None

            # Read number of rectangles
            num_rects = int.from_bytes(header[2:4], byteorder='big')
            logger.debug(f"Received {num_rects} rectangles")

            # Process each rectangle
            for rect_idx in range(num_rects):
                # Read rectangle header efficiently
                rect_header = self._recv_exact(12)
                if not rect_header:
                    logger.error("Failed to read rectangle header")
                    return None

                x = int.from_bytes(rect_header[0:2], byteorder='big')
                y = int.from_bytes(rect_header[2:4], byteorder='big')
                width = int.from_bytes(rect_header[4:6], byteorder='big')
                height = int.from_bytes(rect_header[6:8], byteorder='big')
                encoding_type = int.from_bytes(rect_header[8:12], byteorder='big', signed=True)

                if encoding_type == Encoding.RAW:
                    # Optimize RAW encoding processing
                    pixel_size = self.pixel_format.bits_per_pixel // 8
                    data_size = width * height * pixel_size

                    # Read pixel data in chunks
                    rect_data = self._recv_exact(data_size)
                    if not rect_data or len(rect_data) != data_size:
                        logger.error(f"Failed to read RAW rectangle data")
                        return None

                    # Decode and draw
                    self._decode_raw_rect(rect_data, x, y, width, height, img)

                elif encoding_type == Encoding.COPY_RECT:
                    # Optimize COPY_RECT processing
                    rect_data = self._recv_exact(4)
                    if not rect_data:
                        logger.error("Failed to read COPY_RECT data")
                        return None
                    self._decode_copy_rect(rect_data, x, y, width, height, img)

                elif encoding_type == Encoding.DESKTOP_SIZE:
                    # Handle desktop size changes
                    logger.debug(f"Desktop size changed to {width}x{height}")
                    self.width = width
                    self.height = height
                    new_img = Image.new('RGB', (self.width, self.height), color='black')
                    new_img.paste(img, (0, 0))
                    img = new_img
                else:
                    logger.warning(f"Unsupported encoding type: {encoding_type}")
                    continue

            # Store the frame for future incremental updates
            self._last_frame = img

            # Convert image to PNG with optimization
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True, quality=95)
            img_byte_arr.seek(0)

            return img_byte_arr.getvalue()

        except Exception as e:
            logger.error(f"Error capturing screen: {str(e)}")
            return None

    def _recv_exact(self, size: int) -> Optional[bytes]:
        """Receive exactly size bytes from the socket efficiently."""
        try:
            data = bytearray()
            while len(data) < size:
                chunk = self.socket.recv(min(self._socket_buffer_size, size - len(data)))
                if not chunk:
                    return None
                data.extend(chunk)
            return bytes(data)
        except Exception as e:
            logger.error(f"Error receiving data: {str(e)}")
            return None

    def close(self):
        """Close the connection to the remote MacOs machine."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

    def send_key_event(self, key: int, down: bool) -> bool:
        """Send a key event to the remote MacOs machine.

        Args:
            key: X11 keysym value representing the key
            down: True for key press, False for key release

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Message type 4 = KeyEvent
            message = bytearray([4])

            # Down flag (1 = pressed, 0 = released)
            message.extend([1 if down else 0])

            # Padding (2 bytes)
            message.extend([0, 0])

            # Key (4 bytes, big endian)
            message.extend(key.to_bytes(4, byteorder='big'))

            logger.debug(f"Sending KeyEvent: key=0x{key:08x}, down={down}")
            self.socket.sendall(message)
            return True

        except Exception as e:
            logger.error(f"Error sending key event: {str(e)}")
            return False

    def send_pointer_event(self, x: int, y: int, button_mask: int) -> bool:
        """Send a pointer (mouse) event to the remote MacOs machine.

        Args:
            x: X position (0 to framebuffer_width-1)
            y: Y position (0 to framebuffer_height-1)
            button_mask: Bit mask of pressed buttons:
                bit 0 = left button (1)
                bit 1 = middle button (2)
                bit 2 = right button (4)
                bit 3 = wheel up (8)
                bit 4 = wheel down (16)
                bit 5 = wheel left (32)
                bit 6 = wheel right (64)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Ensure coordinates are within framebuffer bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))

            # Message type 5 = PointerEvent
            message = bytearray([5])

            # Button mask (1 byte)
            message.extend([button_mask & 0xFF])

            # X position (2 bytes, big endian)
            message.extend(x.to_bytes(2, byteorder='big'))

            # Y position (2 bytes, big endian)
            message.extend(y.to_bytes(2, byteorder='big'))

            logger.debug(f"Sending PointerEvent: x={x}, y={y}, button_mask={button_mask:08b}")
            self.socket.sendall(message)
            return True

        except Exception as e:
            logger.error(f"Error sending pointer event: {str(e)}")
            return False

    def send_mouse_click(self, x: int, y: int, button: int = 1, double_click: bool = False, delay_ms: int = 100) -> bool:
        """Send a mouse click at the specified position.

        Args:
            x: X position
            y: Y position
            button: Mouse button (1=left, 2=middle, 3=right)
            double_click: Whether to perform a double-click
            delay_ms: Delay between press and release in milliseconds

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Calculate button mask
            button_mask = 1 << (button - 1)

            # Move mouse to position first (no buttons pressed)
            if not self.send_pointer_event(x, y, 0):
                return False

            # Single click or first click of double-click
            if not self.send_pointer_event(x, y, button_mask):
                return False

            # Wait for press-release delay
            time.sleep(delay_ms / 1000.0)

            # Release button
            if not self.send_pointer_event(x, y, 0):
                return False

            # If double click, perform second click
            if double_click:
                # Wait between clicks
                time.sleep(delay_ms / 1000.0)

                # Second press
                if not self.send_pointer_event(x, y, button_mask):
                    return False

                # Wait for press-release delay
                time.sleep(delay_ms / 1000.0)

                # Second release
                if not self.send_pointer_event(x, y, 0):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error sending mouse click: {str(e)}")
            return False

    def send_text(self, text: str) -> bool:
        """Send text as a series of key press/release events.

        Args:
            text: The text to send

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Standard ASCII to X11 keysym mapping for printable ASCII characters
            # For most characters, the keysym is just the ASCII value
            success = True

            for char in text:
                # Special key mapping for common non-printable characters
                if char == '\n' or char == '\r':  # Return/Enter
                    key = 0xff0d
                elif char == '\t':  # Tab
                    key = 0xff09
                elif char == '\b':  # Backspace
                    key = 0xff08
                elif char == ' ':  # Space
                    key = 0x20
                else:
                    # For printable ASCII and Unicode characters
                    key = ord(char)

                # If it's an uppercase letter, we need to simulate a shift press
                need_shift = char.isupper() or char in '~!@#$%^&*()_+{}|:"<>?'

                if need_shift:
                    # Press shift (left shift keysym = 0xffe1)
                    if not self.send_key_event(0xffe1, True):
                        success = False
                        break

                # Press key
                if not self.send_key_event(key, True):
                    success = False
                    break

                # Release key
                if not self.send_key_event(key, False):
                    success = False
                    break

                if need_shift:
                    # Release shift
                    if not self.send_key_event(0xffe1, False):
                        success = False
                        break

                # Small delay between keys to avoid overwhelming the server
                time.sleep(0.01)

            return success

        except Exception as e:
            logger.error(f"Error sending text: {str(e)}")
            return False

    def send_key_combination(self, keys: List[int]) -> bool:
        """Send a key combination (e.g., Ctrl+Alt+Delete).

        Args:
            keys: List of X11 keysym values to press in sequence

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Press all keys in sequence
            for key in keys:
                if not self.send_key_event(key, True):
                    return False

            # Release all keys in reverse order
            for key in reversed(keys):
                if not self.send_key_event(key, False):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error sending key combination: {str(e)}")
            return False

# From src/vnc_client.py
def encrypt_MACOS_PASSWORD(password: str, challenge: bytes) -> bytes:
    """Encrypt VNC password for authentication.

    Args:
        password: VNC password
        challenge: Challenge bytes from server

    Returns:
        bytes: Encrypted response
    """
    # Convert password to key (truncate to 8 chars or pad with zeros)
    key = password.ljust(8, '\x00')[:8].encode('ascii')

    # VNC uses a reversed bit order for each byte in the key
    reversed_key = bytes([((k >> 0) & 1) << 7 |
                         ((k >> 1) & 1) << 6 |
                         ((k >> 2) & 1) << 5 |
                         ((k >> 3) & 1) << 4 |
                         ((k >> 4) & 1) << 3 |
                         ((k >> 5) & 1) << 2 |
                         ((k >> 6) & 1) << 1 |
                         ((k >> 7) & 1) << 0 for k in key])

    # Create a pyDes instance for encryption
    k = pyDes.des(reversed_key, pyDes.ECB, pad=None)

    # Encrypt the challenge with the key
    result = bytearray()
    for i in range(0, len(challenge), 8):
        block = challenge[i:i+8]
        cipher_block = k.encrypt(block)
        result.extend(cipher_block)

    return bytes(result)

# From src/vnc_client.py
def connect(self) -> Tuple[bool, Optional[str]]:
        """Connect to the remote MacOs machine and perform the RFB handshake.

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message) where success is True if connection
                                        was successful and error_message contains the reason for
                                        failure if success is False
        """
        try:
            logger.info(f"Attempting connection to remote MacOs machine at {self.host}:{self.port}")
            logger.debug(f"Connection parameters: encryption={self.encryption}, username={'set' if self.username else 'not set'}, password={'set' if self.password else 'not set'}")

            # Create socket and connect
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout
            logger.debug(f"Created socket with 10 second timeout")

            try:
                self.socket.connect((self.host, self.port))
                logger.info(f"Successfully established TCP connection to {self.host}:{self.port}")
            except ConnectionRefusedError:
                error_msg = f"Connection refused by {self.host}:{self.port}. Ensure remote MacOs machine is running and port is correct."
                logger.error(error_msg)
                return False, error_msg
            except socket.timeout:
                error_msg = f"Connection timed out while trying to connect to {self.host}:{self.port}"
                logger.error(error_msg)
                return False, error_msg
            except socket.gaierror as e:
                error_msg = f"DNS resolution failed for host {self.host}: {str(e)}"
                logger.error(error_msg)
                return False, error_msg

            # Receive RFB protocol version
            try:
                version = self.socket.recv(12).decode('ascii')
                self.protocol_version = version.strip()
                logger.info(f"Server protocol version: {self.protocol_version}")

                if not version.startswith("RFB "):
                    error_msg = f"Invalid protocol version string received: {version}"
                    logger.error(error_msg)
                    return False, error_msg

                # Parse version numbers for debugging
                try:
                    major, minor = version[4:].strip().split(".")
                    logger.debug(f"Server RFB version: major={major}, minor={minor}")
                except ValueError:
                    logger.warning(f"Could not parse version numbers from: {version}")
            except socket.timeout:
                error_msg = "Timeout while waiting for protocol version"
                logger.error(error_msg)
                return False, error_msg

            # Send our protocol version
            our_version = b"RFB 003.008\n"
            logger.debug(f"Sending our protocol version: {our_version.decode('ascii').strip()}")
            self.socket.sendall(our_version)

            # In RFB 3.8+, server sends number of security types followed by list of types
            try:
                security_types_count = self.socket.recv(1)[0]
                logger.info(f"Server offers {security_types_count} security types")

                if security_types_count == 0:
                    # Read error message
                    error_length = int.from_bytes(self.socket.recv(4), byteorder='big')
                    error_message = self.socket.recv(error_length).decode('ascii')
                    error_msg = f"Server rejected connection with error: {error_message}"
                    logger.error(error_msg)
                    return False, error_msg

                # Receive available security types
                security_types = self.socket.recv(security_types_count)
                logger.debug(f"Available security types: {[st for st in security_types]}")

                # Log security type descriptions
                security_type_names = {
                    0: "Invalid",
                    1: "None",
                    2: "VNC Authentication",
                    5: "RA2",
                    6: "RA2ne",
                    16: "Tight",
                    18: "TLS",
                    19: "VeNCrypt",
                    20: "GTK-VNC SASL",
                    21: "MD5 hash authentication",
                    22: "Colin Dean xvp",
                    30: "Apple Authentication"
                }

                for st in security_types:
                    name = security_type_names.get(st, f"Unknown type {st}")
                    logger.debug(f"Server supports security type {st}: {name}")
            except socket.timeout:
                error_msg = "Timeout while waiting for security types"
                logger.error(error_msg)
                return False, error_msg

            # Choose a security type we can handle based on encryption preference
            chosen_type = None

            # Check if security type 30 (Apple Authentication) is available
            if 30 in security_types and self.password:
                logger.info("Found Apple Authentication (type 30) - selecting")
                chosen_type = 30
            else:
                error_msg = "Apple Authentication (type 30) not available from server"
                logger.error(error_msg)
                logger.debug("Server security types: " + ", ".join(str(st) for st in security_types))
                logger.debug("We only support Apple Authentication (30)")
                return False, error_msg

            # Send chosen security type
            logger.info(f"Selecting security type: {chosen_type}")
            self.socket.sendall(bytes([chosen_type]))

            # Handle authentication based on chosen type
            if chosen_type == 30:
                logger.debug(f"Starting Apple authentication (type {chosen_type})")
                if not self.password:
                    error_msg = "Password required but not provided"
                    logger.error(error_msg)
                    return False, error_msg

                # Receive Diffie-Hellman parameters from server
                logger.debug("Reading Diffie-Hellman parameters from server")
                try:
                    # Read generator (2 bytes)
                    generator_data = self.socket.recv(2)
                    if len(generator_data) != 2:
                        error_msg = f"Invalid generator data received: {generator_data.hex()}"
                        logger.error(error_msg)
                        return False, error_msg
                    generator = int.from_bytes(generator_data, byteorder='big')
                    logger.debug(f"Generator: {generator}")

                    # Read key length (2 bytes)
                    key_length_data = self.socket.recv(2)
                    if len(key_length_data) != 2:
                        error_msg = f"Invalid key length data received: {key_length_data.hex()}"
                        logger.error(error_msg)
                        return False, error_msg
                    key_length = int.from_bytes(key_length_data, byteorder='big')
                    logger.debug(f"Key length: {key_length}")

                    # Read prime modulus (key_length bytes)
                    prime_data = self.socket.recv(key_length)
                    if len(prime_data) != key_length:
                        error_msg = f"Invalid prime data received, expected {key_length} bytes, got {len(prime_data)}"
                        logger.error(error_msg)
                        return False, error_msg
                    logger.debug(f"Prime modulus received ({len(prime_data)} bytes)")

                    # Read server's public key (key_length bytes)
                    server_public_key = self.socket.recv(key_length)
                    if len(server_public_key) != key_length:
                        error_msg = f"Invalid server public key received, expected {key_length} bytes, got {len(server_public_key)}"
                        logger.error(error_msg)
                        return False, error_msg
                    logger.debug(f"Server public key received ({len(server_public_key)} bytes)")

                    # Import required libraries for Diffie-Hellman key exchange
                    try:
                        from cryptography.hazmat.primitives.asymmetric import dh
                        from cryptography.hazmat.primitives import hashes
                        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
                        import os

                        # Convert parameters to integers for DH
                        p_int = int.from_bytes(prime_data, byteorder='big')
                        g_int = generator

                        # Create parameter numbers
                        parameter_numbers = dh.DHParameterNumbers(p_int, g_int)
                        parameters = parameter_numbers.parameters()

                        # Generate our private key
                        private_key = parameters.generate_private_key()

                        # Get our public key in bytes
                        public_key_bytes = private_key.public_key().public_numbers().y.to_bytes(key_length, byteorder='big')

                        # Convert server's public key to integer
                        server_public_int = int.from_bytes(server_public_key, byteorder='big')
                        server_public_numbers = dh.DHPublicNumbers(server_public_int, parameter_numbers)
                        server_public_key_obj = server_public_numbers.public_key()

                        # Generate shared key
                        shared_key = private_key.exchange(server_public_key_obj)

                        # Generate MD5 hash of shared key for AES
                        md5 = hashes.Hash(hashes.MD5())
                        md5.update(shared_key)
                        aes_key = md5.finalize()

                        # Create credentials array (128 bytes)
                        creds = bytearray(128)

                        # Fill with random data
                        for i in range(128):
                            creds[i] = ord(os.urandom(1))

                        # Add username and password to credentials array
                        username_bytes = self.username.encode('utf-8') if self.username else b''
                        password_bytes = self.password.encode('utf-8')

                        # Username in first 64 bytes
                        username_len = min(len(username_bytes), 63)  # Leave room for null byte
                        creds[0:username_len] = username_bytes[0:username_len]
                        creds[username_len] = 0  # Null terminator

                        # Password in second 64 bytes
                        password_len = min(len(password_bytes), 63)  # Leave room for null byte
                        creds[64:64+password_len] = password_bytes[0:password_len]
                        creds[64+password_len] = 0  # Null terminator

                        # Encrypt credentials with AES-128-ECB
                        cipher = Cipher(algorithms.AES(aes_key), modes.ECB())
                        encryptor = cipher.encryptor()
                        encrypted_creds = encryptor.update(creds) + encryptor.finalize()

                        # Send encrypted credentials followed by our public key
                        logger.debug("Sending encrypted credentials and public key")
                        self.socket.sendall(encrypted_creds + public_key_bytes)

                    except ImportError as e:
                        error_msg = f"Missing required libraries for DH key exchange: {str(e)}"
                        logger.error(error_msg)
                        logger.debug("Install required packages with: pip install cryptography")
                        return False, error_msg
                    except Exception as e:
                        error_msg = f"Error during Diffie-Hellman key exchange: {str(e)}"
                        logger.error(error_msg)
                        return False, error_msg

                except Exception as e:
                    error_msg = f"Error reading DH parameters: {str(e)}"
                    logger.error(error_msg)
                    return False, error_msg

                # Check authentication result
                try:
                    logger.debug("Waiting for Apple authentication result")
                    auth_result = int.from_bytes(self.socket.recv(4), byteorder='big')

                    # Map known Apple VNC error codes
                    apple_auth_errors = {
                        1: "Authentication failed - invalid password",
                        2: "Authentication failed - password required",
                        3: "Authentication failed - too many attempts",
                        560513588: "Authentication failed - encryption mismatch or invalid credentials",
                        # Add more error codes as discovered
                    }

                    if auth_result != 0:
                        error_msg = apple_auth_errors.get(auth_result, f"Authentication failed with unknown error code: {auth_result}")
                        logger.error(f"Apple authentication failed: {error_msg}")
                        if auth_result == 560513588:
                            error_msg += "\nThis error often indicates:\n"
                            error_msg += "1. Password encryption/encoding mismatch\n"
                            error_msg += "2. Screen Recording permission not granted\n"
                            error_msg += "3. Remote Management/Screen Sharing not enabled"
                            logger.debug("This error often indicates:")
                            logger.debug("1. Password encryption/encoding mismatch")
                            logger.debug("2. Screen Recording permission not granted")
                            logger.debug("3. Remote Management/Screen Sharing not enabled")
                        return False, error_msg

                    logger.info("Apple authentication successful")
                except Exception as e:
                    error_msg = f"Error reading authentication result: {str(e)}"
                    logger.error(error_msg)
                    return False, error_msg
            else:
                error_msg = f"Only Apple Authentication (type 30) is supported"
                logger.error(error_msg)
                return False, error_msg

            # Send client init (shared flag)
            logger.debug("Sending client init with shared flag")
            self.socket.sendall(b'\x01')  # non-zero = shared

            # Receive server init
            logger.debug("Waiting for server init message")
            server_init_header = self.socket.recv(24)
            if len(server_init_header) < 24:
                error_msg = f"Incomplete server init header received: {server_init_header.hex()}"
                logger.error(error_msg)
                return False, error_msg

            # Parse server init
            self.width = int.from_bytes(server_init_header[0:2], byteorder='big')
            self.height = int.from_bytes(server_init_header[2:4], byteorder='big')
            self.pixel_format = PixelFormat(server_init_header[4:20])

            name_length = int.from_bytes(server_init_header[20:24], byteorder='big')
            logger.debug(f"Server reports desktop size: {self.width}x{self.height}")
            logger.debug(f"Server name length: {name_length}")

            if name_length > 0:
                name_data = self.socket.recv(name_length)
                self.name = name_data.decode('utf-8', errors='replace')
                logger.debug(f"Server name: {self.name}")

            logger.info(f"Successfully connected to remote MacOs machine: {self.name}")
            logger.debug(f"Screen dimensions: {self.width}x{self.height}")
            logger.debug(f"Initial pixel format: {self.pixel_format}")

            # Set preferred pixel format (32-bit true color)
            logger.debug("Setting preferred pixel format")
            self._set_pixel_format()

            # Set encodings (prioritize the ones we can actually handle)
            logger.debug("Setting supported encodings")
            self._set_encodings([Encoding.RAW, Encoding.COPY_RECT, Encoding.DESKTOP_SIZE])

            logger.info("VNC connection fully established and configured")
            return True, None

        except Exception as e:
            error_msg = f"Unexpected error during VNC connection: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            return False, error_msg

# From src/vnc_client.py
def capture_screen(self) -> Optional[bytes]:
        """Capture a screenshot from the remote MacOs machine with optimizations."""
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return None

            # Use incremental updates if we have a previous frame
            is_incremental = self._last_frame is not None

            # Create or reuse image
            if is_incremental:
                img = self._last_frame
            else:
                img = Image.new('RGB', (self.width, self.height), color='black')

            # Send FramebufferUpdateRequest message
            msg = bytearray([3])  # message type 3 = FramebufferUpdateRequest
            msg.extend([1 if is_incremental else 0])  # Use incremental updates when possible
            msg.extend(int(0).to_bytes(2, byteorder='big'))  # x-position
            msg.extend(int(0).to_bytes(2, byteorder='big'))  # y-position
            msg.extend(int(self.width).to_bytes(2, byteorder='big'))  # width
            msg.extend(int(self.height).to_bytes(2, byteorder='big'))  # height

            self.socket.sendall(msg)

            # Receive FramebufferUpdate message header with larger buffer
            header = self._recv_exact(4)
            if not header or header[0] != 0:  # 0 = FramebufferUpdate
                logger.error(f"Unexpected message type in response: {header[0] if header else 'None'}")
                return None

            # Read number of rectangles
            num_rects = int.from_bytes(header[2:4], byteorder='big')
            logger.debug(f"Received {num_rects} rectangles")

            # Process each rectangle
            for rect_idx in range(num_rects):
                # Read rectangle header efficiently
                rect_header = self._recv_exact(12)
                if not rect_header:
                    logger.error("Failed to read rectangle header")
                    return None

                x = int.from_bytes(rect_header[0:2], byteorder='big')
                y = int.from_bytes(rect_header[2:4], byteorder='big')
                width = int.from_bytes(rect_header[4:6], byteorder='big')
                height = int.from_bytes(rect_header[6:8], byteorder='big')
                encoding_type = int.from_bytes(rect_header[8:12], byteorder='big', signed=True)

                if encoding_type == Encoding.RAW:
                    # Optimize RAW encoding processing
                    pixel_size = self.pixel_format.bits_per_pixel // 8
                    data_size = width * height * pixel_size

                    # Read pixel data in chunks
                    rect_data = self._recv_exact(data_size)
                    if not rect_data or len(rect_data) != data_size:
                        logger.error(f"Failed to read RAW rectangle data")
                        return None

                    # Decode and draw
                    self._decode_raw_rect(rect_data, x, y, width, height, img)

                elif encoding_type == Encoding.COPY_RECT:
                    # Optimize COPY_RECT processing
                    rect_data = self._recv_exact(4)
                    if not rect_data:
                        logger.error("Failed to read COPY_RECT data")
                        return None
                    self._decode_copy_rect(rect_data, x, y, width, height, img)

                elif encoding_type == Encoding.DESKTOP_SIZE:
                    # Handle desktop size changes
                    logger.debug(f"Desktop size changed to {width}x{height}")
                    self.width = width
                    self.height = height
                    new_img = Image.new('RGB', (self.width, self.height), color='black')
                    new_img.paste(img, (0, 0))
                    img = new_img
                else:
                    logger.warning(f"Unsupported encoding type: {encoding_type}")
                    continue

            # Store the frame for future incremental updates
            self._last_frame = img

            # Convert image to PNG with optimization
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG', optimize=True, quality=95)
            img_byte_arr.seek(0)

            return img_byte_arr.getvalue()

        except Exception as e:
            logger.error(f"Error capturing screen: {str(e)}")
            return None

# From src/vnc_client.py
def close(self):
        """Close the connection to the remote MacOs machine."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

# From src/vnc_client.py
def send_key_event(self, key: int, down: bool) -> bool:
        """Send a key event to the remote MacOs machine.

        Args:
            key: X11 keysym value representing the key
            down: True for key press, False for key release

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Message type 4 = KeyEvent
            message = bytearray([4])

            # Down flag (1 = pressed, 0 = released)
            message.extend([1 if down else 0])

            # Padding (2 bytes)
            message.extend([0, 0])

            # Key (4 bytes, big endian)
            message.extend(key.to_bytes(4, byteorder='big'))

            logger.debug(f"Sending KeyEvent: key=0x{key:08x}, down={down}")
            self.socket.sendall(message)
            return True

        except Exception as e:
            logger.error(f"Error sending key event: {str(e)}")
            return False

# From src/vnc_client.py
def send_pointer_event(self, x: int, y: int, button_mask: int) -> bool:
        """Send a pointer (mouse) event to the remote MacOs machine.

        Args:
            x: X position (0 to framebuffer_width-1)
            y: Y position (0 to framebuffer_height-1)
            button_mask: Bit mask of pressed buttons:
                bit 0 = left button (1)
                bit 1 = middle button (2)
                bit 2 = right button (4)
                bit 3 = wheel up (8)
                bit 4 = wheel down (16)
                bit 5 = wheel left (32)
                bit 6 = wheel right (64)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Ensure coordinates are within framebuffer bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))

            # Message type 5 = PointerEvent
            message = bytearray([5])

            # Button mask (1 byte)
            message.extend([button_mask & 0xFF])

            # X position (2 bytes, big endian)
            message.extend(x.to_bytes(2, byteorder='big'))

            # Y position (2 bytes, big endian)
            message.extend(y.to_bytes(2, byteorder='big'))

            logger.debug(f"Sending PointerEvent: x={x}, y={y}, button_mask={button_mask:08b}")
            self.socket.sendall(message)
            return True

        except Exception as e:
            logger.error(f"Error sending pointer event: {str(e)}")
            return False

# From src/vnc_client.py
def send_mouse_click(self, x: int, y: int, button: int = 1, double_click: bool = False, delay_ms: int = 100) -> bool:
        """Send a mouse click at the specified position.

        Args:
            x: X position
            y: Y position
            button: Mouse button (1=left, 2=middle, 3=right)
            double_click: Whether to perform a double-click
            delay_ms: Delay between press and release in milliseconds

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Calculate button mask
            button_mask = 1 << (button - 1)

            # Move mouse to position first (no buttons pressed)
            if not self.send_pointer_event(x, y, 0):
                return False

            # Single click or first click of double-click
            if not self.send_pointer_event(x, y, button_mask):
                return False

            # Wait for press-release delay
            time.sleep(delay_ms / 1000.0)

            # Release button
            if not self.send_pointer_event(x, y, 0):
                return False

            # If double click, perform second click
            if double_click:
                # Wait between clicks
                time.sleep(delay_ms / 1000.0)

                # Second press
                if not self.send_pointer_event(x, y, button_mask):
                    return False

                # Wait for press-release delay
                time.sleep(delay_ms / 1000.0)

                # Second release
                if not self.send_pointer_event(x, y, 0):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error sending mouse click: {str(e)}")
            return False

# From src/vnc_client.py
def send_text(self, text: str) -> bool:
        """Send text as a series of key press/release events.

        Args:
            text: The text to send

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Standard ASCII to X11 keysym mapping for printable ASCII characters
            # For most characters, the keysym is just the ASCII value
            success = True

            for char in text:
                # Special key mapping for common non-printable characters
                if char == '\n' or char == '\r':  # Return/Enter
                    key = 0xff0d
                elif char == '\t':  # Tab
                    key = 0xff09
                elif char == '\b':  # Backspace
                    key = 0xff08
                elif char == ' ':  # Space
                    key = 0x20
                else:
                    # For printable ASCII and Unicode characters
                    key = ord(char)

                # If it's an uppercase letter, we need to simulate a shift press
                need_shift = char.isupper() or char in '~!@#$%^&*()_+{}|:"<>?'

                if need_shift:
                    # Press shift (left shift keysym = 0xffe1)
                    if not self.send_key_event(0xffe1, True):
                        success = False
                        break

                # Press key
                if not self.send_key_event(key, True):
                    success = False
                    break

                # Release key
                if not self.send_key_event(key, False):
                    success = False
                    break

                if need_shift:
                    # Release shift
                    if not self.send_key_event(0xffe1, False):
                        success = False
                        break

                # Small delay between keys to avoid overwhelming the server
                time.sleep(0.01)

            return success

        except Exception as e:
            logger.error(f"Error sending text: {str(e)}")
            return False

# From src/vnc_client.py
def send_key_combination(self, keys: List[int]) -> bool:
        """Send a key combination (e.g., Ctrl+Alt+Delete).

        Args:
            keys: List of X11 keysym values to press in sequence

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.socket:
                logger.error("Not connected to remote MacOs machine")
                return False

            # Press all keys in sequence
            for key in keys:
                if not self.send_key_event(key, True):
                    return False

            # Release all keys in reverse order
            for key in reversed(keys):
                if not self.send_key_event(key, False):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error sending key combination: {str(e)}")
            return False

import mcp.types
from vnc_client import VNCClient
from vnc_client import capture_vnc_screen

# From src/action_handlers.py
def handle_remote_macos_mouse_scroll(arguments: dict[str, Any]) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Perform a mouse scroll action on a remote MacOs machine."""
    # Use environment variables
    host = MACOS_HOST
    port = MACOS_PORT
    password = MACOS_PASSWORD
    username = MACOS_USERNAME
    encryption = VNC_ENCRYPTION

    # Get required parameters from arguments
    x = arguments.get("x")
    y = arguments.get("y")
    source_width = int(arguments.get("source_width", 1366))
    source_height = int(arguments.get("source_height", 768))
    direction = arguments.get("direction", "down")

    if x is None or y is None:
        raise ValueError("x and y coordinates are required")

    # Ensure source dimensions are positive
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive values")

    # Initialize VNC client
    vnc = VNCClient(host=host, port=port, password=password, username=username, encryption=encryption)

    # Connect to remote MacOs machine
    success, error_message = vnc.connect()
    if not success:
        error_msg = f"Failed to connect to remote MacOs machine at {host}:{port}. {error_message}"
        return [types.TextContent(type="text", text=error_msg)]

    try:
        # Get target screen dimensions
        target_width = vnc.width
        target_height = vnc.height

        # Scale coordinates
        scaled_x = int((x / source_width) * target_width)
        scaled_y = int((y / source_height) * target_height)

        # Ensure coordinates are within the screen bounds
        scaled_x = max(0, min(scaled_x, target_width - 1))
        scaled_y = max(0, min(scaled_y, target_height - 1))

        # First move the mouse to the target location without clicking
        move_result = vnc.send_pointer_event(scaled_x, scaled_y, 0)

        # Map of special keys for page up/down
        special_keys = {
            "up": 0xff55,    # Page Up key
            "down": 0xff56,  # Page Down key
        }

        # Send the appropriate page key based on direction
        key = special_keys["up" if direction.lower() == "up" else "down"]
        key_result = vnc.send_key_event(key, True) and vnc.send_key_event(key, False)

        # Prepare the response with useful details
        scale_factors = {
            "x": target_width / source_width,
            "y": target_height / source_height
        }

        return [types.TextContent(
            type="text",
            text=f"""Mouse move to ({scaled_x}, {scaled_y}) {'succeeded' if move_result else 'failed'}
Page {direction} key press {'succeeded' if key_result else 'failed'}
Source dimensions: {source_width}x{source_height}
Target dimensions: {target_width}x{target_height}
Scale factors: {scale_factors['x']:.4f}x, {scale_factors['y']:.4f}y"""
        )]
    finally:
        # Close VNC connection
        vnc.close()

# From src/action_handlers.py
def handle_remote_macos_mouse_click(arguments: dict[str, Any]) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Perform a mouse click action on a remote MacOs machine."""
    # Use environment variables
    host = MACOS_HOST
    port = MACOS_PORT
    password = MACOS_PASSWORD
    username = MACOS_USERNAME
    encryption = VNC_ENCRYPTION

    # Get required parameters from arguments
    x = arguments.get("x")
    y = arguments.get("y")
    source_width = int(arguments.get("source_width", 1366))
    source_height = int(arguments.get("source_height", 768))
    button = int(arguments.get("button", 1))

    if x is None or y is None:
        raise ValueError("x and y coordinates are required")

    # Ensure source dimensions are positive
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive values")

    # Initialize VNC client
    vnc = VNCClient(host=host, port=port, password=password, username=username, encryption=encryption)

    # Connect to remote MacOs machine
    success, error_message = vnc.connect()
    if not success:
        error_msg = f"Failed to connect to remote MacOs machine at {host}:{port}. {error_message}"
        return [types.TextContent(type="text", text=error_msg)]

    try:
        # Get target screen dimensions
        target_width = vnc.width
        target_height = vnc.height

        # Scale coordinates
        scaled_x = int((x / source_width) * target_width)
        scaled_y = int((y / source_height) * target_height)

        # Ensure coordinates are within the screen bounds
        scaled_x = max(0, min(scaled_x, target_width - 1))
        scaled_y = max(0, min(scaled_y, target_height - 1))

        # Single click
        result = vnc.send_mouse_click(scaled_x, scaled_y, button, False)

        # Prepare the response with useful details
        scale_factors = {
            "x": target_width / source_width,
            "y": target_height / source_height
        }

        return [types.TextContent(
            type="text",
            text=f"""Mouse click (button {button}) from source ({x}, {y}) to target ({scaled_x}, {scaled_y}) {'succeeded' if result else 'failed'}
Source dimensions: {source_width}x{source_height}
Target dimensions: {target_width}x{target_height}
Scale factors: {scale_factors['x']:.4f}x, {scale_factors['y']:.4f}y"""
        )]
    finally:
        # Close VNC connection
        vnc.close()

# From src/action_handlers.py
def handle_remote_macos_send_keys(arguments: dict[str, Any]) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Send keyboard input to a remote MacOs machine."""
    # Use environment variables
    host = MACOS_HOST
    port = MACOS_PORT
    password = MACOS_PASSWORD
    username = MACOS_USERNAME
    encryption = VNC_ENCRYPTION

    # Get required parameters from arguments
    text = arguments.get("text")
    special_key = arguments.get("special_key")
    key_combination = arguments.get("key_combination")

    if not text and not special_key and not key_combination:
        raise ValueError("Either text, special_key, or key_combination must be provided")

    # Initialize VNC client
    vnc = VNCClient(host=host, port=port, password=password, username=username, encryption=encryption)

    # Connect to remote MacOs machine
    success, error_message = vnc.connect()
    if not success:
        error_msg = f"Failed to connect to remote MacOs machine at {host}:{port}. {error_message}"
        return [types.TextContent(type="text", text=error_msg)]

    try:
        result_message = []

        # Map of special key names to X11 keysyms
        special_keys = {
            "enter": 0xff0d,
            "return": 0xff0d,
            "backspace": 0xff08,
            "tab": 0xff09,
            "escape": 0xff1b,
            "esc": 0xff1b,
            "delete": 0xffff,
            "del": 0xffff,
            "home": 0xff50,
            "end": 0xff57,
            "page_up": 0xff55,
            "page_down": 0xff56,
            "left": 0xff51,
            "up": 0xff52,
            "right": 0xff53,
            "down": 0xff54,
            "f1": 0xffbe,
            "f2": 0xffbf,
            "f3": 0xffc0,
            "f4": 0xffc1,
            "f5": 0xffc2,
            "f6": 0xffc3,
            "f7": 0xffc4,
            "f8": 0xffc5,
            "f9": 0xffc6,
            "f10": 0xffc7,
            "f11": 0xffc8,
            "f12": 0xffc9,
            "space": 0x20,
        }

        # Map of modifier key names to X11 keysyms
        modifier_keys = {
            "ctrl": 0xffe3,    # Control_L
            "control": 0xffe3,  # Control_L
            "shift": 0xffe1,   # Shift_L
            "alt": 0xffe9,     # Alt_L
            "option": 0xffe9,  # Alt_L (Mac convention)
            "cmd": 0xffeb,     # Command_L (Mac convention)
            "command": 0xffeb,  # Command_L (Mac convention)
            "win": 0xffeb,     # Command_L
            "super": 0xffeb,   # Command_L
            "fn": 0xffed,      # Function key
            "meta": 0xffeb,    # Command_L (Mac convention)
        }

        # Map for letter keys (a-z)
        letter_keys = {chr(i): i for i in range(ord('a'), ord('z') + 1)}

        # Map for number keys (0-9)
        number_keys = {str(i): ord(str(i)) for i in range(10)}

        # Process special key
        if special_key:
            if special_key.lower() in special_keys:
                key = special_keys[special_key.lower()]
                if vnc.send_key_event(key, True) and vnc.send_key_event(key, False):
                    result_message.append(f"Sent special key: {special_key}")
                else:
                    result_message.append(f"Failed to send special key: {special_key}")
            else:
                result_message.append(f"Unknown special key: {special_key}")
                result_message.append(f"Supported special keys: {', '.join(special_keys.keys())}")

        # Process text
        if text:
            if vnc.send_text(text):
                result_message.append(f"Sent text: '{text}'")
            else:
                result_message.append(f"Failed to send text: '{text}'")

        # Process key combination
        if key_combination:
            keys = []
            for part in key_combination.lower().split('+'):
                part = part.strip()
                if part in modifier_keys:
                    keys.append(modifier_keys[part])
                elif part in special_keys:
                    keys.append(special_keys[part])
                elif part in letter_keys:
                    keys.append(letter_keys[part])
                elif part in number_keys:
                    keys.append(number_keys[part])
                elif len(part) == 1:
                    # For any other single character keys
                    keys.append(ord(part))
                else:
                    result_message.append(f"Unknown key in combination: {part}")
                    break

            if len(keys) == len(key_combination.split('+')):
                if vnc.send_key_combination(keys):
                    result_message.append(f"Sent key combination: {key_combination}")
                else:
                    result_message.append(f"Failed to send key combination: {key_combination}")

        return [types.TextContent(type="text", text="\n".join(result_message))]
    finally:
        vnc.close()

# From src/action_handlers.py
def handle_remote_macos_mouse_double_click(arguments: dict[str, Any]) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Perform a mouse double-click action on a remote MacOs machine."""
    # Use environment variables
    host = MACOS_HOST
    port = MACOS_PORT
    password = MACOS_PASSWORD
    username = MACOS_USERNAME
    encryption = VNC_ENCRYPTION

    # Get required parameters from arguments
    x = arguments.get("x")
    y = arguments.get("y")
    source_width = int(arguments.get("source_width", 1366))
    source_height = int(arguments.get("source_height", 768))
    button = int(arguments.get("button", 1))

    if x is None or y is None:
        raise ValueError("x and y coordinates are required")

    # Ensure source dimensions are positive
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive values")

    # Initialize VNC client
    vnc = VNCClient(host=host, port=port, password=password, username=username, encryption=encryption)

    # Connect to remote MacOs machine
    success, error_message = vnc.connect()
    if not success:
        error_msg = f"Failed to connect to remote MacOs machine at {host}:{port}. {error_message}"
        return [types.TextContent(type="text", text=error_msg)]

    try:
        # Get target screen dimensions
        target_width = vnc.width
        target_height = vnc.height

        # Scale coordinates
        scaled_x = int((x / source_width) * target_width)
        scaled_y = int((y / source_height) * target_height)

        # Ensure coordinates are within the screen bounds
        scaled_x = max(0, min(scaled_x, target_width - 1))
        scaled_y = max(0, min(scaled_y, target_height - 1))

        # Double click
        result = vnc.send_mouse_click(scaled_x, scaled_y, button, True)

        # Prepare the response with useful details
        scale_factors = {
            "x": target_width / source_width,
            "y": target_height / source_height
        }

        return [types.TextContent(
            type="text",
            text=f"""Mouse double-click (button {button}) from source ({x}, {y}) to target ({scaled_x}, {scaled_y}) {'succeeded' if result else 'failed'}
Source dimensions: {source_width}x{source_height}
Target dimensions: {target_width}x{target_height}
Scale factors: {scale_factors['x']:.4f}x, {scale_factors['y']:.4f}y"""
        )]
    finally:
        # Close VNC connection
        vnc.close()

# From src/action_handlers.py
def handle_remote_macos_mouse_move(arguments: dict[str, Any]) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Move the mouse cursor on a remote MacOs machine."""
    # Use environment variables
    host = MACOS_HOST
    port = MACOS_PORT
    password = MACOS_PASSWORD
    username = MACOS_USERNAME
    encryption = VNC_ENCRYPTION

    # Get required parameters from arguments
    x = arguments.get("x")
    y = arguments.get("y")
    source_width = int(arguments.get("source_width", 1366))
    source_height = int(arguments.get("source_height", 768))

    if x is None or y is None:
        raise ValueError("x and y coordinates are required")

    # Ensure source dimensions are positive
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive values")

    # Initialize VNC client
    vnc = VNCClient(host=host, port=port, password=password, username=username, encryption=encryption)

    # Connect to remote MacOs machine
    success, error_message = vnc.connect()
    if not success:
        error_msg = f"Failed to connect to remote MacOs machine at {host}:{port}. {error_message}"
        return [types.TextContent(type="text", text=error_msg)]

    try:
        # Get target screen dimensions
        target_width = vnc.width
        target_height = vnc.height

        # Scale coordinates
        scaled_x = int((x / source_width) * target_width)
        scaled_y = int((y / source_height) * target_height)

        # Ensure coordinates are within the screen bounds
        scaled_x = max(0, min(scaled_x, target_width - 1))
        scaled_y = max(0, min(scaled_y, target_height - 1))

        # Move mouse pointer (button_mask=0 means no buttons are pressed)
        result = vnc.send_pointer_event(scaled_x, scaled_y, 0)

        # Prepare the response with useful details
        scale_factors = {
            "x": target_width / source_width,
            "y": target_height / source_height
        }

        return [types.TextContent(
            type="text",
            text=f"""Mouse move from source ({x}, {y}) to target ({scaled_x}, {scaled_y}) {'succeeded' if result else 'failed'}
Source dimensions: {source_width}x{source_height}
Target dimensions: {target_width}x{target_height}
Scale factors: {scale_factors['x']:.4f}x, {scale_factors['y']:.4f}y"""
        )]
    finally:
        # Close VNC connection
        vnc.close()

# From src/action_handlers.py
def handle_remote_macos_open_application(arguments: dict[str, Any]) -> List[types.TextContent]:
    """
    Opens or activates an application on the remote MacOS machine using VNC.

    Args:
        arguments: Dictionary containing:
            - identifier: App name, path, or bundle ID

    Returns:
        List containing a TextContent with the result
    """
    # Use environment variables
    host = MACOS_HOST
    port = MACOS_PORT
    password = MACOS_PASSWORD
    username = MACOS_USERNAME
    encryption = VNC_ENCRYPTION

    identifier = arguments.get("identifier")
    if not identifier:
        raise ValueError("identifier is required")

    start_time = time.time()

    # Initialize VNC client
    vnc = VNCClient(host=host, port=port, password=password, username=username, encryption=encryption)

    # Connect to remote MacOs machine
    success, error_message = vnc.connect()
    if not success:
        error_msg = f"Failed to connect to remote MacOs machine at {host}:{port}. {error_message}"
        return [types.TextContent(type="text", text=error_msg)]

    try:
        # Send Command+Space to open Spotlight
        cmd_key = 0xffeb  # Command key
        space_key = 0x20  # Space key

        # Press Command+Space
        vnc.send_key_event(cmd_key, True)
        vnc.send_key_event(space_key, True)

        # Release Command+Space
        vnc.send_key_event(space_key, False)
        vnc.send_key_event(cmd_key, False)

        # Small delay to let Spotlight open
        time.sleep(0.5)

        # Type the application name
        vnc.send_text(identifier)

        # Small delay to let Spotlight find the app
        time.sleep(0.5)

        # Press Enter to launch
        enter_key = 0xff0d
        vnc.send_key_event(enter_key, True)
        vnc.send_key_event(enter_key, False)

        end_time = time.time()
        processing_time = round(end_time - start_time, 3)

        return [types.TextContent(
            type="text",
            text=f"Launched application: {identifier}\nProcessing time: {processing_time}s"
        )]

    finally:
        # Close VNC connection
        vnc.close()

# From src/action_handlers.py
def handle_remote_macos_mouse_drag_n_drop(arguments: dict[str, Any]) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Perform a mouse drag operation on a remote MacOs machine."""
    # Use environment variables
    host = MACOS_HOST
    port = MACOS_PORT
    password = MACOS_PASSWORD
    username = MACOS_USERNAME
    encryption = VNC_ENCRYPTION

    # Get required parameters from arguments
    start_x = arguments.get("start_x")
    start_y = arguments.get("start_y")
    end_x = arguments.get("end_x")
    end_y = arguments.get("end_y")
    source_width = int(arguments.get("source_width", 1366))
    source_height = int(arguments.get("source_height", 768))
    button = int(arguments.get("button", 1))
    steps = int(arguments.get("steps", 10))
    delay_ms = int(arguments.get("delay_ms", 10))

    # Validate required parameters
    if any(x is None for x in [start_x, start_y, end_x, end_y]):
        raise ValueError("start_x, start_y, end_x, and end_y coordinates are required")

    # Ensure source dimensions are positive
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive values")

    # Initialize VNC client
    vnc = VNCClient(host=host, port=port, password=password, username=username, encryption=encryption)

    # Connect to remote MacOs machine
    success, error_message = vnc.connect()
    if not success:
        error_msg = f"Failed to connect to remote MacOs machine at {host}:{port}. {error_message}"
        return [types.TextContent(type="text", text=error_msg)]

    try:
        # Get target screen dimensions
        target_width = vnc.width
        target_height = vnc.height

        # Scale coordinates
        scaled_start_x = int((start_x / source_width) * target_width)
        scaled_start_y = int((start_y / source_height) * target_height)
        scaled_end_x = int((end_x / source_width) * target_width)
        scaled_end_y = int((end_y / source_height) * target_height)

        # Ensure coordinates are within the screen bounds
        scaled_start_x = max(0, min(scaled_start_x, target_width - 1))
        scaled_start_y = max(0, min(scaled_start_y, target_height - 1))
        scaled_end_x = max(0, min(scaled_end_x, target_width - 1))
        scaled_end_y = max(0, min(scaled_end_y, target_height - 1))

        # Calculate step sizes
        dx = (scaled_end_x - scaled_start_x) / steps
        dy = (scaled_end_y - scaled_start_y) / steps

        # Move to start position
        if not vnc.send_pointer_event(scaled_start_x, scaled_start_y, 0):
            return [types.TextContent(type="text", text="Failed to move to start position")]

        # Press button
        button_mask = 1 << (button - 1)
        if not vnc.send_pointer_event(scaled_start_x, scaled_start_y, button_mask):
            return [types.TextContent(type="text", text="Failed to press mouse button")]

        # Perform drag
        for step in range(1, steps + 1):
            current_x = int(scaled_start_x + dx * step)
            current_y = int(scaled_start_y + dy * step)
            if not vnc.send_pointer_event(current_x, current_y, button_mask):
                return [types.TextContent(type="text", text=f"Failed during drag at step {step}")]
            time.sleep(delay_ms / 1000.0)  # Convert ms to seconds

        # Release button at final position
        if not vnc.send_pointer_event(scaled_end_x, scaled_end_y, 0):
            return [types.TextContent(type="text", text="Failed to release mouse button")]

        # Prepare the response with useful details
        scale_factors = {
            "x": target_width / source_width,
            "y": target_height / source_height
        }

        return [types.TextContent(
            type="text",
            text=f"""Mouse drag (button {button}) completed:
From source ({start_x}, {start_y}) to ({end_x}, {end_y})
From target ({scaled_start_x}, {scaled_start_y}) to ({scaled_end_x}, {scaled_end_y})
Source dimensions: {source_width}x{source_height}
Target dimensions: {target_width}x{target_height}
Scale factors: {scale_factors['x']:.4f}x, {scale_factors['y']:.4f}y
Steps: {steps}
Delay: {delay_ms}ms"""
        )]

    finally:
        # Close VNC connection
        vnc.close()

import pytest
from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import AsyncMock

# From tests/conftest.py
def configure_event_loop():
    """
    Configure the event loop policy based on the environment.
    - Use WindowsSelectorEventLoopPolicy on Windows
    - Use default policy on other platforms but with a custom loop factory for CI
    """
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    else:
        # Check if we're in a CI environment
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            # Set default event loop policy but with a simple loop factory
            # This avoids issues with file descriptors in containers
            
            # Create a new event loop that doesn't try to add file descriptors
            # that might cause permission issues in CI environments
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Disable signal handling in event loops to avoid permission issues
            if hasattr(loop, '_handle_signals'):
                loop._handle_signals = lambda: None
            
            if hasattr(loop, '_signal_handlers'):
                loop._signal_handlers = {}

# From tests/conftest.py
def mock_global_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'MACOS_HOST': 'test-host',
        'MACOS_PORT': '5900',
        'MACOS_USERNAME': 'test-user',
        'MACOS_PASSWORD': 'test-password',
        'VNC_ENCRYPTION': 'prefer_on'
    }):
        yield

# From tests/conftest.py
def global_mock_vnc_client():
    """Provide a mock VNCClient for testing."""
    with patch('src.vnc_client.VNCClient') as mock_vnc_class:
        mock_instance = MagicMock()
        mock_vnc_class.return_value = mock_instance
        
        # Set up common mock properties
        mock_instance.width = 1366
        mock_instance.height = 768
        mock_instance.connect.return_value = (True, None)
        
        yield mock_instance

from base64 import b64encode
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
from mcp.server import Server
import mcp.server.stdio
from livekit import api
from livekit_handler import LiveKitHandler
from action_handlers import handle_remote_macos_get_screen
from action_handlers import handle_remote_macos_mouse_scroll
from action_handlers import handle_remote_macos_send_keys
from action_handlers import handle_remote_macos_mouse_move
from action_handlers import handle_remote_macos_mouse_click
from action_handlers import handle_remote_macos_mouse_double_click
from action_handlers import handle_remote_macos_open_application
from action_handlers import handle_remote_macos_mouse_drag_n_drop

from typing import Callable
from livekit.rtc import Room
from livekit.rtc import RemoteParticipant
from livekit.rtc import DataPacketKind

# From mcp_remote_macos_use/livekit_handler.py
class LiveKitHandler:
    def __init__(self):
        self.room: Optional[Room] = None
        self._message_handlers: Dict[str, Callable] = {}
        
        # LiveKit configuration
        self.url = os.getenv('LIVEKIT_URL')
        self.api_key = os.getenv('LIVEKIT_API_KEY')
        self.api_secret = os.getenv('LIVEKIT_API_SECRET')
        
        if not all([self.url, self.api_key, self.api_secret]):
            logger.warning("LiveKit environment variables not fully configured")
            return
            
        logger.info("LiveKit configuration loaded")

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type"""
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

    async def handle_data_message(self, data: bytes, participant: RemoteParticipant):
        """Handle incoming data messages"""
        try:
            message = data.decode('utf-8')
            logger.info(f"Received data message from {participant.identity}: {message}")
            
            # Call appropriate handler if registered
            if message in self._message_handlers:
                await self._message_handlers[message](participant)
            
        except Exception as e:
            logger.error(f"Error handling data message: {str(e)}")

    async def start(self, room_name: str, token: str):
        """Start LiveKit connection"""
        if not all([self.url, self.api_key, self.api_secret]):
            logger.error("LiveKit environment variables not configured")
            return False

        try:
            self.room = Room()
            
            @self.room.on("participant_connected")
            def on_participant_connected(participant: RemoteParticipant):
                logger.info(f"participant connected: {participant.sid} {participant.identity}")

            @self.room.on("data_received")
            def on_data_received(data: bytes, participant: RemoteParticipant):
                asyncio.create_task(self.handle_data_message(data, participant))

            # Connect to the room with auto_subscribe disabled since we only need data channel
            await self.room.connect(self.url, token, auto_subscribe=False)
            logger.info(f"Connected to LiveKit room: {room_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to start LiveKit: {str(e)}")
            return False

    async def send_data(self, message: str, reliable: bool = True):
        """Send data to all participants in the room"""
        if not self.room:
            logger.error("Room not initialized")
            return False

        try:
            await self.room.local_participant.publish_data(
                message.encode('utf-8'),
                kind=DataPacketKind.RELIABLE if reliable else DataPacketKind.LOSSY
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send data: {str(e)}")
            return False

    async def stop(self):
        """Stop LiveKit connection"""
        if self.room:
            try:
                await self.room.disconnect()
                logger.info("Disconnected from LiveKit room")
            except Exception as e:
                logger.error(f"Error disconnecting from LiveKit: {str(e)}")

# From mcp_remote_macos_use/livekit_handler.py
def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type"""
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

# From mcp_remote_macos_use/livekit_handler.py
def on_participant_connected(participant: RemoteParticipant):
                logger.info(f"participant connected: {participant.sid} {participant.identity}")

# From mcp_remote_macos_use/livekit_handler.py
def on_data_received(data: bytes, participant: RemoteParticipant):
                asyncio.create_task(self.handle_data_message(data, participant))

from langchain_core.tools import tool
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# From tools/fetch_web_page_raw_html.py
def fetch_web_page_raw_html(url: str) -> str:
    """Fetches the raw HTML of a web page. If a CSS selector is provided, returns only the matching elements."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument("--disable-gpu")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
        
    service = Service('/usr/bin/chromedriver')
    
    driver = webdriver.Chrome(options=options, service=service)

    driver.get(url)

    return driver.execute_script("return document.body.outerHTML;")

from langchain_community.tools import DuckDuckGoSearchResults

# From tools/duck_duck_go_news_search.py
def duck_duck_go_news_search(query: str):
    """Search for news using DuckDuckGo."""
    return DuckDuckGoSearchResults(backend="news").invoke(query)


# From tools/overwrite_file.py
def overwrite_file(file_path: str, content: str) -> str:
    """Replaces the file at the given path with the given content, returning a string message confirming success."""
    with open(file_path, 'w') as file:
        file.write(content)
    return f"File at {file_path} has been successfully overwritten."


# From tools/duck_duck_go_web_search.py
def duck_duck_go_web_search(query: str):
    """Search the web using DuckDuckGo."""
    return DuckDuckGoSearchResults().invoke(query)

from langchain_community.document_loaders.url_selenium import SeleniumURLLoader

# From tools/fetch_web_page_content.py
def fetch_web_page_content(url: str):
    """Fetch content from a web page."""
    loader = SeleniumURLLoader(
        urls=[url],
        executable_path="/usr/bin/chromedriver",
        arguments=['--headless', '--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
    )
    pages = loader.load()
    
    return pages[0]


# From tools/request_human_input.py
def request_human_input(prompt: str) -> str:
    """Request human input via Python's input method."""
    print(prompt)
    return input("> ")

import utils

# From tools/list_available_agents.py
def list_available_agents():
    """List the name of available agents along with the type of task it's designed to be assigned."""
    return utils.all_agents()


# From tools/run_shell_command.py
def run_shell_command(command: str):
    """Run a shell command and return the output."""
    print(f"Running shell command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return { "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode }


# From tools/read_file.py
def read_file(file_path: str) -> str:
    """Returns the content of the file at the given file path."""
    with open(file_path, 'r') as file:
        return file.read()


# From tools/assign_agent_to_task.py
def assign_agent_to_task(agent_name: str, task: str):
    """Assign an agent to a task. This function returns the response from the agent."""
    print(f"Assigning agent {agent_name} to task: {task}")
    # Handle the case where the call to the agent fails (might be a job for the toolmaker)
    try:
        agent_module = utils.load_module(f"agents/{agent_name}.py")
        agent_function = getattr(agent_module, agent_name)
        result = agent_function(task=task)
        del sys.modules[agent_module.__name__]
        response = result["messages"][-1].content
        print(f"{agent_name} responded:")
        print(response)
        return response
    except Exception as e:
        exception_trace = traceback.format_exc()
        error = f"An error occurred while assigning {agent_name} to task {task}:\n {e}\n{exception_trace}"
        print(error)
        return error


# From tools/write_to_file.py
def write_to_file(file: str, file_contents: str) -> str:
    """Write the contents to a new file, will not overwrite an existing file."""
    if os.path.exists(file):
        raise FileExistsError(f"File {file} already exists and will not be overwritten.")

    print(f"Writing to file: {file}")
    with open(file, 'w') as f:
        f.write(file_contents)

    return f"File {file} written successfully."


# From tools/delete_file.py
def delete_file(file_path: str) -> str:
    """Deletes the file at the given path and returns a string confirming success."""
    try:
        os.remove(file_path)
        return f"File at {file_path} has been deleted successfully."
    except Exception as e:
        return str(e)

from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
import config

# From agents/tool_maker.py
def reasoning(state: MessagesState):
    print()
    print("tool_maker is thinking...")
    messages = state['messages']
    tooled_up_model = config.default_langchain_model.bind_tools(tools)
    response = tooled_up_model.invoke(messages)
    return {"messages": [response]}

# From agents/tool_maker.py
def check_for_tool_calls(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    
    if last_message.tool_calls:
        if not last_message.content.strip() == "":
            print("tool_maker thought this:")
            print(last_message.content)
        print()
        print("tool_maker is acting by invoking these tools:")
        print([tool_call["name"] for tool_call in last_message.tool_calls])
        return "tools"
    
    return END

# From agents/tool_maker.py
def tool_maker(task: str) -> str:
    """Creates new tools for agents to use."""
    return graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(task)]}
    )

from tools.duck_duck_go_web_search import duck_duck_go_web_search
from tools.fetch_web_page_content import fetch_web_page_content

# From agents/web_researcher.py
def web_researcher(task: str) -> str:
    """Researches the web."""
    return graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(task)]}
    )


# From agents/agent_smith.py
def agent_smith(task: str) -> str:
    """Designs and implements new agents, each designed to play a unique role."""
    return graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(task)]}
    )

from tools.list_available_agents import list_available_agents
from tools.assign_agent_to_task import assign_agent_to_task

# From agents/hermes.py
def feedback_and_wait_on_human_input(state: MessagesState):
    # if messages only has one element we need to start the conversation
    if len(state['messages']) == 1:
        message_to_human = "What can I help you with?"
    else:
        message_to_human = state["messages"][-1].content
    
    print(message_to_human)

    human_input = ""
    while not human_input.strip():
        human_input = input("> ")
    
    return {"messages": [HumanMessage(human_input)]}

# From agents/hermes.py
def check_for_exit(state: MessagesState) -> Literal["reasoning", END]:
    last_message = state['messages'][-1]
    if last_message.content.lower() == "exit":
        return END
    else:
        return "reasoning"

# From agents/hermes.py
def hermes(uuid: str):
    """The orchestrator that interacts with the user to understand goals, plan out how agents can meet the goal, assign tasks, and coordinate the activities agents."""
    print(f"Starting session with AgentK (id:{uuid})")
    print("Type 'exit' to end the session.")

    return graph.invoke(
        {"messages": [SystemMessage(system_prompt)]},
        config={"configurable": {"thread_id": uuid}}
    )

from tools.write_to_file import write_to_file
from tools.overwrite_file import overwrite_file
from tools.delete_file import delete_file
from tools.read_file import read_file
from tools.run_shell_command import run_shell_command

# From agents/software_engineer.py
def software_engineer(task: str) -> str:
    """Creates, modifies, and deletes code, manages files, runs shell commands, and collaborates with other agents."""
    return graph.invoke(
        {"messages": [SystemMessage(system_prompt), HumanMessage(task)]}
    )


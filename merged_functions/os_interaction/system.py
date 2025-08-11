# Merged file for os_interaction/system
# This file contains code merged from multiple repositories

import pyautogui
import platform
import time
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

from operate.config import Config

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

import base64
import io
import json
import os
import traceback
import easyocr
import ollama
import pkg_resources
from PIL import Image
from ultralytics import YOLO
from operate.exceptions import ModelNotRecognizedException
from operate.models.prompts import get_system_prompt
from operate.models.prompts import get_user_first_message_prompt
from operate.models.prompts import get_user_prompt
from operate.utils.label import add_labels
from operate.utils.label import get_click_position_in_percent
from operate.utils.label import get_label_coordinates
from operate.utils.ocr import get_text_coordinates
from operate.utils.ocr import get_text_element
from operate.utils.screenshot import capture_screen_with_cursor
from operate.utils.screenshot import compress_screenshot
from operate.utils.style import ANSI_BRIGHT_MAGENTA
from operate.utils.style import ANSI_GREEN
from operate.utils.style import ANSI_RED
from operate.utils.style import ANSI_RESET

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

import datetime
import http.server
import signal
import socketserver
import subprocess
import argparse
import sys
import threading
import re

# From MacOS-Agent/macos_agent_server.py
class DeferredLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def print_messages(self):
        for message in self.messages:
            print(message)
        self.messages = []

# From MacOS-Agent/macos_agent_server.py
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

# From MacOS-Agent/macos_agent_server.py
class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

# From MacOS-Agent/macos_agent_server.py
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

# From MacOS-Agent/macos_agent_server.py
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

# From MacOS-Agent/macos_agent_server.py
def info(self, message):
        self.messages.append(message)

# From MacOS-Agent/macos_agent_server.py
def print_messages(self):
        for message in self.messages:
            print(message)
        self.messages = []

# From MacOS-Agent/macos_agent_server.py
def log_request(self, code="-", size="-"):
        super().log_request(code, size)
        self.server.deferred_logger.print_messages()
        sys.stderr.write("\n")

# From MacOS-Agent/macos_agent_server.py
def deferred_info(self, message):
        self.server.deferred_logger.info(message)

# From MacOS-Agent/macos_agent_server.py
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

# From MacOS-Agent/macos_agent_server.py
def handle_request_point(self, data):
        point = data.get("point")
        handlers = {
            "ping": lambda _: {"result": "pong"},
            "get_llm_system_prompt": lambda _: self.get_llm_system_prompt(),
            "execute_script": lambda d: self.execute_script_request(d),
        }
        return handlers.get(point, lambda _: None)(data)

# From MacOS-Agent/macos_agent_server.py
def get_llm_system_prompt(self, with_knowledge=True):
        template = self.load_prompt_template()
        return template.format(
            os_version=self.get_os_version(),
            current_time=self.get_current_time(),
            knowledge=(self.get_knowledge() if with_knowledge else ""),
        ).strip()

# From MacOS-Agent/macos_agent_server.py
def get_llm_reply_prompt(self, llm_output, execution):
        template = self.load_reply_prompt_template()
        return template.format(
            llm_system_prompt=self.get_llm_system_prompt(with_knowledge=False),
            llm_output=llm_output,
            execution=execution,
        ).strip()

# From MacOS-Agent/macos_agent_server.py
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

# From MacOS-Agent/macos_agent_server.py
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

# From MacOS-Agent/macos_agent_server.py
def get_os_version(self):
        return (
            subprocess.check_output(["sw_vers", "-productName"]).decode("utf-8").strip()
            + " "
            + subprocess.check_output(["sw_vers", "-productVersion"])
            .decode("utf-8")
            .strip()
        )

# From MacOS-Agent/macos_agent_server.py
def get_current_time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# From MacOS-Agent/macos_agent_server.py
def get_knowledge(self):
        try:
            with open("knowledge.md", "r") as file:
                return file.read().strip()
        except FileNotFoundError:
            return ""

# From MacOS-Agent/macos_agent_server.py
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

# From MacOS-Agent/macos_agent_server.py
def extract_scripts(self, llm_output):
        # Extract all code block content from the llm_output
        scripts = re.findall(r"```applescript(.*?)```", llm_output, re.DOTALL)
        return list(set(scripts))

# From MacOS-Agent/macos_agent_server.py
def extract_user_goal(self, llm_output):
        match = re.search(r"<user_goal>(.*?)</user_goal>", llm_output, re.DOTALL)
        return match.group(1).strip() if match else ""

# From MacOS-Agent/macos_agent_server.py
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

# From MacOS-Agent/macos_agent_server.py
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


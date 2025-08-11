# Merged file for os_interaction/utils
# This file contains code merged from multiple repositories

from setuptools import setup
from setuptools import find_packages


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

import argparse
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

import sys
import os
import time
import asyncio
from prompt_toolkit.shortcuts import message_dialog
from prompt_toolkit import prompt
from operate.exceptions import ModelNotRecognizedException
import platform
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
def main(model, terminal_prompt, voice_mode=False, verbose_mode=False):
    """
    Main function for the Self-Operating Computer.

    Parameters:
    - model: The model used for generating responses.
    - terminal_prompt: A string representing the prompt provided in the terminal.
    - voice_mode: A boolean indicating whether to enable voice mode.

    Returns:
    None
    """

    mic = None
    # Initialize `WhisperMic`, if `voice_mode` is True

    config.verbose = verbose_mode
    config.validation(model, voice_mode)

    if voice_mode:
        try:
            from whisper_mic import WhisperMic

            # Initialize WhisperMic if import is successful
            mic = WhisperMic()
        except ImportError:
            print(
                "Voice mode requires the 'whisper_mic' module. Please install it using 'pip install -r requirements-audio.txt'"
            )
            sys.exit(1)

    # Skip message dialog if prompt was given directly
    if not terminal_prompt:
        message_dialog(
            title="Self-Operating Computer",
            text="An experimental framework to enable multimodal models to operate computers",
            style=style,
        ).run()

    else:
        print("Running direct prompt...")

    # # Clear the console
    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033c", end="")

    if terminal_prompt:  # Skip objective prompt if it was given as an argument
        objective = terminal_prompt
    elif voice_mode:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RESET} Listening for your command... (speak now)"
        )
        try:
            objective = mic.listen()
        except Exception as e:
            print(f"{ANSI_RED}Error in capturing voice input: {e}{ANSI_RESET}")
            return  # Exit if voice input fails
    else:
        print(
            f"[{ANSI_GREEN}Self-Operating Computer {ANSI_RESET}|{ANSI_BRIGHT_MAGENTA} {model}{ANSI_RESET}]\n{USER_QUESTION}"
        )
        print(f"{ANSI_YELLOW}[User]{ANSI_RESET}")
        objective = prompt(style=style)

    system_prompt = get_system_prompt(model, objective)
    system_message = {"role": "system", "content": system_prompt}
    messages = [system_message]

    loop_count = 0

    session_id = None

    while True:
        if config.verbose:
            print("[Self Operating Computer] loop_count", loop_count)
        try:
            operations, session_id = asyncio.run(
                get_next_action(model, messages, objective, session_id)
            )

            stop = operate(operations, model)
            if stop:
                break

            loop_count += 1
            if loop_count > 10:
                break
        except ModelNotRecognizedException as e:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}"
            )
            break
        except Exception as e:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] -> {e} {ANSI_RESET}"
            )
            break

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
from dotenv import load_dotenv
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
import base64
import json

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

# From utils/style.py
def supports_ansi():
    """
    Check if the terminal supports ANSI escape codes
    """
    plat = platform.system()
    supported_platform = plat != "Windows" or "ANSICON" in os.environ
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return supported_platform and is_a_tty

import re

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

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import socket
import pyDes
from base64 import b64encode
from mcp.server.models import InitializationOptions
import mcp.types
from mcp.server import NotificationOptions
from mcp.server import Server
import mcp.server.stdio
from livekit import api
from livekit_handler import LiveKitHandler
from vnc_client import VNCClient
from vnc_client import capture_vnc_screen
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


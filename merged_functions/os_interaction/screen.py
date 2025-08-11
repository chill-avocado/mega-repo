# Merged file for os_interaction/screen
# This file contains code merged from multiple repositories

import sys
import os
import subprocess
import platform
import base64
import json
import openai
import argparse
from dotenv import load_dotenv

# From self-operating-computer/evaluate.py
def supports_ansi():
    """
    Check if the terminal supports ANSI escape codes
    """
    plat = platform.system()
    supported_platform = plat != "Windows" or "ANSICON" in os.environ
    is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return supported_platform and is_a_tty

# From self-operating-computer/evaluate.py
def format_evaluation_prompt(guideline):
    prompt = EVALUATION_PROMPT.format(guideline=guideline)
    return prompt

# From self-operating-computer/evaluate.py
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

# From self-operating-computer/evaluate.py
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

# From self-operating-computer/evaluate.py
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

# From self-operating-computer/evaluate.py
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

# From self-operating-computer/evaluate.py
def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    model = get_test_model()

    print(f"{ANSI_BLUE}[EVALUATING MODEL `{model}`]{ANSI_RESET}")
    print(f"{ANSI_BRIGHT_MAGENTA}[STARTING EVALUATION]{ANSI_RESET}")

    passed = 0
    failed = 0
    for objective, guideline in TEST_CASES.items():
        print(f"{ANSI_BLUE}[EVALUATING]{ANSI_RESET} '{objective}'")

        result = run_test_case(objective, guideline, model)
        if result:
            print(f"{ANSI_GREEN}[PASSED]{ANSI_RESET} '{objective}'")
            passed += 1
        else:
            print(f"{ANSI_RED}[FAILED]{ANSI_RESET} '{objective}'")
            failed += 1

    print(
        f"{ANSI_BRIGHT_MAGENTA}[EVALUATION COMPLETE]{ANSI_RESET} {passed} test{'' if passed == 1 else 's'} passed, {failed} test{'' if failed == 1 else 's'} failed"
    )

import pyautogui
from PIL import Image
from PIL import ImageDraw
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


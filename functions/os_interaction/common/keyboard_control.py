# Common keyboard_control component for os_interaction
def send_keys(text, interval=0.0):
    """
    Send keystrokes to the active window.
    
    Args:
        text (str): The text to type.
        interval (float, optional): Interval between keystrokes in seconds.
    """
    try:
        import pyautogui
        pyautogui.write(text, interval=interval)
    except ImportError:
        try:
            # Fall back to pynput if pyautogui is not available
            from pynput.keyboard import Controller
            keyboard = Controller()
            for char in text:
                keyboard.press(char)
                keyboard.release(char)
                if interval > 0:
                    import time
                    time.sleep(interval)
        except ImportError:
            raise ImportError("Either pyautogui or pynput is required for keyboard control")

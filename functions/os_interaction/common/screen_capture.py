# Common screen_capture component for os_interaction
def capture_screen(region=None):
    """
    Capture the screen or a region of the screen.
    
    Args:
        region (tuple, optional): Region to capture as (x, y, width, height).
            If None, captures the entire screen.
    
    Returns:
        image: The captured screen image.
    """
    try:
        # Try to use PIL first
        from PIL import ImageGrab
        if region:
            return ImageGrab.grab(bbox=region)
        else:
            return ImageGrab.grab()
    except ImportError:
        try:
            # Fall back to pyautogui if PIL is not available
            import pyautogui
            if region:
                return pyautogui.screenshot(region=region)
            else:
                return pyautogui.screenshot()
        except ImportError:
            raise ImportError("Either PIL or pyautogui is required for screen capture")

# Common browser_control component for browser_automation
class BrowserControl:
    """Basic browser control functionality."""
    
    def __init__(self, headless=False):
        """
        Initialize the browser controller.
        
        Args:
            headless (bool): Whether to run the browser in headless mode.
        """
        self.headless = headless
        self.browser = None
    
    def start(self):
        """Start the browser."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            options = Options()
            if self.headless:
                options.add_argument('--headless')
            
            self.browser = webdriver.Chrome(options=options)
            return True
        except ImportError:
            raise ImportError("Selenium is required for browser automation")
    
    def navigate(self, url):
        """Navigate to a URL."""
        if not self.browser:
            self.start()
        self.browser.get(url)
    
    def find_element(self, selector, by_type='css'):
        """Find an element on the page."""
        if not self.browser:
            raise RuntimeError("Browser not started")
        
        from selenium.webdriver.common.by import By
        
        by_map = {
            'css': By.CSS_SELECTOR,
            'id': By.ID,
            'name': By.NAME,
            'xpath': By.XPATH,
            'link_text': By.LINK_TEXT,
            'tag': By.TAG_NAME,
            'class': By.CLASS_NAME
        }
        
        by_type = by_type.lower()
        if by_type not in by_map:
            raise ValueError(f"Invalid selector type: {by_type}")
        
        return self.browser.find_element(by_map[by_type], selector)
    
    def close(self):
        """Close the browser."""
        if self.browser:
            self.browser.quit()
            self.browser = None

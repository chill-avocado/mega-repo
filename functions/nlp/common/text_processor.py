# Common text_processor component for nlp
class TextProcessor:
    """Basic text processing functionality."""
    
    def __init__(self):
        """Initialize the text processor."""
        self.stop_words = set([
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
            'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
            'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on', 'at', 'by', 'with'
        ])
    
    def tokenize(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): Text to tokenize.
        
        Returns:
            list: List of tokens.
        """
        import re
        # Simple tokenization by splitting on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())
    
    def remove_stop_words(self, tokens):
        """
        Remove stop words from a list of tokens.
        
        Args:
            tokens (list): List of tokens.
        
        Returns:
            list: Tokens with stop words removed.
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def extract_entities(self, text):
        """
        Extract entities from text using simple pattern matching.
        
        Args:
            text (str): Text to analyze.
        
        Returns:
            dict: Dictionary of entity types and their values.
        """
        import re
        
        entities = {
            'emails': re.findall(r'[\w\.-]+@[\w\.-]+', text),
            'urls': re.findall(r'https?://[\w\.-/]+', text),
            'phone_numbers': re.findall(r'\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', text),
            'dates': re.findall(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text)
        }
        
        return entities
    
    def get_sentiment(self, text):
        """
        Get sentiment of text using a simple lexicon-based approach.
        
        Args:
            text (str): Text to analyze.
        
        Returns:
            float: Sentiment score (-1 to 1).
        """
        # Simple sentiment lexicon
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'wonderful', 'fantastic',
            'amazing', 'love', 'happy', 'best', 'better', 'awesome', 'nice',
            'perfect', 'pleasant', 'enjoy', 'liked', 'like', 'joy'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'negative', 'horrible', 'worst',
            'worse', 'hate', 'sad', 'disappointing', 'poor', 'unpleasant',
            'dislike', 'disliked', 'unfortunate', 'unhappy', 'sorry'
        }
        
        # Tokenize and remove stop words
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        
        # Count positive and negative words
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        # Calculate sentiment score
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total

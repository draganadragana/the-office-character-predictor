import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
print(word_tokenize("Hello, world!"))  # Should output ['Hello', ',', 'world', '!']
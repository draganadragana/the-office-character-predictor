import pandas as pd
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# Loading the dataset
df = pd.read_csv('D:/0 FAKS/2 letnik/machine learning and data mining/The Office - Character Predictor/data/selected-lines.csv')

# Cleaning the text (expand contractions, lowercase, keep stopwords)
def clean_text(text):
    text = contractions.fix(text)
    words = text.lower().split()  # tokenization that keeps stopwords
    words = [word for word in words if word.isalnum()]  # keeps only words
    return ' '.join(words)

df['Cleaned_Line'] = df['Line'].apply(clean_text)

# Adding additional features
df['Line_Length'] = df['Line'].apply(lambda x: len(x.split()))  # word count
df['Sentiment_Polarity'] = df['Line'].apply(lambda x: TextBlob(x).sentiment.polarity)  # range -1 to 1 for the sentiment score

# Vectorizing using TF-IDF (keeps 500 most important terms, stopwords are kept)
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(df['Cleaned_Line'])
y = df['Character']

# Saving the results
pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out()).to_csv('D:/0 FAKS/2 letnik/machine learning and data mining/The Office - Character Predictor/data/tfidf_features.csv', index=False)
df.to_csv('D:/0 FAKS/2 letnik/machine learning and data mining/The Office - Character Predictor/data/cleaned-dataset.csv', index=False)

# Printing exploration info
print("Character distribution:\n", df['Character'].value_counts())
print("\nSample cleaned line:", df['Cleaned_Line'].iloc[0])
print("\nExample line length and sentiment:\n", df[['Line_Length', 'Sentiment_Polarity']].head())

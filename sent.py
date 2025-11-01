import pandas as pd
from textblob import TextBlob

# Read the review data from the CSV file
input_file = "flipkart_reviews.csv"
output_file = "flipkart_reviews_with_sentiments.csv"

# Load the CSV into a pandas DataFrame
df = pd.read_csv(input_file)

# Define a function to analyze sentiment
def analyze_sentiment(review):
    analysis = TextBlob(review)
    # Determine sentiment polarity
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Apply the sentiment analysis function to each review
df["Sentiment"] = df["Review"].apply(analyze_sentiment)

# Save the results to a new CSV file
df.to_csv(output_file, index=False)

print(f"Sentiment analysis complete. Results saved to {output_file}")

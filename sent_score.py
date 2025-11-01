import pandas as pd
from textblob import TextBlob

# Load the review data
# Replace 'flipkart_reviews.csv' with the path to your CSV file
# The CSV should have columns "Smartphone" and "Review"
df = pd.read_csv("flipkart_reviews.csv")

# Check if the required columns exist
if "Phone Name" not in df.columns or "Review" not in df.columns:
    raise ValueError("The input CSV file must have 'Phone Name' and 'Review' columns.")

# Define aspects and their related keywords
aspects = {
    "Camera": ["camera", "photo", "picture", "lens"],
    "Performance": ["performance", "speed", "lag", "processor"],
    "Display": ["display", "screen", "resolution", "brightness"],
    "Battery Life": ["battery", "charging", "power", "backup"]
}

# Function to calculate sentiment polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Function to extract aspect-specific reviews and calculate polarity
def calculate_aspect_polarity_and_confidence(reviews, aspect_keywords):
    aspect_reviews = [review for review in reviews if any(keyword in review.lower() for keyword in aspect_keywords)]
    if not aspect_reviews:
        return 0, 0  # If no reviews mention the aspect, return 0 for both polarity and confidence
    polarities = [get_polarity(review) for review in aspect_reviews]
    avg_polarity = sum(polarities) / len(polarities)
    confidence = len(aspect_reviews) 
    #avg_polarity=avg_polarity*confidence # Confidence based on the number of reviews
    return avg_polarity, confidence

# Initialize results dictionary
results = []

# Iterate over each smartphone
for smartphone in df["Phone Name"].unique():
    smartphone_reviews = df[df["Phone Name"] == smartphone]["Review"].tolist()
    aspect_scores = {"Phone Name": smartphone}
    for aspect, keywords in aspects.items():
        avg_polarity, confidence = calculate_aspect_polarity_and_confidence(smartphone_reviews, keywords)
        aspect_scores[f"{aspect} Polarity"] = avg_polarity
        aspect_scores[f"{aspect} Confidence"] = confidence
    results.append(aspect_scores)

# Convert results to DataFrame
aspect_df = pd.DataFrame(results)

# Save to CSV
output_file = "aspect_sentiment_analysis_with_confidence.csv"
aspect_df.to_csv(output_file, index=False)

# Print results
print(f"Aspect-based sentiment analysis with confidence scores saved to {output_file}")
print(aspect_df)

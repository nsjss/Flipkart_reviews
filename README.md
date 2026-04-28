# 📱 Flipkart Phone Review Analysis & Recommendation System

An AI-powered system that scrapes real user reviews from Flipkart, performs **feature-level sentiment analysis**, and generates **data-driven phone recommendations**.

---

## 🚀 Overview

Most phone comparison platforms rely heavily on specifications or generic ratings, which often fail to reflect real user experience. Users often spend a significant amount of time reading hundreds of reviews to understand a product.

This project automates that entire process by extracting and analyzing real user reviews, converting unstructured text into meaningful insights, and generating personalized recommendations.

---

## 🧠 Key Features

- 🔎 **Web Scraping**  
  Automatically collects user reviews from Flipkart based on the phone name  

- 🧹 **Text Preprocessing**  
  Cleans raw review text using tokenization, stopword removal, and normalization  

- 💬 **Feature-Level Sentiment Analysis**  
  Extracts sentiment for specific features like:
  - Camera  
  - Battery  
  - Display  
  - Performance  

- 📊 **Feature Scoring System**  
  Assigns numerical scores to each feature based on sentiment  

- 🏆 **Recommendation Engine**  
  Ranks phones based on weighted scores and user preferences  

---

## ⚙️ Tech Stack

- **Language:** Python  

- **Libraries Used:**  
  - `BeautifulSoup` / `Selenium` – Web scraping  
  - `Pandas`, `NumPy` – Data processing  
  - `NLTK` / `TextBlob` – Sentiment analysis  
  - `re` – Text cleaning  

---

## 📈 Impact

- Processes **500–1000+ reviews per phone automatically**  
- Reduces manual review analysis time from **~30 minutes to a few seconds**  
- Converts **unstructured text into structured insights**  
- Enables faster and more reliable decision-making  

---

## 🏗️ System Architecture

1. User inputs phone name  
2. Scraper fetches reviews from Flipkart  
3. Text preprocessing cleans and structures the data  
4. Sentiment analysis identifies feature-specific opinions  
5. Scores are computed for each feature  
6. Phones are ranked and recommended  

---

## 📸 Example Output


---

## 🧪 How to Run

```bash
git clone https://github.com/nsjss/flipkart-phone-analysis.git
cd flipkart-phone-analysis

pip install -r requirements.txt

python main.py


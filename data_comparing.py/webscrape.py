import csv
from bs4 import BeautifulSoup
import requests

# Flipkart search URL base
base_url = "https://www.flipkart.com/search?q=MacBook%20Air%20m2&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"
header = {
    'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1.1 Safari/605.1.15",
    'Accept-Language': 'en-US,en;q=0.5'
}
base_url2="https://www.dxomark.com/smartphones/"
# Function to dynamically generate Flipkart URL
def create_flipkart_url(base_url, search_query):
    encoded_query = search_query.replace(" ", "+")
    return base_url.replace("MacBook%20Air%20m2", encoded_query)

# Input from the user
user_query = input("Enter the phone name: ")

# Generate URL dynamically
url = create_flipkart_url(base_url, user_query)
webpagee=requests.get(base_url2,headers=header)
print(webpagee)
print("Generated URL:", url)

# Request the page
webpage = requests.get(url, headers=header)
soup = BeautifulSoup(webpage.content, 'html.parser')

# Locate product links
links = soup.find_all('a', attrs={'class': 'CGtC98'})
link = []
rev_link = []

for i in range(len(links)):
    link_url = links[i].get('href')
    if link_url:
        link.append('https://flipkart.com' + link_url)
        new_webpage = requests.get(link[-1], headers=header)
        newsoup = BeautifulSoup(new_webpage.content, 'html.parser')
        
        name = newsoup.find('span', attrs={'class': "VU-ZEz"})
        review_section = newsoup.find('div', attrs={'class': 'DOjaWF gdgoEp'})
        if review_section:
            review_div = review_section.find('div', attrs={'class': '_23J90q RcXBOT'})
            if review_div:
                review = review_div.find_parent('a')
                if review:
                    rev_link.append("https://flipkart.com" + review.get('href'))
        
        if name and user_query.lower() in name.text.lower():
            print(name.text)

if not rev_link:
    print("No review links found.")
    exit()

print("First Review Link:", rev_link[0])

# Create CSV file to save reviews
output_file = "flipkart_reviews.csv"
unique_reviews = set()  # Set to store unique reviews

with open(output_file, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Phone Name", "Review"])

    # Pagination for reviews
    page_no = 1

    while True:
        page_link = f"{rev_link[0]}&page={page_no}"
        response = requests.get(page_link, headers=header)

        if response.status_code != 200:
            print(f"Failed to fetch page {page_no}: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        rev2_links = soup.find_all('div', attrs={'class': 'ZmyHeo'})

        if not rev2_links:
            print(f"No reviews found on page {page_no}. Stopping.")
            break

        for rev2_link in rev2_links:
            inner_div = rev2_link.find('div', attrs={'class': ''})
            if inner_div and inner_div.text.strip():
                review_text = inner_div.text.strip()

                if review_text not in unique_reviews:
                    unique_reviews.add(review_text)
                    writer.writerow([user_query, review_text])

        print(f"Page {page_no} processed.")
        page_no += 1

print(f"Reviews saved to {output_file}.")

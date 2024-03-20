import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Function to scrape Wikipedia's website which is under Creative Commons Attribution/Share-Alike 4.0 International License
# https://creativecommons.org/licenses/by-sa/4.0/deed.en
def scrape_images(player_name):

    player_name = player_name.replace(" ", "_")

    # Construct Wikipedia URL
    url = f"https://en.wikipedia.org/wiki/{player_name}"

    # Send a GET request
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')

    save_dir = f"static/wikipedia_images/{player_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img in images:
        img_url = urljoin(url, img['src'])
        img_name = img_url.split("/")[-1]

        # Download and save the image
        img_data = requests.get(img_url).content
        with open(os.path.join(save_dir, img_name), 'wb') as file:
            file.write(img_data)

        print(f"Downloaded {img_name}")
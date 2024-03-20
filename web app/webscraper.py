import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_images(player_name):

    # Construct Wikipedia URL
    player_name_formatted = player_name.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{player_name_formatted}"

    # Send a GET request
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')

    save_dir = f"./static/wikipedia_images/{player_name_formatted}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize a hashtable for the player's images
    player_images = {}

    for img in images:
        img_url = urljoin(url, img['src'])
        img_name = img_url.split("/")[-1]
        save_path = os.path.join(save_dir, img_name)

        # Download and save the image
        img_data = requests.get(img_url).content
        with open(save_path, 'wb') as file:
            file.write(img_data)

        print(f"Downloaded {img_name}")

        # Add the save path to the hashtable
        if player_name not in player_images.keys():
            player_images[player_name] = [save_path]
        else:
            player_images[player_name].append(save_path)


    return player_images

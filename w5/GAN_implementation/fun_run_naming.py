import requests
from bs4 import BeautifulSoup
import random
import numpy as np

# use requests to download the page

WIKIPEDIA_CHARACTER_NAMES = "https://en.wikipedia.org/wiki/List_of_Star_Wars_characters"
response = requests.get(WIKIPEDIA_CHARACTER_NAMES)

# load the page into BeautifulSoup
soup = BeautifulSoup(response.text, features="html.parser")

# load all h3, then get the text
h3s = soup.find_all("h3")
print(len(h3s))
name_list = []
for h3 in h3s[:-9]:
    # print(h3.get_text().strip().replace("[edit]", ""))
    name_list.append(h3.get_text().strip().replace("[edit]", ""))

# print(name_list)
name_list = [name.replace("\"", "")for name in name_list]

# split all names on spaces
name_list = [name.split(" ") for name in name_list]
# get first names, last names and single names each as their own list
first_names = [name[0] for name in name_list if len(name) > 1]
last_names = [name[-1] for name in name_list if len(name) > 1]
single_names = [name[0] for name in name_list if len(name) == 1]

def get_random_name():
    return random.choice(name_list) + " " + random.choice(name_list)


def get_random_name():
    if np.random.rand() < 0.7:
        return np.random.choice(first_names) + " " + np.random.choice(last_names)
    else:
        return np.random.choice(single_names)

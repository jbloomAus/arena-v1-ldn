import requests
from bs4 import BeautifulSoup
import random

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

# split all names on spaces
name_list = [name.split(" ") for name in name_list]
# flatten list
name_list = [name for names in name_list for name in names]
# remove apostrophes
name_list = [name.replace("\"", "") for name in name_list]
# remove empty strings or strings with only punctuation
name_list = [name for name in name_list if name.isalpha()]


def get_random_name_pair(name_list):
    name1 = random.choice(name_list)
    name2 = random.choice(name_list)
    while name1 == name2:
        name2 = random.choice(name_list)
    return name1 + " " + name2


def get_random_name():
    return random.choice(name_list) + " " + random.choice(name_list)
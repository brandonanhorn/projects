import requests
from bs4 import BeautifulSoup

## Main draws
r = requests.get("https://www.lottomaxnumbers.com/numbers/2013")
soup = BeautifulSoup(r.content, "lxml")

table = soup.find_all("li",{"class":"ballGen ball pngfix"})

numbers = []
for i in table:
    numbers.append(i.text)

numbers.reverse()
numbers

## Max millions
bonus = []
z = requests.get("https://www.lottomaxnumbers.com/numbers/lotto-max-result-12-21-2012")
soup = BeautifulSoup(z.content, "lxml")

table = soup.find_all("li",{"class":"ballGen"})

for y in table:
    bonus.append(y.text)

bonus = bonus[8:]
bonus.reverse()
bonus
len(bonus)/7/2

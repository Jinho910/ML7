import requests
from bs4 import BeautifulSoup

URL = "https://www.iban.com/currency-codes"
html = requests.get(URL)
soup = BeautifulSoup(html.text, 'html.parser')
ctry = soup.find_all('tr')
ctry = ctry[1:]

cur = []
for i, tr in enumerate(ctry):
    info = {}
    elems = tr.find_all('td')
    info['Ctry'] = elems[0].text.lower().capitalize()
    info['Code'] = elems[2].text
    cur.append(info)

print("Hello, choose a number")
for i, ctry in enumerate(cur):
    print(f"# {i} {ctry['Ctry']}")

while True:
    try:
        num = int(input("#: "))
    except:
        print("That was not a number")
        continue

    if num > len(cur) - 1:
        print("Choose a number from the list")
        continue
    else:
        print(f"You chose {cur[num]['Ctry']} ")
        print(f"The country code is {cur[num]['Code']}")
        break

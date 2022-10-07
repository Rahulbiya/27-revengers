import requests
import json
import textwrap

import matplotlib as mt
import pandas

url = "https://bing-news-search1.p.rapidapi.com/news/search"

querystring = {"q":"agriculture India","freshness":"Day","textFormat":"Raw","safeSearch":"Off"}

headers = {
	"X-BingApis-SDK": "true",
	"X-RapidAPI-Key": "1f62d360a0mshab4da6de118667bp13703cjsn05c591a9a491",
	"X-RapidAPI-Host": "bing-news-search1.p.rapidapi.com"
}

response = requests.request("GET", url, headers=headers, params=querystring)

response1=response.json()

with open("news.json","w") as file :
    json.dump(response1,file)

print(response1)
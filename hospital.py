import json
import textwrap

import requests
import matplotlib as mt
import pandas

url = "https://trueway-places.p.rapidapi.com/FindPlacesNearby"

querystring = {"location":"19.076090,72.877426","type":"hospital","radius":"9000","language":"en"}

headers = {
	"X-RapidAPI-Host": "trueway-places.p.rapidapi.com",
	"X-RapidAPI-Key": "1f62d360a0mshab4da6de118667bp13703cjsn05c591a9a491"
}

response = requests.request("GET", url, headers=headers, params=querystring)

response1=response.json()

with open("hospital.json","w") as file :
    json.dump(response1,file)

print(response1)

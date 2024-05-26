import requests

def fetch_data(url):
    response = requests.get(url)
    data = response.json()
    return data
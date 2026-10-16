import requests
import httpx
from sqlalchemy.orm import Session as DatabaseSession
from other import Client

s = requests.Session()
s.get("https://untracked.example.com")
with requests.Session() as s:
    s.get("https://with.example.com")
lower = requests.session()
client = httpx.Client()
client.get("https://client.example.com")

async def load():
    async with httpx.AsyncClient() as c:
        await c.get("https://async.example.com")

def Session():
    return None
Session()
class Session:
    pass
Session()
DatabaseSession()
Client()
# requests.Session()
text = "httpx.Client()"

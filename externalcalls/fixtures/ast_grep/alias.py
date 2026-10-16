import requests
import requests as r
import requests as r2
import urllib.request as ur
from requests import get as g
from requests import get
from urllib import request
from urllib.request import urlopen
from requests import (
    post as p,
    put,
)
from other import get as unrelated

def nested():
    import httpx as inner
    return inner.get("https://nested.example.com")

requests.get("https://direct.example.com")
r.get("https://alias.example.com")
r2.post("https://second.example.com")
ur.urlopen("https://urllib.example.com")
g("https://named.example.com")
get("https://bare.example.com")
request.urlopen("https://submodule.example.com")
urlopen("https://urlopen.example.com")
p("https://post.example.com")
put("https://put.example.com")
# import requests as commented
text = "import requests as instring"

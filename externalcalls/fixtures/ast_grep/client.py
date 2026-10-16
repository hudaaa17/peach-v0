import requests
import httpx
import urllib.request


def get(url):
    # a repo-defined function that happens to be called `get`
    return url


def fetch_all(base):
    a = requests.get("https://api.example.com/a")
    b = requests.post(base + "/b", json={})
    c = requests.request("GET", "https://api.example.com/c")
    d = httpx.get("https://api.example.com/d")
    e = urllib.request.urlopen("https://api.example.com/e")
    o = requests.options("https://api.example.com/o")
    h = httpx.stream("GET", "https://api.example.com/h")
    f = get("https://not-http.example.com")
    sess = requests.Session()
    # requests.get("https://commented.example.com")
    s = 'requests.get("https://instring.example.com")'
    t = """
    requests.get("https://docstring.example.com")
    """
    return a, b, c, d, e, f, o, h, sess, s, t
import requests
import os
from config import IMPORTED11

MODULE2 = "https://module.com/x"
MODULE3 = "https://module.com/x"
requests.get(MODULE3)  # 3
URL9 = "https://module.com/ignored"
BASE18 = "https://a.com"
BASE19 = "https://a.com"
SHARED25 = "https://a.com/x"
SHARED26 = "https://a.com/x"

def c01():
    url = "https://a.com/x"
    requests.get(url)  # 1
def c02():
    requests.get(MODULE2)  # 2
def c04():
    a = "https://a.com"
    b = a
    requests.get(b)  # 4
def c05():
    a = b
    b = a
    requests.get(a)  # 5
def c06():
    a = "https://a.com"
    b = a
    c = b
    d = c
    requests.get(d)  # 6
def c07():
    url = "https://a.com/x"
    url = "https://b.com/x"
    requests.get(url)  # 7
MULTI8 = "https://a.com/x"
MULTI8 = "https://b.com/x"
def c08():
    requests.get(MULTI8)  # 8
def c09():
    URL9 = "https://local.com/x"
    requests.get(URL9)  # 9
def c10(url):
    requests.get(url)  # 10
def c11():
    requests.get(IMPORTED11)  # 11
def c12():
    if condition:
        url = "https://a.com/x"
    requests.get(url)  # 12
def c13():
    requests.get(url)  # 13
    url = "https://a.com/x"
def c14():
    url += "x"
    requests.get(url)  # 14
def c15():
    url, other = values
    requests.get(url)  # 15
def c16():
    for url in values:
        pass
    requests.get(url)  # 16
def c18():
    requests.get(BASE18 + "/x")  # 18
def c19():
    requests.get(f"{BASE19}/users")  # 19
def c20():
    BASE = "https://a.com"
    url = f"{BASE}/x"
    requests.get(url)  # 20
def c21():
    url = os.environ["X"]
    requests.get(url)  # 21
def c22():
    BASE = "https://a."
    requests.get(BASE + env)  # 22
def c23():
    requests.get(NEVER23)  # 23
def c24_outer():
    OUTER24 = "https://a.com/x"
    def inner():
        requests.get(OUTER24)  # 24
def unrelated25():
    SHARED25 = "https://b.com/x"
def c25():
    requests.get(SHARED25)  # 25
def c26a():
    requests.get(SHARED26)  # 26
def c26b():
    requests.get(SHARED26)  # 26
def c27():
    url = "https://unicode.com/x"
    "héllo 😀"; requests.get(url)  # 27
def credentials():
    url = "https://user:pw@a.com:8080/x"
    requests.get(url)  # 30

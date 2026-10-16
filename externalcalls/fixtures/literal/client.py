import requests
import httpx
import os

# Case numbers are test identifiers, not runtime code.
requests.get("https://api.stripe.com/v1/charges")  # 1
requests.request("GET", "https://a.com/x")  # 2
requests.request("GET", url="https://a.com/x")  # 3
httpx.stream("GET", "https://a.com/x")  # 4
requests.get(f"https://a.com/{i}")  # 5
requests.get(f"https://a.{env}/x")  # 6
requests.get("https://a.com/" + i)  # 7
requests.get("https://a." + env)  # 8
requests.get("/users")  # 9
requests.get("https://a.com/\n")  # 10
requests.get(b"https://a.com")  # 11
requests.get(url)  # 12
requests.get(BASE + "/x")  # 13
requests.get(f"{BASE}/x")  # 14
requests.get(self.url)  # 15
requests.get(os.environ["X"]); requests.get(os.getenv("X")); requests.get(os.environ.get("X"))  # 16
requests.get(get_url())  # 17
requests.get()  # 18
requests.get(*args)  # 19
requests.get("https://one.com/x"); requests.get("https://two.com/y")  # 30
"héllo 😀"; requests.get("https://unicode.com/x")  # 31

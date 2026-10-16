import g from "got";
import k from "ky";
g("https://a.example.com");
k("https://b.example.com");
g.get("https://a.example.com/member");
k.post("https://b.example.com/member");

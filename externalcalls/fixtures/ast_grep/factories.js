const s = axios.create({ baseURL: "https://factory.example.com" });
s.get("https://untracked.example.com");
const g = got.extend({});
g.get("https://got.example.com");
const k = ky.create({});
k.get("https://ky.example.com");
const k2 = ky.extend({});
const agent = superagent.agent();
agent.get("https://agent.example.com");
function Session() {}
Session();
class Client {}
new Client();
// axios.create({});
const text = "got.extend({})";

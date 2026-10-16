interface Config { baseURL: string }
async function load() {
  const s = await axios.create<Config>({ baseURL: "https://factory.example.com" });
  s.get("https://untracked.example.com");
  const g = got.extend<Config>({});
  const k = await ky.create({});
  const k2 = ky.extend({});
  const agent = superagent.agent();
  agent.get("https://agent.example.com");
}
// axios.create<Config>({});
const text = "got.extend({})";

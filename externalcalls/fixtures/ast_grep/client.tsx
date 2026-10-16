import axios from "axios";

export function Panel() {
  const load = async () => {
    const r = await fetch("https://api.example.com/tsx");
    return <div>{String(r.ok)}</div>;
  };
  return <button onClick={() => axios.post("https://api.example.com/tsx2")}>go</button>;
}

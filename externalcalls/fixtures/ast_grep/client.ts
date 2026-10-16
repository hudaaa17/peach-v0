import axios from "axios";

interface User { id: number }

export async function load(id: number): Promise<User> {
  const r = await axios.get<User>("https://api.example.com/users/" + id);
  const s = await fetch("https://api.example.com/ts");
  return r.data;
}

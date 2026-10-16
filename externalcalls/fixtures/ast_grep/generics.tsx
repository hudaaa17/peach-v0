interface User { id: number }
export async function variants() {
  const a = axios.get<User>("https://api.example.com/nonawait");
  const café = await fetch<User>("https://api.example.com/generic-fetch");
  const c = await axios.get("https://api.example.com/plain-await");
  return <div />;
}

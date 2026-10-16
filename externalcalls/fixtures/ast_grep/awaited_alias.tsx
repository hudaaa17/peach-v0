import ax from "axios";
import { get } from "axios";
await ax.get<User>("https://a.example.com");
await get<T>("https://b.example.com");

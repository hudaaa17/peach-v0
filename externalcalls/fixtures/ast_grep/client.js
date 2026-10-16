const axios = require("axios");
const got = require("got");
const ky = require("ky");
const http = require("http");
const https = require("https");
const superagent = require("superagent");

function get(url) { return url; }   // a repo-defined `get`

async function main(base) {
  const a = await fetch("https://api.example.com/a");
  const b = await axios.get("https://api.example.com/b");
  const c = await axios({ url: "https://api.example.com/c" });
  const d = await got("https://api.example.com/d");
  const e = http.request("http://api.example.com/e");
  const f = https.request("https://api.example.com/f");
  const g = await ky.post("https://api.example.com/g");
  const h = await superagent.get("https://api.example.com/h");
  const j = http.get("http://api.example.com/j");
  const k = https.get("https://api.example.com/k");
  const l = await window.fetch("https://api.example.com/l");
  const m = await globalThis.fetch("https://api.example.com/m");
  const i = get("https://not-http.example.com");
  // fetch("https://commented.example.com");
  /* axios.get("https://blockcomment.example.com") */
  const s = 'fetch("https://instring.example.com")';
  const t = `fetch("https://template.example.com")`;
  return [a, b, c, d, e, f, g, h, i, j, k, l, m, s, t];
}
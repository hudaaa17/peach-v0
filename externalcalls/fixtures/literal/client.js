import axios from 'axios';
axios.get("https://a.com/x", {a: 1}); // 20
axios({url: "https://a.com/y", method: "get"}); // 21
axios("https://a.com/z"); // 22
axios({url}); // 23
fetch(`https://a.com/${i}`); // 24
fetch("https://a.com/" + i); // 25
fetch(process.env.X); // 26
fetch("https://user:pw@a.com:8080/x"); // 29

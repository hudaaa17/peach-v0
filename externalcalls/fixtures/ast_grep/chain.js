const axios = require("axios");

axios
  .get("https://api.example.com/chain")
  .then((r) => r.data)
  .catch(() => null);
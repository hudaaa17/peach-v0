import {IMPORTED11} from './config';
const {REQUIRE11} = require('./config');
const MODULE2 = "https://module.com/x";
const MODULE3 = "https://module.com/x";
fetch(MODULE3); // 3
const URL9 = "https://module.com/ignored";
const BASE18 = "https://a.com";
const BASE19 = "https://a.com";
const SHARED25 = "https://a.com/x";
const SHARED26 = "https://a.com/x";
function c01() { const url = "https://a.com/x"; fetch(url); } // 1
function c02() { fetch(MODULE2); } // 2
function c04() { const a = "https://a.com"; const b = a; fetch(b); } // 4
function c05() { const a = b; const b = a; fetch(a); } // 5
function c06() { const a = "https://a.com"; const b = a; const c = b; const d = c; fetch(d); } // 6
function c07() { let url = "https://a.com/x"; url = "https://b.com/x"; fetch(url); } // 7
var MULTI8 = "https://a.com/x";
MULTI8 = "https://b.com/x";
function c08() { fetch(MULTI8); } // 8
function c09() { const URL9 = "https://local.com/x"; fetch(URL9); } // 9
function c10(url) { fetch(url); } // 10
function c11a() { fetch(IMPORTED11); } // 11
function c11b() { fetch(REQUIRE11); } // 11
function c12() { if (condition) { const url = "https://a.com/x"; } fetch(url); } // 12
function c13() { fetch(url); const url = "https://a.com/x"; } // 13
function c14() { url += "x"; fetch(url); } // 14
function c15() { const {url} = values; fetch(url); } // 15
function c16() { for (const url of values) {} fetch(url); } // 16
function c17() { let url; fetch(url); } // 17
function c18() { fetch(BASE18 + "/x"); } // 18
function c19() { fetch(`${BASE19}/users`); } // 19
function c20() { const BASE = "https://a.com"; const url = `${BASE}/x`; fetch(url); } // 20
function c21() { const url = process.env.X; fetch(url); } // 21
function c22() { const BASE = "https://a."; fetch(BASE + env); } // 22
function c23() { fetch(NEVER23); } // 23
function outer24() { const OUTER24 = "https://a.com/x"; function inner() { fetch(OUTER24); } } // 24
function unrelated25() { const SHARED25 = "https://b.com/x"; }
function c25() { fetch(SHARED25); } // 25
function c26a() { fetch(SHARED26); } // 26
function c26b() { fetch(SHARED26); } // 26
function c27() { const url = "https://unicode.com/x"; "héllo 😀"; fetch(url); } // 27
function credentials() { const url = "https://user:pw@a.com:8080/x"; fetch(url); } // 30

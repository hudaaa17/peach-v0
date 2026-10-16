export const MODULE2: string = "https://module.com/x";
function c02() { fetch(MODULE2); } // 2
function c01() { const url: string = "https://a.com/x"; fetch((url as string)); } // 1
function c10(url: string = "https://ignored.com") { fetch(url); } // 10

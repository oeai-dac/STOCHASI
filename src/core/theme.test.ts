import { toRgb01 } from "./theme.js";

let pass = 0, fail = 0; const F: string[] = [];
function c(n: string, ok: boolean, d = "") { ok ? pass++ : (fail++, F.push(n)); console.log((ok ? "  \x1b[32m✓\x1b[0m " : "  \x1b[31m✗\x1b[0m ") + n + (d ? " — " + d : "")); }
const near = (a: number, b: number) => Math.abs(a - b) < 1e-3;
const eq = (v: [number, number, number], e: [number, number, number]) => near(v[0], e[0]) && near(v[1], e[1]) && near(v[2], e[2]);

console.log("\n\x1b[1mTheme-Farbparser\x1b[0m\n");

// Der helle Matrix-Hintergrund muss exakt dem bisher hartcodierten Renderer-Wert entsprechen
c("#f6f4f2 → [0.965,0.957,0.949] (Hell = bisheriger Renderer-Default)", eq(toRgb01("#f6f4f2"), [0.965, 0.957, 0.949]), toRgb01("#f6f4f2").map((x) => x.toFixed(3)).join(","));
c("#232020 → dunkler Hintergrund", eq(toRgb01("#232020"), [35 / 255, 32 / 255, 32 / 255]));
c("Kurzform #fff → weiß", eq(toRgb01("#fff"), [1, 1, 1]));
c("Großschreibung #FFFFFF → weiß", eq(toRgb01("#FFFFFF"), [1, 1, 1]));
c("rgb(35, 32, 32) (getComputedStyle-Form) = #232020", eq(toRgb01("rgb(35, 32, 32)"), toRgb01("#232020")));
c("rgba(246, 244, 242, 1) = #f6f4f2", eq(toRgb01("rgba(246, 244, 242, 1)"), toRgb01("#f6f4f2")));
c("führende/anhängende Leerzeichen toleriert", eq(toRgb01("  #232020  "), toRgb01("#232020")));
c("ungültige Eingabe → heller Fallback", eq(toRgb01("nonsense"), [0.965, 0.957, 0.949]));

console.log(`\n\x1b[1mErgebnis:\x1b[0m ${pass} bestanden, ${fail} fehlgeschlagen`);
if (fail) { console.log("\x1b[31mFehlgeschlagen:\x1b[0m " + F.join(", ")); process.exit(1); }

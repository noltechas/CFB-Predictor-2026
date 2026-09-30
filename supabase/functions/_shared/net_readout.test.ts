import { assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { MARGIN_C, MARGIN_LO, N_MARGIN, N_TOTAL, pmfMedian, readouts, TOTAL_C } from "./net.ts";

const at = (value: number) => value - MARGIN_LO + 1;   // margin bin index of an integer margin

Deno.test("a point mass reads as its own margin", () => {
  const p = new Float64Array(N_MARGIN);
  p[at(7)] = 1;
  assertAlmostEquals(pmfMedian(p, MARGIN_C, N_MARGIN), 7, 1e-12);
});

Deno.test("median interpolates inside the bin where the CDF crosses one half", () => {
  const p = new Float64Array(N_MARGIN);
  p[at(-3)] = 0.4;
  p[at(7)] = 0.6;
  assertAlmostEquals(pmfMedian(p, MARGIN_C, N_MARGIN), 7 - 0.5 + (0.5 - 0.4) / 0.6, 1e-12);
});

Deno.test("display readouts are medians; win probability is unchanged", () => {
  const m = new Float64Array(N_MARGIN), t = new Float64Array(N_TOTAL);
  m[at(-10)] = 0.3; m[at(3)] = 0.7;          // skewed: mean 0.9, median inside the +3 bin
  t[40] = 1;
  const r = readouts(m, t);
  assertAlmostEquals(r.wp, 0.7, 1e-12);
  assertAlmostEquals(r.em, 0.3 * -10 + 0.7 * 3, 1e-12);
  assertAlmostEquals(r.mm, 3 - 0.5 + (0.5 - 0.3) / 0.7, 1e-12);
  assertEquals(r.mt, TOTAL_C[40]);
});

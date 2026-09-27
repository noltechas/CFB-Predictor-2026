// Proves the production path: fetch weights from private Storage, run the
// hand-written forward pass, and reproduce a snap the Python ensemble scored.
// No wasm, no CDN — cold start is a 4.5 MB fetch and nothing else.
import { Meta, Net, readouts } from "../_shared/net.ts";

const URL_ = Deno.env.get("SUPABASE_URL")!;
const KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

let net: Net | null = null;
let loadMs = 0;

async function getNet(): Promise<Net> {
  if (net) return net;
  const t0 = performance.now();
  const head = { Authorization: `Bearer ${KEY}`, apikey: KEY };
  const [wRes, mRes] = await Promise.all([
    fetch(`${URL_}/storage/v1/object/edge-model/weights.bin`, { headers: head }),
    fetch(`${URL_}/storage/v1/object/edge-model/weights.json`, { headers: head }),
  ]);
  if (!wRes.ok || !mRes.ok) throw new Error(`weights fetch ${wRes.status}/${mRes.status}`);
  const meta = await mRes.json() as Meta;
  net = new Net(await wRes.arrayBuffer(), meta);
  loadMs = performance.now() - t0;
  return net;
}

Deno.serve(async (req) => {
  try {
    const n = await getNet();
    const body = await req.json().catch(() => ({}));
    const { state, fr, al, lead, pts, pm, pt } = body;
    if (!state) {
      return Response.json({ ok: true, warm: loadMs === 0, load_ms: +loadMs.toFixed(1),
                             members: n.meta.n_members, n_state: n.meta.n_state,
                             version: n.meta.model_version });
    }
    const t1 = performance.now();
    const out = n.run(Float64Array.from(state), fr, al, lead, pts,
                      pm.map((r: number[]) => Float32Array.from(r)),
                      pt.map((r: number[]) => Float32Array.from(r)));
    const r = readouts(out.margin, out.total);
    return Response.json({ ok: true, load_ms: +loadMs.toFixed(1),
                           run_ms: +(performance.now() - t1).toFixed(2), ...r });
  } catch (e) {
    return Response.json({ ok: false, error: String(e) }, { status: 500 });
  }
});

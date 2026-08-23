// tiny fetch wrapper, same origin (the python service serves this page)

export async function fetchHealth() {
  const r = await fetch("/health");
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function recommend(body, signal) {
  // the service has a 40 s deadline; give the browser a little more before giving up
  const timer = new AbortController();
  const timeout = setTimeout(() => timer.abort(), 50_000);
  const combined = signal ? anySignal([signal, timer.signal]) : timer.signal;
  let r;
  try {
    r = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: combined,
    });
  } catch (e) {
    if (timer.signal.aborted && !(signal && signal.aborted)) {
      throw new Error("the request took longer than 50 s, try again or switch off the llm rerank");
    }
    throw e;
  } finally {
    clearTimeout(timeout);
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = data?.error?.message || `request failed (${r.status})`;
    throw new Error(msg);
  }
  return data;
}

function anySignal(signals) {
  if (typeof AbortSignal.any === "function") return AbortSignal.any(signals);
  const c = new AbortController();
  for (const s of signals) {
    if (s.aborted) c.abort(s.reason);
    else s.addEventListener("abort", () => c.abort(s.reason), { once: true });
  }
  return c.signal;
}

// only ever render http(s) urls coming from the catalog
export function safeUrl(u) {
  if (typeof u !== "string") return null;
  try {
    const parsed = new URL(u);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

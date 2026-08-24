import { useEffect, useState } from "react";
import { fetchHealth, recommend, safeUrl } from "./api.js";

const EXAMPLES = [
  "I need an outfit to go to the beach this summer",
  "warm waterproof boots for hiking in the snow, under $80",
  "what should my husband wear to an outdoor wedding in june",
  "something cozy for working from home in winter",
  "a gift for my 6 year old daughter who loves unicorns",
  "smart casual outfit for a job interview at a startup",
];

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("theme") || "light";
    } catch {
      return "light";
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* private mode etc, fine */
    }
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "light" ? "dark" : "light"))];
}

function StatusPill({ health, error }) {
  if (error) return <span className="pill pill-bad">api offline</span>;
  if (!health) return <span className="pill">checking...</span>;
  const llm = health.llm?.model ? `llm: ${health.llm.model}` : "no llm (regex planner)";
  const idx = health.index ? `${health.index.rows.toLocaleString()} items` : "no index";
  return (
    <span className={"pill " + (health.index_loaded ? "pill-ok" : "pill-bad")}>
      {idx} · {llm}
    </span>
  );
}

function Stars({ rating, count }) {
  if (!count) return <span className="muted">no ratings yet</span>;
  if (rating == null) return <span className="muted">rating unavailable ({count.toLocaleString()} ratings)</span>;
  return (
    <span title={`${rating} average from ${count} ratings`}>
      {"★".repeat(Math.round(rating))}
      {"☆".repeat(5 - Math.round(rating))} <span className="muted">({count.toLocaleString()})</span>
    </span>
  );
}

function ProductCard({ item }) {
  const img = safeUrl(item.image_url);
  const url = safeUrl(item.url);
  return (
    <article className="card">
      <div className="card-img">
        {img ? <img src={img} alt="" /> : <div className="noimg">no image</div>}
      </div>
      <div className="card-body">
        <div className="card-title">
          <span className="rank">#{item.rank}</span>{" "}
          {url ? (
            <a href={url} target="_blank" rel="noopener noreferrer">
              {item.title}
            </a>
          ) : (
            item.title
          )}
        </div>
        <div className="card-meta">
          <span className="price">
            {item.price_known ? `$${item.price.toFixed(2)}` : "price n/a"}
          </span>
          <Stars rating={item.average_rating} count={item.rating_number} />
          {item.store ? <span className="muted">· {item.store}</span> : null}
        </div>
        <p className="reason">{item.reason}</p>
        {item.matched_keywords?.length ? (
          <div className="tags">
            {item.matched_keywords.map((k) => (
              <span className="tag" key={k}>
                {k}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function PlanSummary({ plan, llmInfo }) {
  const bits = [];
  if (plan.audience) bits.push(`audience: ${plan.audience}`);
  if (plan.occasion) bits.push(`occasion: ${plan.occasion}`);
  if (plan.season) bits.push(`season: ${plan.season}`);
  if (plan.brand) bits.push(`brand: ${plan.brand}`);
  if (plan.budget_max != null)
    bits.push(`budget: ${plan.budget_min != null ? `$${plan.budget_min} - ` : "up to "}$${plan.budget_max} (${plan.budget_scope})`);
  return (
    <div className="plan">
      <div>
        <strong>Understood as:</strong> {plan.intent}
      </div>
      {bits.length ? <div className="muted">{bits.join(" · ")}</div> : null}
      <div className="muted small">
        planner: {llmInfo.planner_used}
        {llmInfo.plan_cache_hit ? " (cached)" : ""}
        {llmInfo.rerank_used ? " · reranked by llm" : " · retrieval order"}
        {llmInfo.calls ? ` · ${llmInfo.calls} llm call${llmInfo.calls > 1 ? "s" : ""}, ${(llmInfo.input_tokens + llmInfo.output_tokens).toLocaleString()} tokens` : ""}
      </div>
    </div>
  );
}

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(false);
  const [query, setQuery] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [minPrice, setMinPrice] = useState("");
  const [audience, setAudience] = useState("");
  const [k, setK] = useState(4);
  const [priceMode, setPriceMode] = useState("auto"); // auto | strict | relaxed
  const [useLlm, setUseLlm] = useState(true);
  // quick: bounded planner + deterministic reasons, sub-second. full: llm rerank, slower
  const [mode, setMode] = useState("quick");
  const rerank = mode === "full";
  const [abort, setAbort] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    let alive = true;
    const tick = () =>
      fetchHealth()
        .then((h) => alive && (setHealth(h), setHealthError(false)))
        .catch(() => alive && setHealthError(true));
    tick();
    const id = setInterval(tick, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  async function run(q) {
    const text = (q ?? query).trim();
    if (!text) return;
    setQuery(text);
    setLoading(true);
    setError(null);
    if (abort) abort.abort(); // a new request supersedes the one in flight
    const controller = new AbortController();
    setAbort(controller);
    try {
      const body = { query: text, k: Number(k) || 4, use_llm: useLlm, rerank };
      if (maxPrice !== "" && !Number.isNaN(Number(maxPrice))) body.max_price = Number(maxPrice);
      if (minPrice !== "" && !Number.isNaN(Number(minPrice))) body.min_price = Number(minPrice);
      if (audience) body.audience = audience;
      if (priceMode !== "auto") body.include_unpriced = priceMode === "relaxed";
      const data = await recommend(body, controller.signal);
      if (!controller.signal.aborted) setResult(data);
    } catch (e) {
      if (e.name === "AbortError") return; // superseded or cancelled: keep what is on screen
      setError(e.message);
      setResult(null);
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div>
            <div className="title">Fashion Stylist</div>
            <div className="subtitle">ask for an outfit like you would ask a friend</div>
          </div>
          <div className="header-right">
            <a className="ghost-link" href="/overview">how it works</a>
            <StatusPill health={health} error={healthError} />
            <button className="ghost" onClick={toggleTheme} aria-label="toggle theme">
              {theme === "light" ? "dark mode" : "light mode"}
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        <section className="search">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              run();
            }}
          >
            <div className="search-row">
              <input
                className="search-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. I need an outfit to go to the beach this summer"
                maxLength={500}
              />
              <button className="primary" type="submit" disabled={loading}>
                {loading ? "thinking..." : "find it"}
              </button>
            </div>
            <div className="options">
              <label>
                max $ per item
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={maxPrice}
                  onChange={(e) => setMaxPrice(e.target.value)}
                  placeholder="any"
                />
              </label>
              <label>
                min $ per item
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={minPrice}
                  onChange={(e) => setMinPrice(e.target.value)}
                  placeholder="any"
                />
              </label>
              <label>
                items per slot
                <input type="number" min="1" max="10" value={k} onChange={(e) => setK(e.target.value)} />
              </label>
              <label>
                for
                <select value={audience} onChange={(e) => setAudience(e.target.value)}>
                  <option value="">anyone (let the planner decide)</option>
                  <option value="women">women</option>
                  <option value="men">men</option>
                  <option value="girls">girls</option>
                  <option value="boys">boys</option>
                  <option value="baby">baby</option>
                  <option value="unisex">unisex</option>
                </select>
              </label>
              <label>
                unpriced items
                <select value={priceMode} onChange={(e) => setPriceMode(e.target.value)}>
                  <option value="auto">auto (strict for max $, allowed for budgets in the text)</option>
                  <option value="strict">strict: known prices only when a budget applies</option>
                  <option value="relaxed">relaxed: allow unknown prices, flagged</option>
                </select>
              </label>
              <label className="check">
                <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
                use llm
              </label>
              <div className="seg" role="radiogroup" aria-label="response mode">
                <button
                  type="button"
                  className={mode === "quick" ? "seg-btn on" : "seg-btn"}
                  aria-pressed={mode === "quick"}
                  onClick={() => setMode("quick")}
                  title="bounded planner wait, cached plans, template reasons"
                >
                  quick <span className="seg-note">~0.1-0.5 s</span>
                </button>
                <button
                  type="button"
                  className={mode === "full" ? "seg-btn on" : "seg-btn"}
                  aria-pressed={mode === "full"}
                  onClick={() => setMode("full")}
                  disabled={!useLlm}
                  title="one llm rerank call per slot, llm-written reasons"
                >
                  full rerank <span className="seg-note">~1-2 s</span>
                </button>
              </div>
            </div>
          </form>
          <div className="chips">
            {EXAMPLES.map((ex) => (
              <button key={ex} className="chip" onClick={() => run(ex)} disabled={loading}>
                {ex}
              </button>
            ))}
          </div>
        </section>

        {error ? <div className="error">{error}</div> : null}

        {result ? (
          <section className="results">
            <PlanSummary plan={result.plan} llmInfo={result.llm_info} />
            {result.note ? <p className="note">{result.note}</p> : null}
            {result.slots.map((slot, i) => (
              <div className="slot" key={`${i}-${slot.name}`}>
                <h2>
                  {slot.name} <span className="muted small">searched: {slot.search_query}</span>
                </h2>
                {slot.items.length === 0 ? (
                  <p className="muted">nothing matched the constraints for this slot</p>
                ) : (
                  <div className="grid">
                    {slot.items.map((item) => (
                      <ProductCard key={item.row_id} item={item} />
                    ))}
                  </div>
                )}
              </div>
            ))}
            {result.warnings?.length ? (
              <details className="warnings">
                <summary>{result.warnings.length} note(s) from the service</summary>
                <ul>
                  {result.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </details>
            ) : null}
            <div className="muted small footer">
              {Object.entries(result.timings)
                .map(([key, v]) => `${key.replace("_ms", "")} ${v} ms`)
                .join(" · ")}{" "}
              · index: {result.index_info.rows.toLocaleString()} items ({result.index_info.sampling})
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}

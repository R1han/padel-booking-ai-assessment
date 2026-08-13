import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// The shipped slot grid. There are no 11:00-14:00 slots anywhere in the dataset, so the
// rail draws that break instead of implying the courts run continuously.
const MORNING = ["06:00", "07:00", "08:00", "09:00", "10:00"];
const REST = ["15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"];

const SESSION = `web-${Math.random().toString(36).slice(2, 10)}`;

const STARTERS = [
  { text: "Any indoor court free at Al Quoz tomorrow evening?" },
  { text: "What happens if it rains during my outdoor booking?" },
  { text: "ابغى ملعب داخلي في العين", lang: "ar" },
  { text: "Which branch is cheapest in the evening?" },
];

const isArabic = (text) => /[؀-ۿ]/.test(text);

function useNow(active) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

/** The hour rail: one tick per bookable hour, with the midday break drawn as a break. */
function HourRail({ slots, onPick, hold }) {
  const first = slots[0];
  const free = useMemo(() => {
    const map = new Map();
    slots.forEach((slot) => map.set(slot.start_time, slot));
    return map;
  }, [slots]);

  const tick = (time) => {
    const slot = free.get(time);
    const held = slot && hold?.slot_ids.includes(slot.id);
    return (
      <button
        key={time}
        type="button"
        className={`hour${slot ? " free" : ""}${held ? " held" : ""}`}
        disabled={!slot}
        onClick={() => slot && onPick(slot)}
        title={
          held
            ? `${time} · ${slot.court_code} · held for you`
            : slot
              ? `${time} · ${slot.court_code} · ${slot.price_aed} AED`
              : `${time} · taken`
        }
      >
        {time.slice(0, 2)}
      </button>
    );
  };

  return (
    <div className="rail">
      <div className="rail-head">
        <b>{first.branch_name}</b>
        <span>{first.date}</span>
        <span>{slots.length} free</span>
      </div>
      <div className="hours">
        {MORNING.map(tick)}
        <div className="hour-gap" title="No courts are scheduled 11:00-14:00" />
        {REST.map(tick)}
      </div>
      <div className="rail-scale" aria-hidden="true">
        {MORNING.map((t) => <span key={t}>{t.slice(0, 2)}</span>)}
        <span className="gap-label">·</span>
        {REST.map((t) => <span key={t}>{t.slice(0, 2)}</span>)}
      </div>
    </div>
  );
}

function SlotChip({ slot, hold, now }) {
  const held = hold && hold.slot_ids.includes(slot.id);
  const left = held ? Math.max(0, hold.expires_at * 1000 - now) : 0;
  const pct = held ? (left / (hold.ttl * 1000)) * 100 : 0;
  return (
    <div className={`slot${held ? " held" : ""}`}>
      <span className="court">{slot.court_code}</span>
      <span className="when">{slot.start_time}</span>
      <span className="price">
        {held ? `held ${Math.ceil(left / 1000)}s` : `${slot.price_aed} AED`}
      </span>
      {held && <div className="drain" style={{ width: `${pct}%` }} />}
    </div>
  );
}

function Trace({ trace }) {
  if (!trace) return null;
  const slow = trace.ttft_ms && trace.ttft_ms > 2000;
  return (
    <div className="trace">
      <span className={slow ? "warn" : undefined}>
        first token {trace.ttft_ms ?? "—"}ms
      </span>
      <span>total {trace.latency_ms}ms</span>
      <span>
        {trace.input_tokens}/{trace.output_tokens} tok
      </span>
      <span>${trace.cost_usd?.toFixed(4)}</span>
      <span>{trace.steps?.map((s) => s.step).join(" → ")}</span>
    </div>
  );
}

export default function App() {
  const [turns, setTurns] = useState([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [hold, setHold] = useState(null);
  const endRef = useRef(null);
  const now = useNow(Boolean(hold));

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, busy]);

  useEffect(() => {
    if (hold && hold.expires_at * 1000 < now) setHold(null);
  }, [hold, now]);

  const send = useCallback(
    async (message) => {
      if (!message.trim() || busy) return;
      setBusy(true);
      setDraft("");
      setTurns((prev) => [
        ...prev,
        { who: "you", text: message },
        { who: "assistant", text: "", tools: [], streaming: true },
      ]);

      const patch = (fields) =>
        setTurns((prev) => {
          const next = [...prev];
          next[next.length - 1] = { ...next[next.length - 1], ...fields };
          return next;
        });

      try {
        const response = await fetch("/api/v1/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, session_id: SESSION }),
        });
        if (!response.ok) throw new Error(`Server returned ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let text = "";
        const tools = [];

        // SSE frames are separated by a blank line; a frame can straddle two chunks.
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            const line = frame.split("\n").find((l) => l.startsWith("data: "));
            if (!line) continue;
            const payload = JSON.parse(line.slice(6));
            if (payload.text !== undefined) {
              text += payload.text;
              patch({ text });
            } else if (payload.name) {
              tools.push(payload.name);
              patch({ tools: [...tools] });
            } else if (payload.retrieved_ids) {
              patch({ trace: payload, slots: payload.slots ?? [], streaming: false });
            } else if (payload.message) {
              patch({ error: payload.message, streaming: false });
            }
          }
        }
        patch({ streaming: false });
      } catch (error) {
        patch({ error: `Could not reach the assistant. ${error.message}`, streaming: false });
      } finally {
        setBusy(false);
      }
    },
    [busy],
  );

  const holdSlot = useCallback(async (slot) => {
    const response = await fetch("/api/v1/holds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slot_ids: [slot.id],
        duration_min: slot.duration_min ?? 60,
        session_id: SESSION,
      }),
    });
    const body = await response.json();
    if (response.ok) {
      setHold({ ...body, ttl: body.expires_at - Math.floor(Date.now() / 1000) });
      setDraft(`Book ${slot.court_code} at ${slot.start_time} on ${slot.date}`);
    }
  }, []);

  return (
    <div className="app">
      <header className="masthead">
        <div className="wordmark">
          Baseline <span>Padel</span>
        </div>
        <div className="today">8 branches · UAE</div>
      </header>

      <main className="transcript">
        {turns.length === 0 && (
          <div className="opener">
            <p>
              Ask about branches, courts, coaches, classes or prices, in English or Arabic,
              and book a court in the conversation.
            </p>
            <div className="starters">
              {STARTERS.map((s) => (
                <button
                  key={s.text}
                  type="button"
                  className="starter"
                  lang={s.lang}
                  dir={s.lang === "ar" ? "rtl" : undefined}
                  onClick={() => send(s.text)}
                >
                  {s.text}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, i) => (
          <article
            key={i}
            className={`turn ${turn.who}`}
            dir={isArabic(turn.text) ? "rtl" : "ltr"}
          >
            <div className="who">{turn.who === "you" ? "You" : "Baseline"}</div>
            {turn.tools?.length > 0 && (
              <div className="tools">
                {turn.tools.map((name, j) => (
                  <span className="tool" key={`${name}-${j}`}>
                    {name}
                  </span>
                ))}
              </div>
            )}
            <div className={`say${turn.streaming ? " pending" : ""}`}>{turn.text}</div>
            {turn.error && <div className="fail">{turn.error}</div>}
            {turn.slots?.length > 0 && (
              <>
                <HourRail slots={turn.slots} onPick={holdSlot} hold={hold} />
                <div className="slots">
                  {turn.slots.map((slot) => (
                    <SlotChip key={slot.id} slot={slot} hold={hold} now={now} />
                  ))}
                </div>
              </>
            )}
            <Trace trace={turn.trace} />
          </article>
        ))}
        <div ref={endRef} />
      </main>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          send(draft);
        }}
      >
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Ask about courts, coaches or prices…"
          dir={isArabic(draft) ? "rtl" : "ltr"}
          aria-label="Message"
        />
        <button type="submit" disabled={busy || !draft.trim()}>
          {busy ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}

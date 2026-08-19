"use client";

import { useMemo } from "react";
import { EXAMPLE_QUESTIONS } from "@/lib/constants";
import { useStore } from "@/lib/store";
import { useAskQuestion } from "./useAskQuestion";

function recentQuestions(history: string[], limit = 4): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (let i = history.length - 1; i >= 0 && out.length < limit; i--) {
    const q = history[i];
    if (!seen.has(q)) {
      seen.add(q);
      out.push(q);
    }
  }
  return out;
}

/** Recent questions (last 4, most recent first, deduped) followed by the
 * example set -- one horizontally-scrollable chip row, tap to ask directly. */
export function SuggestionChips() {
  const history = useStore((s) => s.questionHistory);
  const ask = useAskQuestion();

  const recent = useMemo(() => recentQuestions(history), [history]);
  const chips = useMemo(() => {
    const seen = new Set(recent);
    return [...recent, ...EXAMPLE_QUESTIONS.filter((q) => !seen.has(q))];
  }, [recent]);

  return (
    <div className="chip-row" role="list" aria-label="Suggested questions">
      {chips.map((q, i) => (
        <button key={`${q}-${i}`} type="button" className="chip" onClick={() => ask(q)}>
          {q}
        </button>
      ))}
    </div>
  );
}

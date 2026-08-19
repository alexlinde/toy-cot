"use client";

import { EXAMPLE_QUESTIONS } from "@/lib/constants";
import { useAskQuestion } from "./useAskQuestion";

export function WelcomeCard() {
  const ask = useAskQuestion();
  return (
    <div className="welcome-card">
      <p className="welcome-card__title">
        Chain-of-Thought VLM ready. Ask questions about the shapes:
      </p>
      <div className="chip-row" role="list" aria-label="Example questions">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button key={q} type="button" className="chip" onClick={() => ask(q)}>
            {q}
          </button>
        ))}
      </div>
      <p className="welcome-card__hint">
        Click any word of an answer&rsquo;s chain to see where the model was looking when it
        emitted that word.
      </p>
    </div>
  );
}

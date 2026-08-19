"use client";

import type { KeyboardEvent } from "react";
import { useState } from "react";
import { useStore } from "@/lib/store";
import { useAskQuestion } from "./useAskQuestion";

/** The text input + Ask button. ArrowUp/ArrowDown cycle prior questions,
 * matching the Tk GUI's entry-field history navigation. */
export function QuestionInput({ disabled = false }: { disabled?: boolean }) {
  const history = useStore((s) => s.questionHistory);
  const ask = useAskQuestion();
  const [value, setValue] = useState("");
  const [historyIndex, setHistoryIndex] = useState(-1);

  function submit() {
    if (!value.trim()) return;
    ask(value);
    setValue("");
    setHistoryIndex(-1);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      submit();
      return;
    }
    if (e.key === "ArrowUp") {
      if (!history.length) return;
      e.preventDefault();
      const next = historyIndex === -1 ? history.length - 1 : Math.max(0, historyIndex - 1);
      setHistoryIndex(next);
      setValue(history[next]);
      return;
    }
    if (e.key === "ArrowDown") {
      if (!history.length || historyIndex === -1) return;
      e.preventDefault();
      if (historyIndex < history.length - 1) {
        const next = historyIndex + 1;
        setHistoryIndex(next);
        setValue(history[next]);
      } else {
        setHistoryIndex(-1);
        setValue("");
      }
    }
  }

  return (
    <div className="input-row">
      <input
        className="text-input"
        aria-label="Ask a question"
        placeholder="Ask a question about the shapes..."
        value={value}
        disabled={disabled}
        onChange={(e) => {
          setValue(e.target.value);
          setHistoryIndex(-1);
        }}
        onKeyDown={handleKeyDown}
        enterKeyHint="send"
      />
      <button type="button" className="ask-btn" disabled={disabled} onClick={submit}>
        Ask
      </button>
    </div>
  );
}

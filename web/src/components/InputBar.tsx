"use client";

import { useStore } from "@/lib/store";
import { LoadingBar } from "./LoadingBar";
import { QuestionInput } from "./QuestionInput";
import { SuggestionChips } from "./SuggestionChips";

/** Swaps between the loading progress bar, an error notice, and the live
 * question input + suggestion chips, keyed off model status. */
export function InputBar() {
  const status = useStore((s) => s.modelStatus.status);

  if (status === "loading") {
    return (
      <div className="input-bar">
        <LoadingBar />
      </div>
    );
  }

  return (
    <div className="input-bar">
      <SuggestionChips />
      {status === "error" && (
        <div className="error-banner">Model failed to load - reload the page to try again.</div>
      )}
      <QuestionInput disabled={status !== "ready"} />
    </div>
  );
}

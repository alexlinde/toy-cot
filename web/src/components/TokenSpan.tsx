"use client";

import type { ChatToken } from "@/lib/store";
import { useStore } from "@/lib/store";

function confidenceClass(prob: number): string | null {
  if (prob < 0.5) return "token--low";
  if (prob < 0.8) return "token--mid";
  if (prob < 0.95) return "token--high";
  return null;
}

/** One word of a rationale/answer chain: tinted by confidence, and -- while
 * its scene is still on the canvas -- hoverable (preview) and clickable
 * (pin) to show the attention map it was emitted with. */
export function TokenSpan({
  token,
  responseId,
  tokenIndex,
  stale,
}: {
  token: ChatToken;
  responseId: number;
  tokenIndex: number;
  stale: boolean;
}) {
  const selection = useStore((s) => s.selection);
  const setHovered = useStore((s) => s.setHovered);
  const toggleSelection = useStore((s) => s.toggleSelection);

  const clickable = !stale && token.attn !== null;
  const isSelected =
    !!selection && selection.responseId === responseId && selection.tokenIndex === tokenIndex;

  const classes = ["token"];
  const conf = confidenceClass(token.prob);
  if (conf) classes.push(conf);
  if (clickable) classes.push("token--clickable");
  if (isSelected) classes.push("token--selected");

  return (
    <span
      className={classes.join(" ")}
      title={stale ? "scene changed - attention no longer applies" : undefined}
      onMouseEnter={() => {
        if (clickable) setHovered({ responseId, tokenIndex });
      }}
      onMouseLeave={() => {
        if (clickable) setHovered(null);
      }}
      onClick={() => {
        if (clickable) toggleSelection({ responseId, tokenIndex });
      }}
    >
      {token.word}
    </span>
  );
}

"use client";

import { useEffect, useRef } from "react";
import type { ChatResponse, ChatToken } from "@/lib/store";
import { useStore } from "@/lib/store";
import { ModelStatsCard } from "./ModelStatsCard";
import { TokenSpan } from "./TokenSpan";
import { WelcomeCard } from "./WelcomeCard";

function TokenList({
  tokens,
  responseId,
  startIndex,
  stale,
}: {
  tokens: ChatToken[];
  responseId: number;
  startIndex: number;
  stale: boolean;
}) {
  return (
    <>
      {tokens.map((t, i) => (
        <TokenSpan
          key={i}
          token={t}
          responseId={responseId}
          tokenIndex={startIndex + i}
          stale={stale}
        />
      ))}
    </>
  );
}

function ResponseBlock({ response }: { response: ChatResponse }) {
  const rationaleLen = response.rationaleTokens.length;
  return (
    <>
      <div className="msg-user">{response.question}</div>
      <div className={response.stale ? "msg-model is-stale" : "msg-model"}>
        <div>
          <span className="msg-model__label">Reasoning:</span>
          <TokenList
            tokens={response.rationaleTokens}
            responseId={response.id}
            startIndex={0}
            stale={response.stale}
          />
        </div>
        <div>
          <span className="msg-model__label">Answer:</span>
          <TokenList
            tokens={response.answerTokens}
            responseId={response.id}
            startIndex={rationaleLen}
            stale={response.stale}
          />
        </div>
        {response.topk.length > 0 && (
          <div className="msg-model__alts">
            alternatives: {response.topk.map(([w, p]) => `${w} ${p.toFixed(2)}`).join(" · ")}
          </div>
        )}
        {response.status === "error" && (
          <div className="msg-model__error">{response.error ?? "Something went wrong."}</div>
        )}
      </div>
    </>
  );
}

/** The scrollable transcript: welcome card, model stats once ready, then
 * every asked question with its streamed response. Auto-scrolls to the
 * bottom on new content unless the user has scrolled up to read back. */
export function ChatLog() {
  const responses = useStore((s) => s.responses);
  const modelStatus = useStore((s) => s.modelStatus);
  const scrollRef = useRef<HTMLDivElement>(null);
  const userScrolledUpRef = useRef(false);

  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    userScrolledUpRef.current = distanceFromBottom > 32;
  }

  // Deliberately no dependency array: re-checked after every render (i.e.
  // every streamed token), which is the only way to keep pinned to the
  // bottom during a fast token stream.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || userScrolledUpRef.current) return;
    el.scrollTop = el.scrollHeight;
  });

  return (
    <div className="chat-log" ref={scrollRef} onScroll={handleScroll}>
      <WelcomeCard />
      {modelStatus.status === "ready" && <ModelStatsCard manifest={modelStatus.manifest} />}
      {responses.map((r) => (
        <ResponseBlock key={r.id} response={r} />
      ))}
    </div>
  );
}

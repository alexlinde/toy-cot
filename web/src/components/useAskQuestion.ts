"use client";

import { useCallback } from "react";
import { useStore } from "@/lib/store";
import { workerClient } from "@/lib/workerClient";

/** Mirror the ask into the address bar: the URL is the share link, so after
 * every question it deep-links exactly what's on screen (?seed=&q= re-asks
 * this question on this scene). Scene changes drop q again (SceneControls). */
function writeAskToUrl(seed: number, question: string): void {
  const url = new URL(window.location.href);
  url.search = "";
  url.searchParams.set("seed", String(seed));
  url.searchParams.set("q", question);
  window.history.replaceState(null, "", url.toString());
}

/** The single "submit a question" path -- used by the text input's Enter/Ask
 * button and by every chip (example or recent-question) that asks on tap
 * instead of just filling the input. */
export function useAskQuestion(): (raw: string) => void {
  const scene = useStore((s) => s.scene);
  const pushQuestionHistory = useStore((s) => s.pushQuestionHistory);
  const clearSelection = useStore((s) => s.clearSelection);
  const askQuestion = useStore((s) => s.askQuestion);

  return useCallback(
    (raw: string) => {
      const question = raw.trim();
      if (!question || !scene) return;
      pushQuestionHistory(question);
      clearSelection();
      const id = workerClient.ask(scene.seed, scene.image, question);
      askQuestion(id, scene.seed, question);
      writeAskToUrl(scene.seed, question);
    },
    [scene, pushQuestionHistory, clearSelection, askQuestion],
  );
}

"use client";

import { useSyncExternalStore } from "react";

/** Matches the desktop/mobile breakpoint used throughout the app. */
const QUERY = "(min-width: 900px)";

function subscribe(callback: () => void): () => void {
  const mql = window.matchMedia(QUERY);
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function getSnapshot(): boolean {
  return window.matchMedia(QUERY).matches;
}

function getServerSnapshot(): boolean {
  return false;
}

/** SSR-safe (min-width: 900px) check: renders the mobile layout for the
 * server/first-paint snapshot, then syncs to the real value on the client. */
export function useIsDesktop(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

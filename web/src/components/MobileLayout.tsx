"use client";

import { AttentionControls } from "./AttentionControls";
import { ChatLog } from "./ChatLog";
import { InputBar } from "./InputBar";
import { SceneCanvas } from "./SceneCanvas";
import { SceneControls } from "./SceneControls";

export function MobileLayout() {
  return (
    <div className="mobile-shell">
      <header className="mobile-header">
        <SceneCanvas />
        <div className="mobile-controls-row">
          <AttentionControls />
          <SceneControls />
        </div>
      </header>
      <ChatLog />
      <InputBar />
    </div>
  );
}

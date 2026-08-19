"use client";

import { AttentionControls } from "./AttentionControls";
import { ChatLog } from "./ChatLog";
import { InputBar } from "./InputBar";
import { SceneCanvas } from "./SceneCanvas";
import { SceneControls } from "./SceneControls";

export function DesktopLayout() {
  return (
    <div className="desktop-shell">
      <aside className="panel-left">
        <SceneCanvas />
        <AttentionControls />
        <SceneControls />
      </aside>
      <section className="panel-right">
        <ChatLog />
        <InputBar />
      </section>
    </div>
  );
}

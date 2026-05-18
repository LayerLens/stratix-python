/**
 * MarkdownLite — a tiny inline-markdown renderer for SDK card text.
 *
 * LayerLens API responses (judge ``evaluation_goal``, judge verdict
 * ``reasoning``, etc.) come back with markdown formatting: paragraph
 * breaks (``\n\n``), line breaks (``\n``), and inline ``**bold**`` /
 * ``*italic*`` segments. Rendering them through a single
 * ``<p>{text}</p>`` collapses the structure into one wall of text
 * which makes long goal/reasoning fields illegible.
 *
 * We intentionally don't pull ``react-markdown`` because the SDK card
 * library lives outside the Next.js app's ``node_modules`` and Node's
 * resolution can't reach down into ``app/frontend/node_modules``. A
 * tiny built-in renderer covers the cases the LayerLens API actually
 * emits without a runtime dep, using only React fragments + element
 * trees so no raw HTML is ever injected.
 *
 * Supported syntax:
 *   - Paragraph break: ``\n\n``
 *   - Line break: ``\n``
 *   - Bold: ``**text**``
 *   - Italic: ``*text*``
 *
 * Anything else (links, headings, lists, code) falls through as
 * plain text. If you need richer markdown in your own app, swap in
 * ``react-markdown`` at the call site — the cards take this through
 * a renderable prop, not as a hard dependency.
 */

import * as React from "react";

import { cn } from "@/lib/utils";

function renderItalic(text: string, keyPrefix: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const matches = Array.from(text.matchAll(/(?<!\*)\*([^*]+)\*(?!\*)/g));
  let lastIndex = 0;
  matches.forEach((m, i) => {
    const idx = m.index ?? 0;
    if (idx > lastIndex) {
      parts.push(
        <React.Fragment key={`${keyPrefix}-t${i}`}>
          {text.slice(lastIndex, idx)}
        </React.Fragment>,
      );
    }
    parts.push(
      <em key={`${keyPrefix}-i${i}`} className="italic">
        {m[1]}
      </em>,
    );
    lastIndex = idx + m[0].length;
  });
  if (lastIndex < text.length) {
    parts.push(
      <React.Fragment key={`${keyPrefix}-tail`}>
        {text.slice(lastIndex)}
      </React.Fragment>,
    );
  }
  return parts;
}

function renderInline(line: string, keyPrefix: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const matches = Array.from(line.matchAll(/\*\*([^*]+)\*\*/g));
  let lastIndex = 0;
  matches.forEach((m, i) => {
    const idx = m.index ?? 0;
    if (idx > lastIndex) {
      parts.push(
        ...renderItalic(line.slice(lastIndex, idx), `${keyPrefix}-t${i}`),
      );
    }
    parts.push(
      <strong key={`${keyPrefix}-b${i}`} className="font-semibold text-foreground">
        {m[1]}
      </strong>,
    );
    lastIndex = idx + m[0].length;
  });
  if (lastIndex < line.length) {
    parts.push(
      ...renderItalic(line.slice(lastIndex), `${keyPrefix}-tail`),
    );
  }
  return parts;
}

export interface MarkdownLiteProps {
  text: string;
  /** Extra classes applied to the outer container. The default uses
   *  shadcn neutral foreground + small body type. */
  className?: string;
}

export function MarkdownLite({ text, className }: MarkdownLiteProps) {
  const paragraphs = text.split(/\n\s*\n/);
  return (
    <div className={cn("space-y-2 text-sm leading-relaxed text-foreground", className)}>
      {paragraphs.map((para, pi) => {
        const lines = para.split("\n");
        return (
          <p key={pi}>
            {lines.map((line, li) => (
              <React.Fragment key={li}>
                {renderInline(line, `p${pi}l${li}`)}
                {li < lines.length - 1 ? <br /> : null}
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}

export default MarkdownLite;

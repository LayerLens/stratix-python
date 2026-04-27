/**
 * MetricCard — General-purpose KPI tile.
 *
 * Built on shadcn/ui ``Card`` so it inherits the same neutral palette
 * and elevation conventions used across CopilotKit's official samples
 * (research-canvas, travel, banking). A single metric with optional
 * unit, trend indicator, and footnote.
 */

import * as React from "react";

import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";

// Inline SVG arrows — keeps the SDK card self-contained without
// pulling ``lucide-react`` (which can't be resolved when this card
// is imported from outside the Next.js app directory).
const ArrowUp = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden {...props}>
    <path d="M12 19V5M5 12l7-7 7 7" />
  </svg>
);
const ArrowDown = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden {...props}>
    <path d="M12 5v14M19 12l-7 7-7-7" />
  </svg>
);
const ArrowRight = (props: React.SVGProps<SVGSVGElement>) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden {...props}>
    <path d="M5 12h14M12 5l7 7-7 7" />
  </svg>
);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TrendDirection = "up" | "down" | "flat";

export interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: TrendDirection;
  trendValue?: string;
  description?: string;
  /** Optional Tailwind className extension for the outer ``Card``. */
  className?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TREND_CONFIG: Record<
  TrendDirection,
  { Icon: typeof ArrowUp; cls: string; label: string }
> = {
  up: {
    Icon: ArrowUp,
    cls: "text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400",
    label: "Trending up",
  },
  down: {
    Icon: ArrowDown,
    cls: "text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400",
    label: "Trending down",
  },
  flat: {
    Icon: ArrowRight,
    cls: "text-muted-foreground bg-muted",
    label: "Stable",
  },
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  trend,
  trendValue,
  description,
  className,
}) => {
  const tc = trend ? TREND_CONFIG[trend] : null;

  return (
    <Card
      className={cn(
        "gap-2 py-5 transition-shadow duration-200 hover:shadow-md",
        className,
      )}
    >
      <CardContent className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            {label}
          </p>
          {tc && trendValue ? (
            <span
              aria-label={tc.label}
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
                tc.cls,
              )}
            >
              <tc.Icon className="h-3 w-3" aria-hidden />
              <span className="tabular-nums">{trendValue}</span>
            </span>
          ) : null}
        </div>

        <div className="flex items-baseline gap-1.5">
          <span className="text-2xl font-semibold tabular-nums tracking-tight">
            {value}
          </span>
          {unit ? (
            <span className="text-sm text-muted-foreground">{unit}</span>
          ) : null}
        </div>

        {description ? (
          <p className="text-xs text-muted-foreground">{description}</p>
        ) : null}
      </CardContent>
    </Card>
  );
};

export default MetricCard;

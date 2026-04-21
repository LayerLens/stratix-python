/**
 * MetricCard — Simple KPI card for embedding in CopilotKit chat.
 *
 * Renders a label, large value with optional unit, trend indicator, and
 * optional description text.
 */

import React from "react";

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
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TREND_CONFIG: Record<
  TrendDirection,
  { arrow: string; cls: string }
> = {
  up: {
    arrow: "\u2191", // up arrow
    cls: "text-emerald-600 dark:text-emerald-400",
  },
  down: {
    arrow: "\u2193", // down arrow
    cls: "text-red-600 dark:text-red-400",
  },
  flat: {
    arrow: "\u2192", // right arrow
    cls: "text-gray-400 dark:text-gray-500",
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
}) => {
  const tc = trend ? TREND_CONFIG[trend] : null;

  return (
    <div className="w-full max-w-xs overflow-hidden rounded-xl border border-gray-200 bg-white px-4 py-3 shadow-sm dark:border-gray-700 dark:bg-gray-800">
      {/* Label */}
      <p className="text-xs font-medium uppercase tracking-wide text-gray-500 dark:text-gray-400">
        {label}
      </p>

      {/* Value row */}
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className="text-2xl font-bold tabular-nums text-gray-900 dark:text-gray-100">
          {value}
        </span>
        {unit && (
          <span className="text-sm text-gray-400">{unit}</span>
        )}
        {tc && trendValue && (
          <span className={`ml-auto flex items-center gap-0.5 text-sm font-medium ${tc.cls}`}>
            <span>{tc.arrow}</span>
            <span>{trendValue}</span>
          </span>
        )}
      </div>

      {/* Description */}
      {description && (
        <p className="mt-1.5 text-xs leading-relaxed text-gray-500 dark:text-gray-400">
          {description}
        </p>
      )}
    </div>
  );
};

export default MetricCard;

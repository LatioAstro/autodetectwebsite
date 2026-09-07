"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

type LightCurvePoint = Record<string, unknown>;

type LightCurveInterval = Record<string, unknown> | [number, number] | [number, number, number];

type MdpLabel = Record<string, unknown> | [number, number, number];

export type LightCurveData = {
    background?: number | null;
    quiescentBackground?: number | null;
    quiescent_background?: number | null;
    flareThreshold?: number | null;
    flare_threshold?: number | null;
    flareIntervals?: LightCurveInterval[];
    confirmedFlareIntervals?: LightCurveInterval[];
    flareMdpLabels?: MdpLabel[];
    fluxScale?: number | null;
    _scanState?: { fluxScale?: number | null } | null;
    points: LightCurvePoint[];
};

// COSI energy band used to convert LAT photon flux into COSI-scaled counts.
const COSI_ENERGY_RANGE_LABEL = "0.2\u20135 MeV";

type LightCurveProps = {
    data: LightCurveData;
    title?: string;
    className?: string;
};

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

function toFiniteNumber(value: unknown): number | null {
    if (typeof value === "number" && Number.isFinite(value)) {
        return value;
    }
    if (typeof value === "string") {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
}

function toBoolean(value: unknown): boolean {
    if (typeof value === "boolean") {
        return value;
    }
    if (typeof value === "number") {
        return value !== 0;
    }
    if (typeof value === "string") {
        const normalized = value.toLowerCase();
        return normalized === "true" || normalized === "1" || normalized === "yes";
    }
    return false;
}

function normalizedPoint(raw: LightCurvePoint) {
    const mjd = toFiniteNumber(raw.mjd ?? raw.time_MJD ?? raw.timeMJD ?? raw.time);
    const flux = toFiniteNumber(raw.flux ?? raw.new_point_flux ?? raw.value);
    const error = toFiniteNumber(raw.error ?? raw.err ?? raw.flux_error ?? raw.fluxError);
    const potentialFlare = toBoolean(
        raw.flare ?? raw.potentialFlarePoint ?? raw.potential_flare_point ?? raw.highlight,
    );

    if (mjd === null || flux === null) {
        return null;
    }

    return {
        mjd,
        flux,
        error,
        potentialFlare,
    };
}

function getIntervalBounds(interval: LightCurveInterval): [number, number] | null {
    if (Array.isArray(interval)) {
        if (interval.length < 2) {
            return null;
        }

        const startArray = toFiniteNumber(interval[0]);
        const endArray = toFiniteNumber(interval[1]);
        if (startArray === null || endArray === null) {
            return null;
        }
        return [Math.min(startArray, endArray), Math.max(startArray, endArray)];
    }

    const start = toFiniteNumber(
        interval.startMjd ?? interval.start ?? interval.mjdStart ?? interval.start_mjd,
    );
    const end = toFiniteNumber(interval.endMjd ?? interval.end ?? interval.mjdEnd ?? interval.end_mjd);

    if (start === null || end === null) {
        return null;
    }

    return [Math.min(start, end), Math.max(start, end)];
}

function getMdpLabel(item: MdpLabel): [number, number, number] | null {
    if (Array.isArray(item)) {
        if (item.length < 3) {
            return null;
        }

        const start = toFiniteNumber(item[0]);
        const end = toFiniteNumber(item[1]);
        const mdp = toFiniteNumber(item[2]);
        if (start === null || end === null || mdp === null) {
            return null;
        }

        return [Math.min(start, end), Math.max(start, end), mdp];
    }

    const start = toFiniteNumber(item.startMjd ?? item.start ?? item.mjdStart ?? item.start_mjd);
    const end = toFiniteNumber(item.endMjd ?? item.end ?? item.mjdEnd ?? item.end_mjd);
    const mdp = toFiniteNumber(item.mdp99 ?? item.mdp99_percent ?? item.mdp ?? item.label);
    if (start === null || end === null || mdp === null) {
        return null;
    }

    return [Math.min(start, end), Math.max(start, end), mdp];
}

function getMdpLabelFromInterval(interval: LightCurveInterval): [number, number, number] | null {
    if (Array.isArray(interval)) {
        if (interval.length < 3) {
            return null;
        }

        const start = toFiniteNumber(interval[0]);
        const end = toFiniteNumber(interval[1]);
        const mdp = toFiniteNumber(interval[2]);
        if (start === null || end === null || mdp === null) {
            return null;
        }

        return [Math.min(start, end), Math.max(start, end), mdp];
    }

    const start = toFiniteNumber(
        interval.startMjd ?? interval.start ?? interval.mjdStart ?? interval.start_mjd,
    );
    const end = toFiniteNumber(interval.endMjd ?? interval.end ?? interval.mjdEnd ?? interval.end_mjd);

    const numericMdp = toFiniteNumber(interval.mdp99 ?? interval.mdp99_percent ?? interval.mdp);
    const labelMdpMatch =
        typeof interval.label === "string" ? interval.label.match(/(-?\d+(?:\.\d+)?)/) : null;
    const labelMdp = labelMdpMatch ? toFiniteNumber(labelMdpMatch[1]) : null;
    const mdp = numericMdp ?? labelMdp;

    if (start === null || end === null || mdp === null) {
        return null;
    }

    return [Math.min(start, end), Math.max(start, end), mdp];
}

function deriveIntervalsFromPotentialFlarePoints(
    points: Array<{ mjd: number; potentialFlare: boolean }>,
): Array<[number, number]> {
    const intervals: Array<[number, number]> = [];
    let runStart: number | null = null;
    let runEnd: number | null = null;

    for (const point of points) {
        if (point.potentialFlare) {
            if (runStart === null) {
                runStart = point.mjd;
            }
            runEnd = point.mjd;
        } else if (runStart !== null && runEnd !== null) {
            intervals.push([runStart, runEnd]);
            runStart = null;
            runEnd = null;
        }
    }

    if (runStart !== null && runEnd !== null) {
        intervals.push([runStart, runEnd]);
    }

    return intervals;
}

function buildPointHoverList(
    point: { mjd: number; flux: number; error: number | null; potentialFlare: boolean },
    mdp99: number | null,
): string {
    const errorText = point.error === null ? "n/a" : point.error.toExponential(3);
    const valueColor = point.potentialFlare ? "#b4535a" : "#111111";
    const lines = [
        `<span style="color:#0b6e99;"><b>Flux</b></span>: <span style="color:${valueColor};">${point.flux.toExponential(3)}</span>`,
        `<span style="color:#4b5563;"><b>Time</b></span>: <span style="color:${valueColor};">${point.mjd.toFixed(5)}</span>`,
        `<span style="color:#b45309;"><b>Flux Error</b></span>: <span style="color:${valueColor};">${errorText}</span>`,
    ];

    if (mdp99 !== null) {
        lines.push(`<span style="color:#7c3aed;"><b>MDP99</b></span>: <span style="color:${valueColor};">${mdp99.toFixed(2)}%</span>`);
    }

    return `${lines.join("<br>")}<extra></extra>`;
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
}

function clipIntervalToRange(
    interval: [number, number],
    minMjd: number,
    maxMjd: number,
): [number, number] | null {
    const clippedStart = Math.max(interval[0], minMjd);
    const clippedEnd = Math.min(interval[1], maxMjd);

    if (clippedStart > clippedEnd) {
        return null;
    }

    return [clippedStart, clippedEnd];
}

export default function LightCurve({ data, title, className }: LightCurveProps) {
    const [showDashedBackground, setShowDashedBackground] = useState(true);
    const [showDashedThreshold, setShowDashedThreshold] = useState(true);
    const [showPotentialFlarePoints, setShowPotentialFlarePoints] = useState(true);
    const [showHighlightedIntervals, setShowHighlightedIntervals] = useState(true);
    const [showMdpLabels, setShowMdpLabels] = useState(false);
    const [showGrid, setShowGrid] = useState(true);
    const [showCosiScale, setShowCosiScale] = useState(true);
    const [mjdRangeMinInput, setMjdRangeMinInput] = useState("");
    const [mjdRangeMaxInput, setMjdRangeMaxInput] = useState("");

    const fluxScale = toFiniteNumber(data._scanState?.fluxScale) ?? toFiniteNumber(data.fluxScale);
    const scaleFactor = showCosiScale && fluxScale !== null ? fluxScale : 1;

    const points = (data.points ?? [])
        .map(normalizedPoint)
        .filter((point): point is NonNullable<ReturnType<typeof normalizedPoint>> => point !== null)
        .sort((a, b) => a.mjd - b.mjd);

    const datasetMinMjd = points.length > 0 ? points[0].mjd : null;
    const datasetMaxMjd = points.length > 0 ? points[points.length - 1].mjd : null;

    useEffect(() => {
        if (datasetMinMjd === null || datasetMaxMjd === null) {
            setMjdRangeMinInput("");
            setMjdRangeMaxInput("");
            return;
        }

        setMjdRangeMinInput(datasetMinMjd.toFixed(5));
        setMjdRangeMaxInput(datasetMaxMjd.toFixed(5));
    }, [datasetMinMjd, datasetMaxMjd]);

    const parsedInputMin = toFiniteNumber(mjdRangeMinInput);
    const parsedInputMax = toFiniteNumber(mjdRangeMaxInput);

    const rangeMinRaw =
        datasetMinMjd === null ? null : parsedInputMin === null ? datasetMinMjd : parsedInputMin;
    const rangeMaxRaw =
        datasetMaxMjd === null ? null : parsedInputMax === null ? datasetMaxMjd : parsedInputMax;

    const effectiveRangeMin =
        datasetMinMjd === null || datasetMaxMjd === null || rangeMinRaw === null || rangeMaxRaw === null
            ? null
            : clamp(Math.min(rangeMinRaw, rangeMaxRaw), datasetMinMjd, datasetMaxMjd);

    const effectiveRangeMax =
        datasetMinMjd === null || datasetMaxMjd === null || rangeMinRaw === null || rangeMaxRaw === null
            ? null
            : clamp(Math.max(rangeMinRaw, rangeMaxRaw), datasetMinMjd, datasetMaxMjd);

    const pointsInRange =
        effectiveRangeMin === null || effectiveRangeMax === null
            ? points
            : points.filter((point) => point.mjd >= effectiveRangeMin && point.mjd <= effectiveRangeMax);

    const mjd = pointsInRange.map((point) => point.mjd);
    const flux = pointsInRange.map((point) => point.flux * scaleFactor);
    const allFlux = points.map((point) => point.flux * scaleFactor);
    const errors = pointsInRange.map((point) => point.error);
    const hasErrorBars = errors.some((value) => value !== null);
    const errorArray = errors.map((value) => (value ?? 0) * scaleFactor);

    const flarePoints = pointsInRange.filter((point) => point.potentialFlare);
    const flareMjd = flarePoints.map((point) => point.mjd);
    const flareFlux = flarePoints.map((point) => point.flux * scaleFactor);

    const intervalCandidates = [
        ...(data.flareIntervals ?? []),
        ...(data.confirmedFlareIntervals ?? []),
    ];

    const intervalsFromJson = intervalCandidates
        .map(getIntervalBounds)
        .filter((interval): interval is [number, number] => interval !== null);

    const baseIntervals =
        intervalsFromJson.length > 0 ? intervalsFromJson : deriveIntervalsFromPotentialFlarePoints(points);

    const intervals =
        effectiveRangeMin === null || effectiveRangeMax === null
            ? baseIntervals
            : baseIntervals
                  .map((interval) => clipIntervalToRange(interval, effectiveRangeMin, effectiveRangeMax))
                  .filter((interval): interval is [number, number] => interval !== null);

    const mdpLabelsFromDedicatedField = (data.flareMdpLabels ?? [])
        .map(getMdpLabel)
        .filter((label): label is [number, number, number] => label !== null);

    const mdpLabelsFromIntervals = intervalCandidates
        .map(getMdpLabelFromInterval)
        .filter((label): label is [number, number, number] => label !== null);

    const baseMdpLabels =
        mdpLabelsFromDedicatedField.length > 0 ? mdpLabelsFromDedicatedField : mdpLabelsFromIntervals;

    const mdpLabels =
        effectiveRangeMin === null || effectiveRangeMax === null
            ? baseMdpLabels
            : baseMdpLabels
                  .map(([start, end, mdp99]) => {
                      const clipped = clipIntervalToRange([start, end], effectiveRangeMin, effectiveRangeMax);
                      return clipped ? [clipped[0], clipped[1], mdp99] : null;
                  })
                  .filter((label): label is [number, number, number] => label !== null);

    const rawIntervalHoverEntries =
        intervalsFromJson.length > 0
            ? intervalCandidates
                  .map((interval) => {
                      const bounds = getIntervalBounds(interval);
                      if (bounds === null) {
                          return null;
                      }
                      const mdpLabel = getMdpLabelFromInterval(interval);
                      return {
                          x0: bounds[0],
                          x1: bounds[1],
                          mdp99: mdpLabel ? mdpLabel[2] : null,
                      };
                  })
                  .filter(
                      (entry): entry is { x0: number; x1: number; mdp99: number | null } => entry !== null,
                  )
            : baseIntervals.map(([x0, x1]) => ({ x0, x1, mdp99: null }));

    const intervalHoverEntries =
        effectiveRangeMin === null || effectiveRangeMax === null
            ? rawIntervalHoverEntries
            : rawIntervalHoverEntries
                  .map((entry) => {
                      const clipped = clipIntervalToRange([entry.x0, entry.x1], effectiveRangeMin, effectiveRangeMax);
                      if (clipped === null) {
                          return null;
                      }
                      return {
                          x0: clipped[0],
                          x1: clipped[1],
                          mdp99: entry.mdp99,
                      };
                  })
                  .filter(
                      (entry): entry is { x0: number; x1: number; mdp99: number | null } => entry !== null,
                  );

    const quiescentBackgroundRaw =
        toFiniteNumber(data.quiescentBackground) ??
        toFiniteNumber(data.quiescent_background) ??
        toFiniteNumber(data.background);
    const quiescentBackground = quiescentBackgroundRaw === null ? null : quiescentBackgroundRaw * scaleFactor;

    const flareThresholdRaw = toFiniteNumber(data.flareThreshold) ?? toFiniteNumber(data.flare_threshold);
    const flareThreshold = flareThresholdRaw === null ? null : flareThresholdRaw * scaleFactor;

    const fluxForScale = flux.length > 0 ? flux : allFlux;
    const yValuesForScale =
        fluxForScale.length > 0
            ? [...fluxForScale, ...(quiescentBackground !== null ? [quiescentBackground] : [])]
            : [0, ...(quiescentBackground !== null ? [quiescentBackground] : [])];

    const yMin = Math.min(...yValuesForScale);
    const yMax = Math.max(...yValuesForScale);
    const ySpan = Math.max(yMax - yMin, Math.abs(yMax) * 0.05, 1e-12);
    const mdpY = yMax - ySpan * 0.06;

    const plotTitle = title ?? "Light Curve";

    const mdpAtMjd = (mjdValue: number): number | null => {
        const match = intervalHoverEntries.find(
            (entry) => entry.mdp99 !== null && mjdValue >= entry.x0 && mjdValue <= entry.x1,
        );
        return match?.mdp99 ?? null;
    };

    const weeklyHoverText = pointsInRange.map((point) =>
        buildPointHoverList(
            {
                ...point,
                flux: point.flux * scaleFactor,
                error: point.error === null ? null : point.error * scaleFactor,
            },
            mdpAtMjd(point.mjd),
        ),
    );

    return (
        <div className={className}>
            <div className="relative mx-auto w-full max-w-6xl">
                <aside className="mb-4 w-full rounded-md border border-zinc-200 bg-zinc-50 p-4 text-sm lg:absolute lg:left-0 lg:top-0 lg:mb-0 lg:w-72 lg:-translate-x-[calc(100%+1rem)] dark:border-zinc-800 dark:bg-zinc-950">
                    <h2 className="mb-3 text-base font-semibold text-zinc-900 dark:text-zinc-100">Plot Options</h2>
                    <div className="space-y-2 text-zinc-700 dark:text-zinc-300">
                        <div className="space-y-2 rounded border border-zinc-200 bg-white p-2 dark:border-zinc-700 dark:bg-zinc-900">
                            <div className="text-xs font-medium uppercase tracking-wide text-zinc-600 dark:text-zinc-400">
                                MJD range
                            </div>
                            <label className="flex items-center justify-between gap-2">
                                <span className="text-xs">Min</span>
                                <input
                                    type="number"
                                    step="0.00001"
                                    value={mjdRangeMinInput}
                                    onChange={(event) => setMjdRangeMinInput(event.target.value)}
                                    disabled={datasetMinMjd === null}
                                    className="w-32 rounded border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
                                />
                            </label>
                            <label className="flex items-center justify-between gap-2">
                                <span className="text-xs">Max</span>
                                <input
                                    type="number"
                                    step="0.00001"
                                    value={mjdRangeMaxInput}
                                    onChange={(event) => setMjdRangeMaxInput(event.target.value)}
                                    disabled={datasetMaxMjd === null}
                                    className="w-32 rounded border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
                                />
                            </label>
                            <button
                                type="button"
                                disabled={datasetMinMjd === null || datasetMaxMjd === null}
                                onClick={() => {
                                    if (datasetMinMjd !== null && datasetMaxMjd !== null) {
                                        setMjdRangeMinInput(datasetMinMjd.toFixed(5));
                                        setMjdRangeMaxInput(datasetMaxMjd.toFixed(5));
                                    }
                                }}
                                className="w-full rounded border border-zinc-300 bg-zinc-100 px-2 py-1 text-xs text-zinc-800 hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
                            >
                                Reset to full range
                            </button>
                        </div>
                        <label className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                checked={showDashedBackground}
                                onChange={(event) => setShowDashedBackground(event.target.checked)}
                            />
                            Show dashed background line
                        </label>
                        <label className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                checked={showDashedThreshold}
                                onChange={(event) => setShowDashedThreshold(event.target.checked)}
                            />
                            Show dashed threshold line
                        </label>
                        <label className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                checked={showPotentialFlarePoints}
                                onChange={(event) => setShowPotentialFlarePoints(event.target.checked)}
                            />
                            Show potential flare points
                        </label>
                        <label className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                checked={showHighlightedIntervals}
                                onChange={(event) => setShowHighlightedIntervals(event.target.checked)}
                            />
                            Show confirmed flare regions
                        </label>
                        <label className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                checked={showMdpLabels}
                                onChange={(event) => setShowMdpLabels(event.target.checked)}
                            />
                            Show MDP99 labels above regions
                        </label>
                        <label className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                checked={showGrid}
                                onChange={(event) => setShowGrid(event.target.checked)}
                            />
                            Show grid
                        </label>
                    </div>
                </aside>

                <div className="min-w-0 w-full">
                    <Plot
                        data={[
                            {
                                x: mjd,
                                y: flux,
                                type: "scatter",
                                mode: "lines+markers",
                                name: "Weekly flux",
                                line: { color: "#000000", width: 1 },
                                marker: { size: 6, color: "#000000", symbol: "circle" },
                                hoverinfo: "skip",
                                error_y: {
                                    type: "data",
                                    array: errorArray,
                                    visible: hasErrorBars,
                                    color: "#808080",
                                    thickness: 1,
                                    width: 2,
                                },
                            },
                            {
                                x: mjd,
                                y: flux,
                                type: "scatter",
                                mode: "markers",
                                name: "Hover points",
                                showlegend: false,
                                marker: { size: 16, color: "rgba(0,0,0,0)", opacity: 0 },
                                hovertext: weeklyHoverText,
                                hovertemplate: "%{hovertext}",
                                hoverlabel: {
                                    bgcolor: "rgba(255, 255, 255, 0.98)",
                                    bordercolor: "#6b7280",
                                    font: { color: "#111111" },
                                },
                            },
                            ...(showPotentialFlarePoints
                                ? [
                                      {
                                          x: flareMjd,
                                          y: flareFlux,
                                          type: "scatter",
                                          mode: "markers",
                                          name: "Flaring points",
                                          marker: { color: "crimson", size: 8, symbol: "circle" },
                                          hoverinfo: "skip",
                                      },
                                  ]
                                : []),
                            ...(showDashedBackground && quiescentBackground !== null
                                ? [
                                      {
                                          x: mjd,
                                          y: mjd.map(() => quiescentBackground),
                                          type: "scatter",
                                          mode: "lines",
                                          name: "Quiescent background",
                                          line: { color: "#1f77b4", width: 1.8, dash: "dash" },
                                          hoverinfo: "skip",
                                      },
                                  ]
                                : []),
                            ...(showDashedThreshold && flareThreshold !== null
                                ? [
                                      {
                                          x: mjd,
                                          y: mjd.map(() => flareThreshold),
                                          type: "scatter",
                                          mode: "lines",
                                          name: "Flare threshold",
                                          opacity: 0.55,
                                          line: { color: "#ff7f0e", width: 1.4, dash: "dash" },
                                                                                    hoverinfo: "skip",
                                      },
                                  ]
                                : []),
                        ]}
                        layout={{
                            title: {
                                text: plotTitle,
                                x: 0.5,
                                xanchor: "center",
                                y: 0.98,
                                yanchor: "top",
                                font: {
                                    family: "'Avenir Next', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
                                    size: 18,
                                    color: "#111111",
                                },
                            },
                            xaxis: {
                                title: {
                                    text: "Time (MJD)",
                                    standoff: 20,
                                    font: {
                                        family: "'Avenir Next', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
                                        size: 15,
                                        color: "#111111",
                                    },
                                },
                                automargin: true,
                                tickformat: ".0f",
                                hoverformat: ".5f",
                                exponentformat: "none",
                                showexponent: "none",
                                showgrid: showGrid,
                                gridcolor: "rgba(0,0,0,0.25)",
                                gridwidth: 1,
                                tickfont: { color: "#111111" },
                                minallowed: effectiveRangeMin ?? undefined,
                                maxallowed: effectiveRangeMax ?? undefined,
                                range:
                                    effectiveRangeMin !== null && effectiveRangeMax !== null
                                        ? [effectiveRangeMin, effectiveRangeMax]
                                        : undefined,
                            },
                            yaxis: {
                                title: {
                                    text: showCosiScale
                                        ? `COSI Photon Flux (ph cm<sup>-2</sup> s<sup>-1</sup>) [${COSI_ENERGY_RANGE_LABEL}]`
                                        : "Photon Flux (ph cm<sup>-2</sup> s<sup>-1</sup>)",
                                    standoff: 18,
                                    font: {
                                        family: "'Avenir Next', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
                                        size: 15,
                                        color: "#111111",
                                    },
                                },
                                automargin: true,
                                tickformat: ".2e",
                                hoverformat: ".3e",
                                exponentformat: "power",
                                showexponent: "all",
                                showgrid: showGrid,
                                gridcolor: "rgba(0,0,0,0.25)",
                                gridwidth: 1,
                                tickfont: { color: "#111111" },
                                minallowed: yMin,
                                maxallowed: yMax,
                            },
                            legend: {
                                x: 0.01,
                                y: 0.99,
                                xanchor: "left",
                                yanchor: "top",
                                bgcolor: "rgba(255,255,255,0.9)",
                                bordercolor: "#d1d5db",
                                borderwidth: 1,
                            },
                            paper_bgcolor: "#ffffff",
                            plot_bgcolor: "#ffffff",
                            font: {
                                family: "'Avenir Next', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif",
                                color: "#111111",
                            },
                            margin: { l: 105, r: 40, t: 90, b: 95 },
                            hovermode: "x",
                            hoverdistance: -1,
                            shapes: showHighlightedIntervals
                                ? intervals.map(([x0, x1]) => ({
                                      type: "rect",
                                      xref: "x",
                                      yref: "paper",
                                      x0,
                                      x1,
                                      y0: 0,
                                      y1: 1,
                                      fillcolor: "rgba(255, 165, 0, 0.22)",
                                      line: { width: 0 },
                                      layer: "below",
                                  }))
                                : [],
                            annotations: showMdpLabels
                                ? mdpLabels.map(([start, end, mdp99]) => ({
                                      x: 0.5 * (start + end),
                                      y: mdpY,
                                      xref: "x",
                                      yref: "y",
                                      text: `MDP99: ${mdp99.toFixed(2)}%`,
                                      showarrow: false,
                                      font: { size: 12, color: "#000000" },
                                      bgcolor: "rgba(255,255,255,0.8)",
                                      bordercolor: "#d1d5db",
                                      borderwidth: 1,
                                      borderpad: 3,
                                  }))
                                : [],
                        }}
                        config={{ responsive: true, displaylogo: false }}
                        style={{ width: "100%", height: "620px" }}
                    />
                </div>
            </div>
        </div>
    );
}
/**
 * Exact decimal helper for display and light accounting math in the dashboard.
 *
 * The API sends every money / quantity / price as a canonical decimal *string* (Python
 * `format(Decimal, "f")`). This module never converts those to binary floats: values are parsed
 * into a BigInt coefficient plus a base-10 scale, and formatting rounds with ROUND_HALF_EVEN (the
 * Python `decimal` default) on the digits themselves.
 *
 * The only float conversion is `toChartNumber`, which exists solely to hand coordinates to the
 * chart library; it must never feed back into displayed accounting values.
 */

export interface Decimal {
  /** value = coefficient / 10^scale */
  readonly coefficient: bigint;
  readonly scale: number;
}

const DECIMAL_RE = /^([+-])?(\d+)?(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/;
const MAX_SCALE = 40;
const MAX_DIGITS = 60;

export const EMPTY_PLACEHOLDER = "—";
export const INVALID_PLACEHOLDER = "invalid";

export function parseDecimal(input: unknown): Decimal | null {
  if (typeof input === "bigint") return { coefficient: input, scale: 0 };
  if (typeof input !== "string") return null; // numbers are rejected on purpose (float boundary)
  const text = input.trim();
  const m = DECIMAL_RE.exec(text);
  if (!m) return null;
  const [, sign, intPart = "", fracPart = "", expPart] = m;
  if (intPart === "" && fracPart === "") return null;
  if (intPart.length + fracPart.length > MAX_DIGITS) return null;
  let scale = fracPart.length;
  let digits = (intPart + fracPart).replace(/^0+(?=\d)/, "") || "0";
  if (expPart !== undefined) {
    const exp = Number.parseInt(expPart, 10);
    if (!Number.isSafeInteger(exp) || Math.abs(exp) > MAX_SCALE) return null;
    scale -= exp;
    if (scale < 0) {
      digits = digits + "0".repeat(-scale);
      scale = 0;
    }
  }
  if (scale > MAX_SCALE) return null;
  let coefficient = BigInt(digits);
  if (sign === "-") coefficient = -coefficient;
  return { coefficient, scale };
}

export function isDecimalString(input: unknown): input is string {
  return parseDecimal(input) !== null;
}

function pow10(n: number): bigint {
  return 10n ** BigInt(n);
}

function rescale(d: Decimal, scale: number): Decimal {
  if (scale === d.scale) return d;
  if (scale > d.scale) return { coefficient: d.coefficient * pow10(scale - d.scale), scale };
  return roundToScale(d, scale);
}

/** Round to `scale` fractional digits using ROUND_HALF_EVEN. */
export function roundToScale(d: Decimal, scale: number): Decimal {
  if (scale >= d.scale) return rescale(d, scale);
  const factor = pow10(d.scale - scale);
  const negative = d.coefficient < 0n;
  const abs = negative ? -d.coefficient : d.coefficient;
  let q = abs / factor;
  const r = abs % factor;
  const twice = r * 2n;
  if (twice > factor || (twice === factor && q % 2n === 1n)) q += 1n;
  return { coefficient: negative ? -q : q, scale };
}

function align(a: Decimal, b: Decimal): [bigint, bigint, number] {
  const scale = Math.max(a.scale, b.scale);
  return [rescale(a, scale).coefficient, rescale(b, scale).coefficient, scale];
}

export function add(a: Decimal, b: Decimal): Decimal {
  const [x, y, scale] = align(a, b);
  return { coefficient: x + y, scale };
}

export function sub(a: Decimal, b: Decimal): Decimal {
  const [x, y, scale] = align(a, b);
  return { coefficient: x - y, scale };
}

export function mul(a: Decimal, b: Decimal): Decimal {
  return { coefficient: a.coefficient * b.coefficient, scale: a.scale + b.scale };
}

/** a / b rounded half-even to `scale` digits. Returns null on division by zero. */
export function div(a: Decimal, b: Decimal, scale = 10): Decimal | null {
  if (b.coefficient === 0n) return null;
  // (a.c / 10^a.s) / (b.c / 10^b.s) = a.c * 10^(b.s + scale + 1 - a.s) / b.c  at scale+1, then round.
  const extra = b.scale + scale + 1 - a.scale;
  let num = a.coefficient;
  let den = b.coefficient;
  if (extra >= 0) num *= pow10(extra);
  else den *= pow10(-extra);
  const negative = num < 0n !== den < 0n;
  const absNum = num < 0n ? -num : num;
  const absDen = den < 0n ? -den : den;
  let q = absNum / absDen;
  // a non-zero remainder past the guard digit must push exact halves upward (sticky bit)
  if (absNum % absDen !== 0n && q % 10n === 5n) q += 1n;
  return roundToScale({ coefficient: negative ? -q : q, scale: scale + 1 }, scale);
}

export function compare(a: Decimal, b: Decimal): -1 | 0 | 1 {
  const [x, y] = align(a, b);
  return x === y ? 0 : x < y ? -1 : 1;
}

export function isZero(d: Decimal): boolean {
  return d.coefficient === 0n;
}

export function sign(d: Decimal): -1 | 0 | 1 {
  return d.coefficient === 0n ? 0 : d.coefficient < 0n ? -1 : 1;
}

export function negate(d: Decimal): Decimal {
  return { coefficient: -d.coefficient, scale: d.scale };
}

export function abs(d: Decimal): Decimal {
  return d.coefficient < 0n ? negate(d) : d;
}

/** Canonical plain string (no exponent), e.g. "-12.50". */
export function toPlainString(d: Decimal): string {
  const negative = d.coefficient < 0n;
  let digits = (negative ? -d.coefficient : d.coefficient).toString();
  if (d.scale > 0) {
    digits = digits.padStart(d.scale + 1, "0");
    digits = `${digits.slice(0, -d.scale)}.${digits.slice(-d.scale)}`;
  }
  return negative && d.coefficient !== 0n ? `-${digits}` : digits;
}

/** Remove trailing fractional zeros (keeps at least `minScale` digits). */
export function normalize(d: Decimal, minScale = 0): Decimal {
  let { coefficient, scale } = d;
  while (scale > minScale && coefficient % 10n === 0n) {
    coefficient /= 10n;
    scale -= 1;
  }
  return { coefficient, scale };
}

function groupThousands(intDigits: string): string {
  return intDigits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

export interface FormatOptions {
  /** fixed number of fractional digits (rounds half-even / pads). */
  fractionDigits?: number;
  /** when fractionDigits is not set: trim trailing zeros but keep at least this many digits. */
  minFractionDigits?: number;
  /** when fractionDigits is not set: round to at most this many digits. */
  maxFractionDigits?: number;
  grouping?: boolean;
  /** "always" prefixes + for positive values (P&L). */
  signDisplay?: "auto" | "always";
  prefix?: string;
  suffix?: string;
}

export function formatDecimalValue(d: Decimal, opts: FormatOptions = {}): string {
  const { grouping = true, signDisplay = "auto", prefix = "", suffix = "" } = opts;
  let value: Decimal;
  if (opts.fractionDigits !== undefined) {
    value = roundToScale(d, opts.fractionDigits);
  } else {
    const maxF = opts.maxFractionDigits ?? Math.max(d.scale, opts.minFractionDigits ?? 0);
    value = normalize(roundToScale(d, Math.min(d.scale, maxF)), opts.minFractionDigits ?? 0);
    if (value.scale < (opts.minFractionDigits ?? 0)) value = rescale(value, opts.minFractionDigits ?? 0);
  }
  const negative = value.coefficient < 0n;
  const plain = toPlainString(negative ? negate(value) : value);
  const [intPart = "0", fracPart] = plain.split(".");
  const body = (grouping ? groupThousands(intPart) : intPart) + (fracPart !== undefined ? `.${fracPart}` : "");
  const signText = negative ? "-" : signDisplay === "always" && value.coefficient > 0n ? "+" : "";
  return `${signText}${prefix}${body}${suffix}`;
}

/** Format a decimal string. null/undefined -> "—"; anything unparsable (including JS numbers) -> "invalid". */
export function formatDecimal(value: unknown, opts: FormatOptions = {}): string {
  if (value === null || value === undefined || value === "") return EMPTY_PLACEHOLDER;
  const d = parseDecimal(value);
  return d ? formatDecimalValue(d, opts) : INVALID_PLACEHOLDER;
}

export function formatMoney(value: unknown, opts: { fractionDigits?: number; signDisplay?: "auto" | "always" } = {}): string {
  return formatDecimal(value, { fractionDigits: opts.fractionDigits ?? 2, prefix: "$", signDisplay: opts.signDisplay ?? "auto" });
}

/** Prices: at least 2 decimals, up to 6, trailing zeros trimmed beyond 2. */
export function formatPrice(value: unknown): string {
  return formatDecimal(value, { minFractionDigits: 2, maxFractionDigits: 6 });
}

/** Quantities: exact, trailing zeros trimmed. */
export function formatQty(value: unknown): string {
  return formatDecimal(value, { minFractionDigits: 0, maxFractionDigits: 8 });
}

export function formatBps(value: unknown, fractionDigits = 1): string {
  const text = formatDecimal(value, { fractionDigits });
  return text === EMPTY_PLACEHOLDER || text === INVALID_PLACEHOLDER ? text : `${text} bps`;
}

/** A value that is already a percentage ("12.5" -> "12.5%"). */
export function formatPercent(value: unknown, fractionDigits = 1): string {
  const text = formatDecimal(value, { fractionDigits });
  return text === EMPTY_PLACEHOLDER || text === INVALID_PLACEHOLDER ? text : `${text}%`;
}

/** A fraction ("0.0213" -> "2.13%"). Exact: multiplies the decimal by 100. */
export function formatFractionAsPercent(value: unknown, fractionDigits = 2, signDisplay: "auto" | "always" = "auto"): string {
  if (value === null || value === undefined || value === "") return EMPTY_PLACEHOLDER;
  const d = parseDecimal(value);
  if (!d) return INVALID_PLACEHOLDER;
  return formatDecimalValue(mul(d, { coefficient: 100n, scale: 0 }), { fractionDigits, suffix: "%", signDisplay });
}

/** Compare two decimal strings exactly; null when either is not a decimal. */
export function compareDecimalStrings(a: unknown, b: unknown): -1 | 0 | 1 | null {
  const x = parseDecimal(a);
  const y = parseDecimal(b);
  return x && y ? compare(x, y) : null;
}

export function decimalEquals(a: unknown, b: unknown): boolean {
  return compareDecimalStrings(a, b) === 0;
}

/** Sum decimal strings exactly; null if any is invalid. */
export function sumDecimalStrings(values: readonly unknown[]): string | null {
  let total: Decimal = { coefficient: 0n, scale: 0 };
  for (const v of values) {
    const d = parseDecimal(v);
    if (!d) return null;
    total = add(total, d);
  }
  return toPlainString(total);
}

/** Exact a - b as a decimal string; null if either is invalid. */
export function subtractDecimalStrings(a: unknown, b: unknown): string | null {
  const x = parseDecimal(a);
  const y = parseDecimal(b);
  return x && y ? toPlainString(sub(x, y)) : null;
}

/** Clamp a decimal-string percentage (0..100) for a progress bar width, as a CSS percent string. */
export function percentWidth(value: unknown): string {
  const d = parseDecimal(value);
  if (!d) return "0%";
  const zero: Decimal = { coefficient: 0n, scale: 0 };
  const hundred: Decimal = { coefficient: 100n, scale: 0 };
  const clamped = compare(d, zero) < 0 ? zero : compare(d, hundred) > 0 ? hundred : d;
  return `${toPlainString(roundToScale(clamped, 2))}%`;
}

/**
 * DISPLAY-ONLY conversion for chart coordinates (Lightweight Charts requires JS numbers).
 * Never use the result for accounting, comparisons shown to the operator, or totals.
 */
export function toChartNumber(value: unknown): number | null {
  const d = parseDecimal(value);
  if (!d) return null;
  const n = Number(toPlainString(d));
  return Number.isFinite(n) ? n : null;
}

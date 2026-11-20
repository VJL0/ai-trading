import { describe, expect, it } from "vitest";
import {
  compareDecimalStrings,
  div,
  EMPTY_PLACEHOLDER,
  formatBps,
  formatDecimal,
  formatFractionAsPercent,
  formatMoney,
  formatPercent,
  formatPrice,
  formatQty,
  INVALID_PLACEHOLDER,
  parseDecimal,
  percentWidth,
  roundToScale,
  subtractDecimalStrings,
  sumDecimalStrings,
  toChartNumber,
  toPlainString,
} from "./decimal";

describe("parseDecimal", () => {
  it("parses canonical decimal strings into an exact coefficient + scale", () => {
    expect(parseDecimal("123.4500")).toEqual({ coefficient: 1234500n, scale: 4 });
    expect(parseDecimal("-0.01")).toEqual({ coefficient: -1n, scale: 2 });
    expect(parseDecimal("1E+3")).toEqual({ coefficient: 1000n, scale: 0 });
    expect(parseDecimal("1.5e-3")).toEqual({ coefficient: 15n, scale: 4 });
  });

  it("rejects JS numbers (the float boundary) and malformed text", () => {
    expect(parseDecimal(0.1)).toBeNull();
    expect(parseDecimal(100)).toBeNull();
    expect(parseDecimal("")).toBeNull();
    expect(parseDecimal("1,000.00")).toBeNull();
    expect(parseDecimal("NaN")).toBeNull();
    expect(parseDecimal("Infinity")).toBeNull();
    expect(parseDecimal("0x10")).toBeNull();
    expect(parseDecimal(".")).toBeNull();
  });
});

describe("exact arithmetic (no binary floats)", () => {
  it("0.1 + 0.2 is exactly 0.3", () => {
    expect(sumDecimalStrings(["0.1", "0.2"])).toBe("0.3");
    expect(compareDecimalStrings(sumDecimalStrings(["0.1", "0.2"]), "0.3")).toBe(0);
  });

  it("keeps precision beyond Number.MAX_SAFE_INTEGER", () => {
    expect(sumDecimalStrings(["9007199254740993.01", "0.01"])).toBe("9007199254740993.02");
    expect(formatMoney("9007199254740993.015")).toBe("$9,007,199,254,740,993.02"); // half-even on digits
  });

  it("subtracts and compares exactly; invalid input yields null rather than a guess", () => {
    expect(subtractDecimalStrings("100004.12", "100000.00")).toBe("4.12");
    expect(sumDecimalStrings(["1", 2 as unknown as string])).toBeNull();
    expect(compareDecimalStrings("1.10", "1.1")).toBe(0);
    expect(compareDecimalStrings("-2", "1")).toBe(-1);
    expect(compareDecimalStrings("abc", "1")).toBeNull();
  });

  it("rounds half-even like Python's decimal default", () => {
    expect(toPlainString(roundToScale(parseDecimal("2.345")!, 2))).toBe("2.34");
    expect(toPlainString(roundToScale(parseDecimal("2.355")!, 2))).toBe("2.36");
    expect(toPlainString(roundToScale(parseDecimal("-2.345")!, 2))).toBe("-2.34");
    expect(toPlainString(roundToScale(parseDecimal("0.125")!, 2))).toBe("0.12");
  });

  it("divides with correct rounding (VWAP style)", () => {
    // (1*585.10 + 2*585.61) / 3 = 585.44
    expect(toPlainString(div(parseDecimal("1756.32")!, parseDecimal("3")!, 4)!)).toBe("585.4400");
    expect(toPlainString(div(parseDecimal("1")!, parseDecimal("3")!, 6)!)).toBe("0.333333");
    expect(toPlainString(div(parseDecimal("2")!, parseDecimal("3")!, 6)!)).toBe("0.666667");
    expect(div(parseDecimal("1")!, parseDecimal("0")!)).toBeNull();
  });
});

describe("formatting", () => {
  it("formats money with grouping, fixed 2 dp and explicit signs for P&L", () => {
    expect(formatMoney("100004.12")).toBe("$100,004.12");
    expect(formatMoney("4.13", { signDisplay: "always" })).toBe("+$4.13");
    expect(formatMoney("-1234.5")).toBe("-$1,234.50");
    expect(formatMoney("0", { signDisplay: "always" })).toBe("$0.00");
  });

  it("formats prices (2..6 dp) and quantities (trimmed) from strings", () => {
    expect(formatPrice("585.4433")).toBe("585.4433");
    expect(formatPrice("585.100000")).toBe("585.10");
    expect(formatPrice("12")).toBe("12.00");
    expect(formatQty("3.000000000")).toBe("3");
    expect(formatQty("0.5")).toBe("0.5");
  });

  it("formats bps and percentages", () => {
    expect(formatBps("1.75")).toBe("1.8 bps");
    expect(formatPercent("70.3")).toBe("70.3%");
    expect(formatFractionAsPercent("0.0213", 2, "always")).toBe("+2.13%");
    expect(formatFractionAsPercent("-0.0042")).toBe("-0.42%");
  });

  it("shows explicit placeholders for missing and invalid values (numbers are invalid)", () => {
    expect(formatMoney(null)).toBe(EMPTY_PLACEHOLDER);
    expect(formatMoney(undefined)).toBe(EMPTY_PLACEHOLDER);
    expect(formatMoney(12.5)).toBe(INVALID_PLACEHOLDER);
    expect(formatDecimal("1e999")).toBe(INVALID_PLACEHOLDER);
    expect(formatBps(null)).toBe(EMPTY_PLACEHOLDER);
  });

  it("clamps utilization widths without float parsing", () => {
    expect(percentWidth("70.3")).toBe("70.30%");
    expect(percentWidth("150")).toBe("100.00%");
    expect(percentWidth("-5")).toBe("0.00%");
    expect(percentWidth("junk")).toBe("0%");
  });

  it("toChartNumber is the only (display) float conversion and rejects non-strings", () => {
    expect(toChartNumber("584.20")).toBe(584.2);
    expect(toChartNumber(584.2)).toBeNull();
    expect(toChartNumber("x")).toBeNull();
  });
});

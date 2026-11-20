import { describe, expect, it } from "vitest";
import { ApiClient, ApiError, buildQuery, describeErrorBody } from "./client";
import { FakeFetch, FILLED_COID } from "../test/helpers";

describe("read-only API", () => {
  it("all operations are GET without credentials, tokens or bodies", async () => {
    const fake = new FakeFetch();
    const api = new ApiClient({ fetchImpl: fake.fetch });
    await Promise.all([api.status(), api.health(), api.activity(), api.activityCurrent(), api.performance(),
      api.sessions(), api.orders(), api.order(FILLED_COID), api.audit()]);
    expect(fake.requests).toHaveLength(9);
    for (const req of fake.requests) {
      expect(req.method).toBe("GET");
      expect(req.body).toBeNull();
      expect(req.headers).toEqual({accept: "application/json"});
    }
  });
  it("surfaces HTTP errors without retries", async () => {
    const fake = new FakeFetch({ "GET /api/v1/status": () => ({status: 503, body: {detail: "offline"}}) });
    await expect(new ApiClient({fetchImpl: fake.fetch}).status()).rejects.toMatchObject({kind: "http", status: 503, detail: "offline"});
    expect(fake.requests).toHaveLength(1);
  });
  it("rejects invalid JSON", async () => {
    const api = new ApiClient({fetchImpl: async () => new Response("bad json")});
    await expect(api.status()).rejects.toBeInstanceOf(ApiError);
  });
  it("reports network failure", async () => {
    const api = new ApiClient({fetchImpl: async () => {throw new Error("offline");}});
    await expect(api.health()).rejects.toMatchObject({kind: "network"});
  });
  it("encodes query values and describes validation errors", () => {
    expect(buildQuery({limit: 5, empty: null, text: "a&b"})).toBe("?limit=5&text=a%26b");
    expect(describeErrorBody({detail: [{loc: ["query", "limit"], msg: "too large"}]})).toBe("query.limit: too large");
  });
});

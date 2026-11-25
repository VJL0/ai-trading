# Local configuration

With the project MCP server connected, tell the AI “start trading.” Use [`.env.example`](../../.env.example)
as the reference for the git-ignored `.env` file. The operator-facing `LAT_TRADING_MODE` accepts `PAPER` or
`LIVE`. A Live selection is refused in this build.

To place Alpaca paper orders, set `LAT_TRADING_MODE=PAPER` and add `ALPACA_PAPER_KEY_ID` and
`ALPACA_PAPER_SECRET_KEY`. The key ID starts with `PK`. With the selection omitted, broker credentials alone
permit market-data reads. Paper without credentials uses internal simulation. The feed defaults to IEX with
paper credentials and synthetic data otherwise.

The credential lookup order is macOS Keychain, the process environment, then `.env`. To use Keychain, add
each variable as the account under the `local-ai-trader` service. The command prompts for the value.

```sh
security add-generic-password -s local-ai-trader -a ALPACA_PAPER_KEY_ID -w
security add-generic-password -s local-ai-trader -a ALPACA_PAPER_SECRET_KEY -w
```

The authenticated operator owns the active, versioned risk policy and any halt or resume decision.
`LAT_AGENT_MAX_PROPOSALS_PER_HOUR` optionally sets the direct-agent proposal ceiling. Sessions also pass the
policy, signed-intent and final execution gates on every order. Live execution remains compiled off.
Operator and agent tokens and intent-signing keys are generated under `var/<mode>/keys/`.

Restart the API and reconnect MCP after changing the local configuration. Existing advanced runtime
overrides continue to work, with explicit process variables taking precedence over `.env`.
For example, developers can set `LAT_API_PORT` for a local port conflict or
`LAT_PROJECT_ROOT` to an absolute path for an isolated workspace. Demo and replay are internal verification
paths. The start-session tool uses the quant workflow without in-app inference by default.

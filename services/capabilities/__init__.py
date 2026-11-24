"""Broker capability verification (read-only).

Answers "what can this Alpaca account actually trade, what data can it see, and what does *our* policy let it
execute?" from live, dated API responses instead of hard-coded assumptions. Four layers are kept separate:

* ``api_schema``      — the Alpaca API exposes the asset class / instrument (assets, option contracts).
* ``market_data``     — the account's data entitlement serves it (free IEX/SIP-historical, crypto, indicative options).
* ``account_permission`` — the account flags allow it (status, crypto_status, options level, shorting, margin).
* ``policy_execution`` — this application's risk policy + execution stack would authorize an order for it.

Research eligibility only needs ``market_data``; execution eligibility needs all four. Nothing here can place,
cancel or modify an order: the HTTP client refuses every non-GET request and every non-allowlisted host.
"""

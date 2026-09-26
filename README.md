# Magicpin Vera deterministic message engine

This submission adapter composes grounded merchant and customer messages from the supplied Vera contexts. It uses the existing deterministic decision and message pipeline; it makes no model or network calls.

## Architecture and safety

`bot.compose(category, merchant, trigger, customer=None)` accepts challenge dataset dictionaries or normalized context models. It places detached inputs in a temporary `ContextStore`, runs the decision engine at the fixed seed-replay time, then passes the plan to the existing message composer. The composer validates relationships and source facts. Unsupported or incomplete inputs remain non-send results; they are not filled with invented offers, dates, availability, consent, or customer history.

The challenge materials do not identify the authoritative canonical 30-pair list or define JSONL output for non-send results. No `submission.jsonl` is included pending that clarification.

## Run locally

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m flask --app src.api run --host 0.0.0.0 --port 8080
```

The Flask command is for local testing; deploy the HTTP service on a host that provides a public URL. No public URL is configured here.

Optional metadata environment variables are `VERA_TEAM_NAME`, `VERA_TEAM_MEMBERS` (JSON array), `VERA_MODEL`, `VERA_APPROACH`, `VERA_CONTACT_EMAIL`, `VERA_VERSION`, and `VERA_SUBMITTED_AT`. The service runs without them; unset identity values are returned blank.

# API settings

Choose a provider and model, enter its API key, then select **Save & Test**.
The connection becomes active only after the test succeeds. Reopening Settings
shows eight password dots when a key is configured for that provider and endpoint.
Leaving the field untouched reuses the session key. Selecting the field lets you
replace it. Changing only the model on the same connection can reuse its key;
changing the provider or endpoint requires its own key.

The dots are a fixed indicator, not a copy or fragment of the key. Keys are kept
in server memory and are not returned to the browser or written to browser storage.
On Windows, a configuration that passes Save & Test is also saved with user-scoped
DPAPI encryption in `.gwasmind/credentials/model.dpapi`. After a restart it is
loaded as configured but unverified; the page re-tests it once without sending a
key, and Settings opens only if that test fails or nothing was saved. Delete the
file to forget the key. A configured environment credential is handled separately
by the existing startup configuration and is never written to this file.

## Reasoning controls

Controls depend on the exact model and API endpoint, including the intermediary
when using a gateway. They do not depend on a manufacturer-wide thinking flag.

| Model capability | Settings shown |
| --- | --- |
| Independent switch and depth | Enable reasoning plus supported depth choices |
| Adjustable depth | Supported depths; None appears only when documented |
| Switch only | Enable reasoning, without a depth menu |
| Mandatory, fixed reasoning | Always on explanation, without editable controls |
| No configurable reasoning | Explanation, without editable controls |
| Capability not verified | Provider default explanation; no speculative parameters |

**Provider default** leaves computation controls to the provider. Available depth
values are model specific: Maximum, Extra high, and High are not interchangeable.
For models where `none` still means minimal reasoning, the menu says so and does
not call it Off. Choices are retained separately for each model and endpoint while
the settings page remains open. The active-session card shows saved settings.

**Refresh** retrieves the current model list without changing the active model.
OpenRouter capabilities are read from the same model-list response. They are bound
to the exact connection and model on Save & Test and remain unchanged for submitted
jobs even if a subsequent Refresh returns different capabilities. Unknown custom
IDs use a local capability lookup; this lookup does not contact the provider or
need an API key. Model availability is determined by the provider, so a curated
catalog entry does not guarantee that a retired model can still be called.

The connection test verifies credentials and the provider's model response. It
does not verify the quality of reasoning or every depth level. Chat and native
gene ranking use the same request adapter for the saved controls. Providers that
cannot disable reasoning use a supported minimal level for the lightweight
intent router. This does not change the user's saved analysis depth.

## Maintaining provider support

`src/gwasmind/provider_reasoning.py` defines exact provider/model capabilities and
their documented request fields. `provider_catalog.py` publishes those capabilities.
The web connection client and native synthesizer share the adapter, and the frontend
only displays the returned capability. Changes should include request-body tests
for both clients, plus settings save/reopen tests when adding a new UI behavior.

Current contracts were checked against official provider documentation and public
model metadata on 2026-09-13. Source URLs are retained in each capability. Providers
using unsupported native/cloud authentication remain unavailable in this dialog;
this change does not add new authentication protocols.

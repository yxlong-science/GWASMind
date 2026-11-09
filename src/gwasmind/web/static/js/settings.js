/**
 * CAUSE-inspired provider/model settings for the preserved GWASMind UI.
 *
 * API credentials are sent only to the loopback Python service and are never
 * persisted in browser storage.  The provider catalog is public metadata; a
 * small built-in fallback keeps DeepSeek configuration usable if an older
 * local service does not expose the catalog endpoint yet.
 */
(function () {
    "use strict";

    const CUSTOM_PROVIDER = "custom";
    const CUSTOM_MODEL = "__custom_model__";
    const credentialPrivacyNotice = "This key is used by your local GWASMind service and is not stored in the browser.";
    let ldlinkBusy = false;
    let ldlinkEditable = true;
    let ldlinkConfigured = false;
    let ldlinkRevision = 0;
    const defaults = {
        providerId: "deepseek",
        apiBase: "https://api.deepseek.com",
        model: "deepseek-v4-pro",
        thinkingMode: "enabled",
        reasoningEffort: "high"
    };
    const fallbackCatalog = {
        version: 1,
        default: {
            provider_id: defaults.providerId,
            model_id: defaults.model,
            base_url: defaults.apiBase,
            thinking_mode: defaults.thinkingMode,
            reasoning_effort: defaults.reasoningEffort
        },
        total_providers: 6,
        total_entries: 7,
        configurable_providers: 5,
        custom_available: true,
        providers: [
            fallbackProvider("deepseek", "DeepSeek", "official", defaults.apiBase,
                [fallbackModel("deepseek-v4-pro", "DeepSeek V4 Pro", true, true, 1000000, 384000),
                 fallbackModel("deepseek-v4-flash", "DeepSeek V4 Flash", false, true, 1000000, 384000)],
                "DeepSeek API Key"),
            fallbackProvider("openai", "OpenAI", "official", "https://api.openai.com/v1",
                [fallbackModel("gpt-5.4", "GPT-5.4", true, false)], "OpenAI API Key"),
            fallbackProvider("google", "Google Gemini", "official", "https://generativelanguage.googleapis.com/v1beta/openai",
                [fallbackModel("gemini-3.1-pro-preview", "Gemini 3.1 Pro", true, false)], "Gemini API Key"),
            fallbackProvider("mistral", "Mistral AI", "official", "https://api.mistral.ai/v1",
                [fallbackModel("mistral-medium-2604", "Mistral Medium 2604", true, false)], "Mistral API Key"),
            fallbackProvider("openrouter", "OpenRouter", "aggregator", "https://openrouter.ai/api/v1",
                [fallbackModel("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", true, false)], "OpenRouter API Key"),
            {
                id: "anthropic", name: "Anthropic", category: "official",
                protocol: "anthropic-messages", base_url: "https://api.anthropic.com",
                auth_label: "Anthropic API Key", configurable: false,
                disabled_reason: "Requires a native Anthropic Messages adapter.", models: []
            },
            {
                id: CUSTOM_PROVIDER, name: "Custom OpenAI-compatible", category: "custom",
                protocol: "openai-chat-completions", base_url: null, auth_label: "API Key",
                configurable: true, custom_base_url: true, allow_custom_model_id: true,
                capabilities: { chat_completions: true, thinking: false }, models: []
            }
        ]
    };

    let catalog = fallbackCatalog;
    let configured = false;
    let verified = false;
    let modalGeneration = 0;
    let activeConfiguration = null;
    let catalogFromServer = false;
    let lastRefreshAt = 0;
    let submissionInFlight = false;
    let modelRefreshInFlight = false;
    let formRevision = 0;
    let appliedFormRevision = 0;
    let settingsRefreshGeneration = 0;
    const publicCatalogPollIntervalMs = 2_000;
    const publicCatalogPollLimit = 30;
    let publicCatalogGeneration = 0;
    let publicCatalogTimer = null;
    const reasoningKinds = new Set(["toggle_effort", "effort", "toggle", "fixed", "none", "unknown"]);
    const reasoningChoices = new Map();
    const resolvedReasoning = new Map();
    let reasoningLookupGeneration = 0;
    let pendingReasoningKey = null;
    let settingsScrollLock = null;
    const credentialMask = "\u2022".repeat(8);
    let credentialMasked = false;
    let credentialEdited = false;

    function formEndpoint() {
        const provider = selectedProvider();
        const override = byId("base-url-override");
        const base = byId("api-base-url");
        return {
            providerId: provider && provider.id !== CUSTOM_PROVIDER && !(override && override.checked)
                ? provider.id : CUSTOM_PROVIDER,
            baseUrl: (base ? base.value.trim() : "").replace(/\/+$/, "")
        };
    }

    function retainedCredentialMatches() {
        if (!configured || !activeConfiguration) return false;
        const endpoint = formEndpoint();
        return endpoint.providerId === activeConfiguration.providerId
            && endpoint.baseUrl === activeConfiguration.baseUrl.replace(/\/+$/, "");
    }

    function clearCredential() {
        const key = byId("api-key");
        if (key) key.value = "";
        credentialMasked = false;
        credentialEdited = false;
    }

    function syncCredentialIndicator() {
        const key = byId("api-key");
        if (!key) return;
        const retained = retainedCredentialMatches();
        key.placeholder = retained ? "Session key saved; leave blank to keep it" : "Paste your API key";
        const status = byId("credential-session-status");
        if (status) {
            status.hidden = !retained;
            status.textContent = "API key is set for this connection. Select the field to replace it.";
        }
        if (credentialMasked && !retained) clearCredential();
        if (retained && !credentialEdited && !key.value) {
            key.value = credentialMask;
            credentialMasked = true;
        }
    }

    function beginCredentialEdit() {
        if (!credentialMasked) return;
        const key = byId("api-key");
        if (key) key.value = "";
        credentialMasked = false;
        credentialEdited = true;
        formRevision += 1;
    }

    function enteredCredential() {
        const key = byId("api-key");
        if (!key || /^\u2022+$/.test(key.value) || (credentialMasked && key.value === credentialMask)) return "";
        return key.value.trim();
    }

    function fallbackModel(id, name, recommended, thinking, contextWindow, maxOutput) {
        return {
            id,
            name,
            recommended: Boolean(recommended),
            native_reasoning: Boolean(thinking),
            thinking: {
                supported: Boolean(thinking),
                default_mode: thinking ? "enabled" : "disabled",
                efforts: thinking ? ["low", "high", "max"] : []
            },
            reasoning: thinking ? {
                kind: "toggle_effort", efforts: ["low", "high", "max"],
                default_mode: "enabled", default_effort: "high", can_disable: true
            } : { kind: "unknown", efforts: [], default_mode: "auto", default_effort: "default" },
            context_window: contextWindow || null,
            max_output_tokens: maxOutput || null
        };
    }

    function fallbackProvider(id, name, category, baseUrl, models, authLabel) {
        return {
            id,
            name,
            category,
            protocol: "openai-chat-completions",
            base_url: baseUrl,
            auth_label: authLabel,
            configurable: true,
            custom_base_url: false,
            allow_custom_model_id: true,
            capabilities: {
                chat_completions: true,
                thinking: models.some((model) => model.thinking.supported)
            },
            models
        };
    }

    function byId(id) {
        return document.getElementById(id);
    }

    function providers() {
        return Array.isArray(catalog.providers) ? catalog.providers : [];
    }

    function configurableProviders() {
        return providers().filter((provider) => (
            provider
            && (provider.configurable || provider.id === CUSTOM_PROVIDER)
        ));
    }

    function providerById(id) {
        return providers().find((provider) => provider.id === id) || null;
    }

    function selectedProvider() {
        const select = byId("model-provider");
        return providerById(select && select.value ? select.value : defaults.providerId)
            || providerById(defaults.providerId)
            || configurableProviders()[0]
            || null;
    }

    function selectedCatalogModel(provider) {
        const select = byId("model-catalog-select");
        if (!provider || !select || select.value === CUSTOM_MODEL) return null;
        return (provider.models || []).find((model) => model.id === select.value) || null;
    }

    function normalizeCatalog(value) {
        if (!value || !Array.isArray(value.providers) || !value.providers.length) {
            throw new Error("Provider catalog is invalid");
        }
        const normalized = {
            ...value,
            default: value.default && typeof value.default === "object"
                ? value.default
                : fallbackCatalog.default,
            providers: value.providers.filter((provider) => provider && typeof provider === "object")
        };
        if (!normalized.providers.some((provider) => provider.id === CUSTOM_PROVIDER)) {
            normalized.providers.push(fallbackCatalog.providers.find((provider) => provider.id === CUSTOM_PROVIDER));
        }
        return normalized;
    }

    async function loadCatalog() {
        const response = await fetch("/api/v1/model-catalog", { cache: "no-store" });
        if (!response.ok) throw new Error("Provider catalog is unavailable");
        return normalizeCatalog(await response.json());
    }

    async function loadSettings() {
        const response = await fetch("/api/v1/model-settings", { cache: "no-store" });
        if (!response.ok) throw new Error("Session model settings are unavailable");
        return response.json();
    }

    function applySessionSettings(value) {
        const model = value.models && value.models.openai ? value.models.openai : {};
        configured = Boolean(model.configured ?? model.available);
        verified = Boolean(model.verified ?? model.available);
        const providerId = model.provider_id || inferProviderId(model.base_url);
        activeConfiguration = {
            providerId,
            baseUrl: model.base_url || defaults.apiBase,
            modelId: model.model_id || defaults.model,
            thinkingMode: model.thinking_mode || defaults.thinkingMode,
            reasoningEffort: model.reasoning_effort || defaults.reasoningEffort,
            reasoning: model.reasoning || null
        };
        if (model.reasoning) resolvedReasoning.set(reasoningKey(activeConfiguration), normalizeReasoning(model.reasoning));
        setLoadStatus("settings-load-status", true, "");
        return value;
    }

    async function refreshSettings() {
        const publicGeneration = stopPublicCatalogPolling();
        const generation = ++settingsRefreshGeneration;
        const revision = formRevision;
        const catalogResult = await Promise.allSettled([loadCatalog(), loadSettings()]);
        if (generation !== settingsRefreshGeneration) return configured;
        if (catalogResult[0].status === "rejected") {
            catalog = fallbackCatalog;
            catalogFromServer = false;
            const retry = byId("catalog-retry");
            if (retry) retry.hidden = false;
            setLoadStatus(
                "catalog-status",
                false,
                "Using the built-in provider list."
            );
        } else {
            catalog = catalogResult[0].value;
            catalogFromServer = true;
            setLoadStatus("catalog-status", true, "");
        }
        const preserveEdits = revision !== formRevision;
        if (!preserveEdits) renderProviderOptions();
        if (catalogResult[1].status === "rejected") {
            configured = false;
            verified = false;
            activeConfiguration = null;
            setLoadStatus(
                "settings-load-status",
                false,
                "Session settings could not be loaded."
            );
            if (!preserveEdits) applyConfiguration(catalog.default || fallbackCatalog.default);
        } else {
            applySessionSettings(catalogResult[1].value);
            if (!preserveEdits && activeConfiguration) applyConfiguration(activeConfiguration);
        }
        if (!preserveEdits) appliedFormRevision = formRevision;
        updateSessionCard();
        syncCredentialIndicator();
        lastRefreshAt = Date.now();
        renderPublicCatalogStatus();
        if (publicGeneration === publicCatalogGeneration) schedulePublicCatalogPoll(publicGeneration, 0);
        return configured;
    }

    function stopPublicCatalogPolling() {
        publicCatalogGeneration += 1;
        if (publicCatalogTimer !== null) window.clearTimeout(publicCatalogTimer);
        publicCatalogTimer = null;
        return publicCatalogGeneration;
    }

    function renderPublicCatalogStatus(stateOverride) {
        const state = stateOverride || (catalog.public_refresh && catalog.public_refresh.state);
        if (!state) return;
        const provider = selectedProvider();
        const source = provider && provider.catalog_source;
        const updated = source && source.updated_at || catalog.public_refresh && catalog.public_refresh.updated_at;
        const date = updated ? new Date(updated) : null;
        const suffix = source && source.name ? ` · ${source.name}` : "";
        const timestamp = date && !Number.isNaN(date.getTime()) ? ` · ${date.toLocaleDateString()}` : "";
        const messages = {
            idle: "Using the saved public LLM list",
            refreshing: "Updating public LLM lists… No API key is used",
            ready: "Public LLM lists updated",
            partial: "Some public LLM lists could not update; saved entries retained",
            offline: "Public LLM lists could not update; saved entries retained",
            waiting: "Public LLM lists are still updating; reopen Settings to check"
        };
        if (!messages[state]) return;
        setLoadStatus("catalog-status", false, `${messages[state]}${suffix}${timestamp}.`);
        const retry = byId("catalog-retry");
        if (retry) retry.hidden = !["offline", "partial", "waiting"].includes(state);
    }

    function applyPublicCatalog(value) {
        // Only options and public metadata change. Do not reapply session
        // settings or reset a credential, endpoint override, or unsaved choice.
        const previousProvider = selectedProvider();
        const previousModel = currentModelId();
        const input = byId("model-name");
        const inputValue = input && input.value;
        const capability = currentReasoning();
        if (capability.kind !== "unknown") resolvedReasoning.set(reasoningKey(), capability);
        catalog = value;
        if (previousProvider && !providerById(previousProvider.id)) catalog.providers.push(previousProvider);
        catalogFromServer = true;
        renderProviderOptions(false);
        renderModelOptions(selectedProvider(), previousModel);
        if (input && byId("model-catalog-select") && byId("model-catalog-select").value === CUSTOM_MODEL) {
            input.value = inputValue;
        }
        setLoadStatus("catalog-status", true, "");
        renderPublicCatalogStatus();
    }

    function schedulePublicCatalogPoll(generation, attempts) {
        if (generation !== publicCatalogGeneration
                || !catalog.public_refresh || catalog.public_refresh.state !== "refreshing") return;
        if (attempts >= publicCatalogPollLimit) {
            renderPublicCatalogStatus("waiting");
            return;
        }
        publicCatalogTimer = window.setTimeout(async () => {
            publicCatalogTimer = null;
            try {
                const value = await loadCatalog();
                if (generation !== publicCatalogGeneration) return;
                applyPublicCatalog(value);
                schedulePublicCatalogPoll(generation, attempts + 1);
            } catch (_) {
                if (generation === publicCatalogGeneration) renderPublicCatalogStatus("offline");
            }
        }, publicCatalogPollIntervalMs);
    }

    async function refreshPublicCatalog() {
        const generation = stopPublicCatalogPolling();
        try {
            const value = await loadCatalog();
            if (generation !== publicCatalogGeneration) return;
            applyPublicCatalog(value);
            schedulePublicCatalogPoll(generation, 0);
        } catch (_) {
            if (generation === publicCatalogGeneration) renderPublicCatalogStatus("offline");
        }
    }

    function setLoadStatus(id, success, text) {
        const element = byId(id);
        if (!element) return;
        if (success) {
            element.hidden = true;
            return;
        }
        const textElement = id === "catalog-status"
            ? byId("catalog-status-text")
            : element.querySelector && element.querySelector("span");
        if (textElement && text) textElement.textContent = text;
        element.hidden = false;
    }

    function renderProviderOptions(renderSelection = true) {
        const select = byId("model-provider");
        if (!select) return;
        const previous = select.value || (activeConfiguration && activeConfiguration.providerId) || defaults.providerId;
        select.replaceChildren();
        configurableProviders().forEach((provider) => {
            const option = document.createElement("option");
            option.value = provider.id;
            option.textContent = provider.name || provider.id;
            select.appendChild(option);
        });
        const desired = configurableProviders().find((provider) => provider.id === previous);
        const fallback = configurableProviders().find((provider) => provider.id === defaults.providerId)
            || configurableProviders()[0];
        select.value = desired ? desired.id : (fallback ? fallback.id : "");
        if (renderSelection) renderProviderSelection();
    }

    function inferProviderId(baseUrl) {
        if (!baseUrl) return defaults.providerId;
        const normalized = String(baseUrl).replace(/\/$/, "").toLowerCase();
        const match = providers().find((provider) => {
            if (!provider.configurable || !provider.base_url) return false;
            const expected = String(provider.base_url).replace(/\/$/, "").toLowerCase();
            return normalized === expected || normalized === `${expected}/chat/completions`;
        });
        return match ? match.id : CUSTOM_PROVIDER;
    }

    function applyConfiguration(value) {
        if (!value) return;
        reasoningChoices.set(reasoningKey({
            providerId: value.providerId || value.provider_id || inferProviderId(value.baseUrl || value.base_url),
            baseUrl: value.baseUrl || value.base_url || defaults.apiBase,
            modelId: value.modelId || value.model_id || defaults.model
        }), {
            mode: value.thinkingMode || value.thinking_mode || defaults.thinkingMode,
            effort: value.reasoningEffort || value.reasoning_effort || defaults.reasoningEffort
        });
        const providerSelect = byId("model-provider");
        const inferred = value.providerId || value.provider_id || inferProviderId(value.baseUrl || value.base_url);
        if (providerSelect && providerById(inferred) && providerById(inferred).configurable) {
            providerSelect.value = inferred;
        }
        renderProviderSelection(value.modelId || value.model_id || defaults.model);
        const base = byId("api-base-url");
        if (base && (value.baseUrl || value.base_url)) base.value = value.baseUrl || value.base_url;
        syncThinkingControls();
        syncCredentialIndicator();
        resolveModelReasoning();
    }

    function renderProviderSelection(preferredModelId) {
        const provider = selectedProvider();
        if (!provider) return;
        const keyLabel = byId("api-key-label");
        if (keyLabel) keyLabel.textContent = `${provider.auth_label || provider.name + " API Key"} *`;
        const base = byId("api-base-url");
        const override = byId("base-url-override");
        const overrideRow = byId("base-url-override-row");
        const advanced = byId("advanced-model-settings");
        const custom = provider.id === CUSTOM_PROVIDER;
        if (override) {
            override.checked = custom;
            override.disabled = custom;
        }
        if (overrideRow) overrideRow.hidden = custom;
        if (advanced && custom) advanced.open = true;
        if (base) {
            if (!custom && provider.base_url) base.value = provider.base_url;
            else if (custom && (!base.value || inferProviderId(base.value) !== CUSTOM_PROVIDER)) base.value = "";
            base.readOnly = !custom;
        }
        updateCustomEndpointWarning();
        renderModelOptions(provider, preferredModelId);
        renderPublicCatalogStatus();
    }

    function defaultModelId(provider) {
        const models = provider && Array.isArray(provider.models) ? provider.models : [];
        const recommended = models.find((model) => model.recommended);
        return (recommended || models[0] || {}).id || "";
    }

    function updateCustomEndpointWarning() {
        const warning = byId("custom-endpoint-warning");
        if (!warning) return;
        const provider = selectedProvider();
        const override = byId("base-url-override");
        warning.hidden = !(
            (provider && provider.id === CUSTOM_PROVIDER)
            || (override && override.checked)
        );
    }

    function renderModelOptions(provider, preferredModelId) {
        const select = byId("model-catalog-select");
        const customGroup = byId("model-custom-group");
        const modelInput = byId("model-name");
        if (!select) {
            if (modelInput && preferredModelId) modelInput.value = preferredModelId;
            syncThinkingControls();
            return;
        }
        const previous = preferredModelId !== undefined
            ? preferredModelId
            : (select.value || (activeConfiguration && activeConfiguration.modelId));
        select.replaceChildren();
        const models = [...(provider.models || [])].sort((left, right) => Number(right.recommended) - Number(left.recommended));
        models.forEach((model) => {
            const option = document.createElement("option");
            option.value = model.id;
            option.textContent = model.name || model.id;
            select.appendChild(option);
        });
        const customOption = document.createElement("option");
        customOption.value = CUSTOM_MODEL;
        customOption.textContent = "Use a custom LLM ID…";
        select.appendChild(customOption);
        const known = models.find((model) => model.id === previous);
        select.value = known ? known.id : (models.length && !previous ? models[0].id : CUSTOM_MODEL);
        const customSelected = select.value === CUSTOM_MODEL;
        if (customGroup) customGroup.hidden = !customSelected;
        if (modelInput) {
            modelInput.value = customSelected ? (known ? known.id : previous || "") : select.value;
        }
        renderModelCapabilities(provider);
        syncThinkingControls();
    }

    function currentModelId() {
        const select = byId("model-catalog-select");
        const input = byId("model-name");
        if (!select || select.value === CUSTOM_MODEL) return input ? input.value.trim() : defaults.model;
        return select.value;
    }

    function renderModelCapabilities(provider) {
        const container = byId("model-capabilities");
        const model = selectedCatalogModel(provider);
        if (container) {
            container.replaceChildren();
            const badges = [];
            if (model && model.context_window) badges.push(`${Number(model.context_window).toLocaleString()} context`);
            if (model && model.max_output_tokens) badges.push(`${Number(model.max_output_tokens).toLocaleString()} max output`);
            const kind = currentReasoning().kind;
            if (kind === "toggle_effort" || kind === "effort") badges.push("Reasoning depth");
            else if (kind === "toggle" || kind === "fixed") badges.push("Reasoning");
            if (provider.allow_custom_model_id !== false) badges.push("Custom LLM IDs supported");
            badges.forEach((label) => {
                const badge = document.createElement("span");
                badge.className = `model-capability${label === "Reasoning depth" ? " thinking" : ""}`;
                badge.textContent = label;
                container.appendChild(badge);
            });
        }
        const budget = byId("model-output-budget");
        const maxOutput = model && model.max_output_tokens;
        if (budget) budget.value = maxOutput ? `${Number(maxOutput).toLocaleString()} tokens` : "Provider-defined";
        const budgetHelp = byId("model-output-budget-help");
        if (budgetHelp) budgetHelp.textContent = maxOutput
            ? "Provider maximum. GWASMind requests the highest supported analysis budget."
            : "The provider controls this limit; GWASMind retains its response validation gates.";
    }

    function reasoningKey(value) {
        const endpoint = value || { ...formEndpoint(), modelId: currentModelId() };
        return JSON.stringify([endpoint.providerId, (endpoint.baseUrl || "").replace(/\/+$/, ""), endpoint.modelId]);
    }

    function normalizeReasoning(value) {
        if (!value || !reasoningKinds.has(value.kind)) return {
            kind: "unknown", efforts: [], default_mode: "auto", default_effort: "default",
            note: "Reasoning controls are not confirmed for this model. The provider's settings apply."
        };
        return { ...value, efforts: Array.isArray(value.efforts)
            ? value.efforts.filter(effort => typeof effort === "string" && /^[a-z_]{1,24}$/.test(effort)) : [] };
    }

    function currentReasoning() {
        const key = reasoningKey();
        if (resolvedReasoning.has(key)) return resolvedReasoning.get(key);
        const provider = selectedProvider();
        const endpoint = formEndpoint();
        const expectedBase = provider && String(provider.base_url || "").replace(/\/+$/, "");
        if (!provider || endpoint.providerId !== provider.id || !expectedBase
                || endpoint.baseUrl !== expectedBase) return normalizeReasoning(null);
        const model = (provider.models || []).find(entry => entry.id === currentModelId());
        if (model && model.reasoning) return normalizeReasoning(model.reasoning);
        // Older local catalogs are supported only for these explicitly known IDs.
        if (provider.id === "deepseek" && ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-flash"].includes(currentModelId())) {
            if (model && model.thinking && model.thinking.supported === false) return normalizeReasoning(null);
            return normalizeReasoning({ kind: "toggle_effort", efforts: ["low", "high", "max"],
                default_mode: "enabled", default_effort: "high", can_disable: true });
        }
        return normalizeReasoning(null);
    }

    function reasoningChoice(capability = currentReasoning()) {
        const remembered = reasoningChoices.get(reasoningKey()) || {};
        const effort = remembered.effort === "default" || capability.efforts.includes(remembered.effort) ? remembered.effort
            : capability.default_effort === "default" || capability.efforts.includes(capability.default_effort) ? capability.default_effort
            : capability.efforts[0] || "default";
        return { effort, mode: remembered.mode || capability.default_mode || "auto" };
    }

    function effortLabel(value, capability) {
        if (value === "none" && capability && capability.can_disable === false) return "None (minimal reasoning)";
        return ({ none: "None", minimal: "Minimal", low: "Low", medium: "Medium", high: "High",
            max: "Maximum", xhigh: "Extra high", default: "Provider default" })[value]
            || value.replace(/_/g, " ").replace(/^./, character => character.toUpperCase());
    }

    async function resolveModelReasoning() {
        const key = reasoningKey();
        if (currentReasoning().kind !== "unknown" || !currentModelId()
                || resolvedReasoning.has(key) || pendingReasoningKey === key) return;
        const endpoint = formEndpoint();
        if (!endpoint.baseUrl) return;
        const generation = ++reasoningLookupGeneration;
        const modal = modalGeneration;
        pendingReasoningKey = key;
        syncThinkingControls();
        try {
            const response = await fetch("/api/v1/model-reasoning", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ provider_id: endpoint.providerId, api_base_url: endpoint.baseUrl, model: currentModelId() })
            });
            const value = await readJson(response);
            if (generation !== reasoningLookupGeneration || key !== reasoningKey() || modal !== modalGeneration) return;
            resolvedReasoning.set(key, normalizeReasoning(response.ok && value.success ? value.reasoning : null));
        } catch (_error) {
            if (generation === reasoningLookupGeneration && key === reasoningKey()) resolvedReasoning.set(key, normalizeReasoning(null));
        } finally {
            if (generation === reasoningLookupGeneration) {
                pendingReasoningKey = null;
                if (key === reasoningKey() && modal === modalGeneration) syncThinkingControls();
            }
        }
    }

    function syncThinkingControls() {
        const panel = byId("thinking-options");
        const effort = byId("model-reasoning-effort");
        const toggle = byId("model-thinking");
        const toggleRow = byId("model-thinking-row");
        const effortGroup = byId("model-reasoning-effort-group");
        const help = byId("model-reasoning-help");
        const capability = currentReasoning();
        const choice = reasoningChoice(capability);
        const hasToggle = ["toggle_effort", "toggle"].includes(capability.kind);
        const hasEffort = ["toggle_effort", "effort"].includes(capability.kind) && capability.efforts.length > 0;
        const effectiveMode = choice.mode === "auto" ? capability.default_mode : choice.mode;
        const enabled = capability.kind === "effort" ? choice.effort !== "none" || capability.can_disable === false
            : effectiveMode !== "disabled";
        if (panel) panel.hidden = false;
        if (toggleRow) toggleRow.hidden = !hasToggle;
        if (toggle) {
            toggle.checked = enabled;
            toggle.disabled = !hasToggle;
            toggle.indeterminate = hasToggle && choice.mode === "auto" && (!capability.default_mode || capability.default_mode === "auto");
        }
        if (effortGroup) effortGroup.hidden = !hasEffort;
        if (effort) {
            const options = hasEffort ? [["default", "Provider default"], ...capability.efforts.map(value => [value, effortLabel(value, capability)])] : [];
            if (typeof effort.replaceChildren === "function") {
                effort.replaceChildren();
                for (const [value, label] of options) {
                    const option = document.createElement("option");
                    option.value = value;
                    option.textContent = label;
                    effort.appendChild(option);
                }
            }
            effort.disabled = !hasEffort || (hasToggle && !enabled);
            effort.value = hasEffort ? choice.effort : "";
        }
        if (help) help.textContent = capability.kind === "none" ? "This LLM does not support reasoning."
            : capability.kind === "fixed" ? "Reasoning is always on for this LLM; there is no adjustable depth."
            : capability.kind === "unknown" ? pendingReasoningKey === reasoningKey()
                ? "Checking the available reasoning controls for this LLM…"
                : capability.note || "Reasoning controls are not confirmed for this LLM. The provider's settings apply."
            : hasToggle && choice.mode === "auto" && capability.default_mode === "auto"
                ? "The provider chooses whether reasoning is enabled. Use the switch to override it."
                    + (hasEffort && choice.effort === "default" ? " The provider also chooses the reasoning depth." : "")
            : !enabled ? "Reasoning is off." + (hasToggle ? " Turn it on to use the selected depth." : " Select another depth to enable it.")
            : hasEffort && choice.effort === "default" ? "The provider chooses the reasoning depth. Select a level to override it."
            : capability.kind === "toggle" ? "Reasoning is on. This LLM does not expose an adjustable depth."
            : capability.note || "Higher reasoning effort can take longer and use more tokens.";
        if (help) help.textContent = help.textContent.replace(/\bmodels\b/gi, 'LLMs').replace(/\bmodel\b/gi, 'LLM');
    }

    function extractRemoteModels(value) {
        const candidates = [
            value && value.models,
            value && value.model_ids,
            value && value.data,
            value && value.data && value.data.models,
            value && value.models && value.models.data,
            value && value.result && value.result.models
        ];
        const rows = candidates.find((candidate) => Array.isArray(candidate));
        if (!rows) throw new Error("The provider did not return a model list.");
        const capabilities = value && value.reasoning_capabilities || {};
        return rows.map((entry) => {
            if (typeof entry === "string") return { id: entry.trim(), reasoning: capabilities[entry.trim()] };
            if (!entry || typeof entry !== "object") return null;
            const id = String(entry.id || entry.model_id || entry.model || "").trim();
            return id ? { ...entry, id, reasoning: capabilities[id] } : null;
        }).filter((entry) => entry && entry.id);
    }

    function mergeRemoteModels(provider, remoteModels) {
        const curated = new Map((provider.models || []).map((model) => [model.id, model]));
        const seen = new Set();
        return remoteModels.reduce((models, remote) => {
            if (seen.has(remote.id)) return models;
            seen.add(remote.id);
            const fallbackProvider = fallbackCatalog.providers.find((entry) => entry.id === provider.id);
            const metadata = curated.get(remote.id)
                || ((fallbackProvider && fallbackProvider.models) || []).find((entry) => entry.id === remote.id)
                || {};
            models.push({
                ...remote,
                ...metadata,
                id: remote.id,
                name: metadata.name || remote.name || remote.id,
                recommended: Boolean(metadata.recommended || remote.recommended),
                reasoning: remote.reasoning || metadata.reasoning,
                thinking: {
                    ...(remote.thinking && typeof remote.thinking === "object" ? remote.thinking : {}),
                    ...(metadata.thinking && typeof metadata.thinking === "object" ? metadata.thinking : {})
                }
            });
            return models;
        }, []);
    }

    function conciseError(error, fallback) {
        const text = String((error && error.message) || fallback || "Request failed")
            .replace(/\s+/g, " ")
            .trim();
        return text.length > 180 ? `${text.slice(0, 177)}…` : text;
    }

    async function readJson(response) {
        try {
            return await response.json();
        } catch (_error) {
            return {};
        }
    }

    async function refreshProviderModels() {
        if (modelRefreshInFlight || submissionInFlight) return;
        stopPublicCatalogPolling();
        const provider = selectedProvider();
        const base = byId("api-base-url");
        const key = byId("api-key");
        const override = byId("base-url-override");
        const button = byId("refresh-models-btn");
        if (!provider) {
            showStatus("error", "Choose a provider first.");
            return;
        }
        const apiBaseUrl = base ? base.value.trim() : "";
        const generation = modalGeneration;
        const revision = formRevision;
        if (!apiBaseUrl) {
            const advanced = byId("advanced-model-settings");
            if (advanced) advanced.open = true;
            showStatus("error", "Enter the API URL first.");
            return;
        }
        modelRefreshInFlight = true;
        if (button) {
            button.disabled = true;
            button.textContent = "Refreshing…";
        }
        showStatus("info", "Refreshing models…");
        try {
            const response = await fetch("/api/v1/model-list", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    provider_id: provider.id === CUSTOM_PROVIDER || (override && override.checked)
                        ? CUSTOM_PROVIDER
                        : provider.id,
                    api_base_url: apiBaseUrl,
                    api_key: enteredCredential()
                })
            });
            const result = await readJson(response);
            if (generation !== modalGeneration || revision !== formRevision
                    || selectedProvider() !== provider || (base && base.value.trim() !== apiBaseUrl)) return;
            if (!response.ok || result.success === false) {
                throw new Error(result.error || "Could not refresh models.");
            }
            const remoteModels = extractRemoteModels(result);
            if (!remoteModels.length) throw new Error("No models were returned.");
            const previous = currentModelId();
            provider.models = mergeRemoteModels(provider, remoteModels);
            for (const model of provider.models) {
                if (model.reasoning) resolvedReasoning.set(reasoningKey({ ...formEndpoint(), modelId: model.id }), normalizeReasoning(model.reasoning));
            }
            renderModelOptions(provider, previous);
            resolveModelReasoning();
            const suffix = result.truncated ? " (limited list)" : "";
            showStatus("success", `${provider.models.length} models loaded${suffix}. Save & Test to apply changes.`);
        } catch (error) {
            if (generation === modalGeneration && revision === formRevision) {
                showStatus("error", conciseError(error, "Could not refresh models."));
            }
        } finally {
            if (button) {
                button.disabled = false;
                button.textContent = "↻ Refresh";
            }
            modelRefreshInFlight = false;
        }
    }

    function readForm() {
        const provider = selectedProvider();
        const override = byId("base-url-override");
        const base = byId("api-base-url");
        const capability = currentReasoning();
        const choice = reasoningChoice(capability);
        let mode = "auto", effort = "default";
        if (["toggle_effort", "toggle"].includes(capability.kind)) {
            mode = ["auto", "enabled", "disabled"].includes(choice.mode) ? choice.mode : "auto";
            effort = capability.kind === "toggle_effort" ? choice.effort : "default";
        } else if (capability.kind === "effort") {
            effort = choice.effort;
            mode = effort === "default" ? "auto" : effort === "none" && capability.can_disable !== false ? "disabled" : "enabled";
        } else if (capability.kind === "none") mode = "disabled";
        const customEndpoint = provider && (provider.id === CUSTOM_PROVIDER || (override && override.checked));
        return {
            provider_id: customEndpoint ? CUSTOM_PROVIDER : (provider ? provider.id : CUSTOM_PROVIDER),
            api_base_url: base ? base.value.trim() : defaults.apiBase,
            api_key: enteredCredential(),
            model: currentModelId(),
            thinking_mode: mode,
            reasoning_effort: effort
        };
    }

    function configurationChanged(value) {
        if (!activeConfiguration) return true;
        return value.provider_id !== activeConfiguration.providerId
            || value.api_base_url.replace(/\/+$/, "") !== activeConfiguration.baseUrl.replace(/\/+$/, "");
    }

    function validateForm(value) {
        const provider = providerById(value.provider_id);
        if (!provider || !provider.configurable) {
            showStatus("error", provider && provider.disabled_reason ? provider.disabled_reason : "Choose a configurable provider.");
            return false;
        }
        if (!value.api_base_url || !value.model) {
            showStatus("error", "Choose a model and provide its API Base URL.");
            return false;
        }
        if (!value.api_key && (!configured || configurationChanged(value))) {
            showStatus("error", "Enter an API key for the selected provider in this session.");
            return false;
        }
        return true;
    }

    async function submitSettings(endpoint) {
        const value = readForm();
        const submittedReasoning = currentReasoning();
        if (!validateForm(value)) return null;
        settingsRefreshGeneration += 1;
        const revision = formRevision;
        const generation = modalGeneration;
        let response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(value)
        });
        let result = await readJson(response);
        if (
            !response.ok
            && typeof result.error === "string"
            && result.error.includes("unknown model setting fields")
            && result.error.includes("provider_id")
        ) {
            const legacyValue = { ...value };
            delete legacyValue.provider_id;
            response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(legacyValue)
            });
            result = await readJson(response);
        }
        if (!response.ok || !result.success) throw new Error(result.error || "Configuration was rejected");
        const model = result.model_status && result.model_status.models
            ? result.model_status.models.openai || {}
            : {};
        configured = Boolean(model.configured ?? true);
        verified = Boolean(model.verified ?? model.available);
        activeConfiguration = {
            providerId: model.provider_id || value.provider_id,
            baseUrl: model.base_url || value.api_base_url,
            modelId: model.model_id || value.model,
            thinkingMode: model.thinking_mode || value.thinking_mode,
            reasoningEffort: model.reasoning_effort || value.reasoning_effort,
            reasoning: model.reasoning || submittedReasoning
        };
        if (model.reasoning) resolvedReasoning.set(reasoningKey(activeConfiguration), normalizeReasoning(model.reasoning));
        const key = byId("api-key");
        if (revision === formRevision && generation === modalGeneration) appliedFormRevision = revision;
        if (key && revision === formRevision && generation === modalGeneration) {
            clearCredential();
            syncCredentialIndicator();
        }
        updateSessionCard();
        return result;
    }

    async function testConnection() {
        if (submissionInFlight) return;
        submissionInFlight = true;
        const generation = modalGeneration;
        const revision = formRevision;
        const button = byId("test-btn");
        const refreshButton = byId("refresh-models-btn");
        showStatus("info", "Testing connection…");
        if (button) {
            button.disabled = true;
            button.textContent = "Saving & Testing…";
        }
        if (refreshButton) refreshButton.disabled = true;
        try {
            const result = await submitSettings("/api/test-llm-connection");
            if (!result) return;
            if (generation === modalGeneration && revision === formRevision) {
                showStatus("success", result.response || "Connection verified.");
                closeSettings();
            }
            if (typeof window.addMessage === "function") {
                window.addMessage("system", "LLM settings are active for this local session");
            }
        } catch (error) {
            if (generation === modalGeneration && revision === formRevision) {
                showStatus("error", conciseError(error, "Connection failed."));
            }
        } finally {
            if (button) {
                button.disabled = false;
                button.textContent = "Save & Test";
            }
            if (refreshButton) refreshButton.disabled = false;
            submissionInFlight = false;
        }
    }

    function updateSessionCard() {
        const card = byId("session-configured-card");
        const summary = byId("session-configured-summary");
        if (card) card.hidden = !configured;
        if (summary && activeConfiguration) {
            const provider = providerById(activeConfiguration.providerId);
            const capability = normalizeReasoning(activeConfiguration.reasoning
                || resolvedReasoning.get(reasoningKey(activeConfiguration))
                || (provider && (provider.models || []).find(model => model.id === activeConfiguration.modelId) || {}).reasoning);
            const state = capability.kind === "none" ? "Not supported"
                : capability.kind === "unknown" ? "Provider default"
                : capability.kind === "fixed" ? "Always on"
                : activeConfiguration.thinkingMode === "disabled" ? "Off"
                : activeConfiguration.thinkingMode === "auto" ? "Provider default"
                : capability.kind === "toggle" ? "On"
                : effortLabel(activeConfiguration.reasoningEffort, capability);
            summary.textContent = `${provider ? provider.name : "Custom"} · ${activeConfiguration.modelId}`
                + ` · Reasoning: ${state}`
                + (verified ? " · Connected" : " · Test required");
        }
    }

    function showStatus(type, message) {
        const status = byId("connection-status");
        if (!status) return;
        status.className = `connection-status ${type}`;
        // Reword prose only; model identifiers such as "fake-model" stay intact.
        status.textContent = String(message)
            .replace(/(?<![\w./:-])models(?![\w/:-]|\.\w)/gi, 'LLMs')
            .replace(/(?<![\w./:-])model(?![\w/:-]|\.\w)/gi, 'LLM');
    }

    async function openSettings() {
        const generation = ++modalGeneration;
        reasoningLookupGeneration += 1;
        pendingReasoningKey = null;
        const modal = byId("settings-modal");
        if (modal) modal.style.display = "flex";
        if (settingsScrollLock === null) {
            settingsScrollLock = [document.documentElement, document.body].filter(element => element && element.style)
                .map(element => ({ element, overflow: element.style.overflow }));
            for (const { element } of settingsScrollLock) element.style.overflow = "hidden";
        }
        showStatus("", "");
        refreshLDLink();
        refreshPhenotypeScopes();
        if (!lastRefreshAt || (formRevision === appliedFormRevision && Date.now() - lastRefreshAt > 5_000)) await refreshSettings();
        else await refreshPublicCatalog();
        if (generation !== modalGeneration) return;
        syncCredentialIndicator();
        resolveModelReasoning();
    }

    function closeSettings() {
        stopPublicCatalogPolling();
        modalGeneration += 1;
        reasoningLookupGeneration += 1;
        pendingReasoningKey = null;
        clearCredential();
        const ldlinkKey = byId("ldlink-api-key");
        if (ldlinkKey) ldlinkKey.value = "";
        ldlinkRevision += 1;
        const modal = byId("settings-modal");
        if (modal) modal.style.display = "none";
        if (settingsScrollLock !== null) {
            for (const { element, overflow } of settingsScrollLock) element.style.overflow = overflow;
            settingsScrollLock = null;
        }
    }

    function bindEvents() {
        const ldlinkKey = byId("ldlink-api-key");
        if (ldlinkKey) ldlinkKey.addEventListener("input", () => { ldlinkRevision += 1; });
        for (const [id, action] of [["ldlink-save-btn", "save"], ["ldlink-test-btn", "test"], ["ldlink-remove-btn", "remove"]]) {
            const button = byId(id);
            if (button) button.addEventListener("click", () => submitLDLink(action));
        }
        const privacy = byId("credential-privacy-text");
        if (privacy) privacy.textContent = credentialPrivacyNotice;
        const provider = byId("model-provider");
        if (provider) provider.addEventListener("change", () => {
            formRevision += 1;
            clearCredential();
            const nextProvider = selectedProvider();
            renderProviderSelection(defaultModelId(nextProvider));
            syncCredentialIndicator();
            resolveModelReasoning();
            showStatus("", "");
        });
        const model = byId("model-catalog-select");
        if (model) model.addEventListener("change", () => {
            formRevision += 1;
            const custom = byId("model-custom-group");
            if (custom) custom.hidden = model.value !== CUSTOM_MODEL;
            const input = byId("model-name");
            if (input && model.value !== CUSTOM_MODEL) input.value = model.value;
            renderModelCapabilities(selectedProvider());
            syncThinkingControls();
            resolveModelReasoning();
        });
        const override = byId("base-url-override");
        if (override) override.addEventListener("change", () => {
            formRevision += 1;
            const base = byId("api-base-url");
            const selected = selectedProvider();
            if (base) {
                if (!override.checked && selected && selected.base_url) base.value = selected.base_url;
                base.readOnly = !override.checked;
            }
            updateCustomEndpointWarning();
            syncThinkingControls();
            clearCredential();
            syncCredentialIndicator();
            resolveModelReasoning();
        });
        const effort = byId("model-reasoning-effort");
        if (effort) effort.addEventListener("change", () => {
            formRevision += 1;
            if (!effort.disabled && (effort.value === "default" || currentReasoning().efforts.includes(effort.value))) {
                reasoningChoices.set(reasoningKey(), { effort: effort.value, mode: effort.value === "default" ? "auto" : "enabled" });
            }
            syncThinkingControls();
        });
        const thinking = byId("model-thinking");
        if (thinking) thinking.addEventListener("change", () => {
            formRevision += 1;
            if (!thinking.disabled) reasoningChoices.set(reasoningKey(), {
                ...reasoningChoice(), mode: thinking.checked ? "enabled" : "disabled"
            });
            syncThinkingControls();
        });
        ["api-key", "api-base-url", "model-name"].forEach((id) => {
            const input = byId(id);
            if (input && typeof input.addEventListener === "function") {
                input.addEventListener("input", () => {
                    formRevision += 1;
                    if (id === "api-key") {
                        // Older input methods may omit beforeinput. Never let
                        // fragments of the synthesized mask become credentials.
                        if (credentialMasked) input.value = input.value.replace(/\u2022/g, "");
                        credentialMasked = false;
                        credentialEdited = true;
                    }
                    if (id === "api-base-url") {
                        clearCredential();
                        syncCredentialIndicator();
                    }
                    if (id !== "api-key") {
                        syncThinkingControls();
                        resolveModelReasoning();
                    }
                });
            }
        });
        const credential = byId("api-key");
        if (credential) {
            ["focus", "beforeinput", "paste"].forEach((event) => {
                credential.addEventListener(event, beginCredentialEdit);
            });
        }
        const refreshModels = byId("refresh-models-btn");
        if (refreshModels) refreshModels.addEventListener("click", refreshProviderModels);
        const modal = byId("settings-modal");
        if (modal) modal.addEventListener("click", (event) => {
            if (event.target === modal) closeSettings();
        });
        const catalogRetry = byId("catalog-retry");
        if (catalogRetry) catalogRetry.addEventListener("click", () => refreshPublicCatalog());
        const settingsRetry = byId("settings-retry");
        if (settingsRetry) settingsRetry.addEventListener("click", () => refreshSettings());
    }

    // A configuration saved by an earlier Save & Test (or supplied by the launch
    // environment) is re-tested once per backend session instead of reopening
    // the dialog; the dialog opens only when that test fails.
    async function verifySavedConfiguration() {
        if (!configured || !activeConfiguration) return false;
        try {
            const response = await fetch("/api/test-llm-connection", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    provider_id: activeConfiguration.providerId,
                    api_base_url: activeConfiguration.baseUrl,
                    model: activeConfiguration.modelId,
                    thinking_mode: activeConfiguration.thinkingMode,
                    reasoning_effort: activeConfiguration.reasoningEffort
                })
            });
            const result = await readJson(response);
            if (response.ok && result.success && result.model_status) {
                applySessionSettings(result.model_status);
                updateSessionCard();
                return verified;
            }
        } catch (error) {
            // Fall through: another page may have verified the same settings.
        }
        // A test started by a previous page can finish first and make this one
        // conflict; the shared session state then already says verified.
        await refreshSettings();
        return verified;
    }

    document.addEventListener("DOMContentLoaded", async () => {
        bindEvents();
        refreshLDLink();
        await refreshSettings();
        if (!verified && !(await verifySavedConfiguration())) {
            window.setTimeout(() => openSettings(), 350);
        }
    });

    window.openSettings = openSettings;
    window.closeSettings = closeSettings;
    window.testConnection = testConnection;
    window.saveSettings = testConnection;
    window.GWASMindModelSettings = {
        isConfigured: () => configured,
        isVerified: () => verified,
        refresh: refreshSettings,
        refreshModels: refreshProviderModels,
        catalogSource: () => catalogFromServer ? "server" : "fallback"
    };

    function renderLDLink(value) {
        const state = byId("ldlink-credential-status");
        if (!state || !value || typeof value !== "object") return;
        ldlinkEditable = value.editable !== false;
        ldlinkConfigured = Boolean(value.configured);
        state.textContent = value.credential_state === "unreadable"
            ? "Saved key cannot be opened by this Windows account. Replace it below."
            : value.credential_state === "invalid_environment"
            ? "The LDlink key in the launch environment cannot be read."
            : ldlinkConfigured ? (value.verified ? "Connected" : "Key available · connection not yet verified")
            : "No LDlink key configured";
        if (value.storage === "environment") state.textContent += " · managed by the launch environment";
        const key = byId("ldlink-api-key");
        if (key) {
            key.disabled = !ldlinkEditable || ldlinkBusy;
            key.placeholder = ldlinkConfigured ? "Leave blank to use the saved key" : "Paste LDlink API key";
        }
        setLDLinkBusy(ldlinkBusy);
    }

    function setLDLinkBusy(value) {
        ldlinkBusy = value;
        for (const id of ["ldlink-save-btn", "ldlink-remove-btn"]) {
            const button = byId(id);
            if (button) button.disabled = value || !ldlinkEditable;
        }
        const test = byId("ldlink-test-btn");
        if (test) test.disabled = value;
        const key = byId("ldlink-api-key");
        if (key) key.disabled = value || !ldlinkEditable;
    }

    async function refreshLDLink() {
        if (!byId("ldlink-api-key")) return;
        const generation = modalGeneration;
        try {
            const response = await fetch("/api/v1/ldlink-settings", { cache: "no-store" });
            if (!response.ok) throw new Error("LDlink settings unavailable");
            const result = await response.json();
            if (generation === modalGeneration) renderLDLink(result.ldlink);
        } catch (_) {
            const state = byId("ldlink-credential-status");
            if (state && generation === modalGeneration) state.textContent = "LDlink settings could not be loaded. Reopen API Settings to retry.";
        }
    }

    async function submitLDLink(action) {
        if (ldlinkBusy) return;
        const key = byId("ldlink-api-key");
        const status = byId("ldlink-connection-status");
        if (!key || !status) return;
        const apiKey = key.value.trim();
        if (action === "save" && !apiKey) {
            status.textContent = "Enter an LDlink key to save. An empty field does not remove the saved key.";
            return;
        }
        const generation = modalGeneration;
        const revision = ldlinkRevision;
        const endpoint = action === "test" ? "/api/v1/ldlink-test" : "/api/v1/ldlink-settings";
        const options = action === "remove" ? { method: "DELETE" } : {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(apiKey ? { api_key: apiKey } : {})
        };
        key.value = "";
        setLDLinkBusy(true);
        status.textContent = action === "test" ? "Testing LDlink access…" : action === "remove" ? "Removing saved key…" : "Saving encrypted key…";
        try {
            const response = await fetch(endpoint, options);
            const result = await response.json();
            if (generation !== modalGeneration || revision !== ldlinkRevision) return;
            renderLDLink(result.ldlink);
            status.textContent = result.response || (response.ok && result.success ? "LDlink settings updated." : "LDlink settings could not be updated.");
        } catch (_) {
            if (generation === modalGeneration && revision === ldlinkRevision) status.textContent = "The local service could not update LDlink settings. Retry after checking the connection.";
        } finally {
            setLDLinkBusy(false);
        }
    }

    window.GWASMindLDLinkSettings = { refresh: refreshLDLink, submit: submitLDLink };

    const PHENOTYPE_RELATION_LABELS = {
        descendant: "Narrower terms under the target",
        sibling: "Terms sharing the target's most specific parent",
        sibling_descendant: "Narrower terms under those terms"
    };
    const MAX_RELATED_PHENOTYPES = 8;
    let phenotypeScopeBusy = false;

    function textElement(tag, text, className) {
        const element = document.createElement(tag);
        element.textContent = text;
        if (className) element.className = className;
        return element;
    }

    function phenotypeScopeCard(scope) {
        const card = document.createElement("details");
        card.className = "phenotype-scope-card";
        card.append(textElement("summary", `${scope.target.name} (${scope.target.id}) · ${scope.related.length} related`));
        const selector = scope.selected_by === "user" ? "you" : "the model";
        card.append(textElement("p", `Open Targets ${scope.ontology_release} · selected by ${selector} · updated ${String(scope.updated_at).slice(0, 10)}`,
            "settings-field-help"));
        const status = textElement("p", "", "settings-field-help");
        status.setAttribute("role", "status");
        const selected = new Set(scope.related.map(row => row.id));
        const boxes = [];
        for (const [relation, title] of Object.entries(PHENOTYPE_RELATION_LABELS)) {
            const rows = scope.candidates.filter(row => row.relation === relation);
            if (!rows.length) continue;
            const group = document.createElement("fieldset");
            group.className = "phenotype-scope-group";
            group.append(textElement("legend", title));
            for (const row of rows) {
                const label = document.createElement("label");
                const box = document.createElement("input");
                box.type = "checkbox";
                box.value = row.id;
                box.checked = selected.has(row.id);
                box.addEventListener("change", () => {
                    if (boxes.filter(item => item.checked).length > MAX_RELATED_PHENOTYPES) {
                        box.checked = false;
                        status.textContent = `At most ${MAX_RELATED_PHENOTYPES} related phenotypes can be kept.`;
                    }
                });
                boxes.push(box);
                label.append(box, document.createTextNode(` ${row.name} (${row.id})`));
                group.append(label);
            }
            card.append(group);
        }
        if (!boxes.length) card.append(textElement("p", "EFO lists no related candidates for this target.", "settings-field-help"));
        const save = textElement("button", "Save related phenotypes", "test-button");
        save.type = "button";
        save.disabled = !boxes.length;
        save.addEventListener("click", () => submitPhenotypeScope(scope.target.id, "save",
            boxes.filter(box => box.checked).map(box => box.value), status));
        const reset = textElement("button", "Reselect on the next analysis");
        reset.type = "button";
        reset.addEventListener("click", () => submitPhenotypeScope(scope.target.id, "reset", null, status));
        card.append(status, save, reset);
        return card;
    }

    function renderPhenotypeScopes(scopes) {
        const list = byId("phenotype-scope-list");
        if (!list) return;
        list.replaceChildren();
        if (!Array.isArray(scopes) || !scopes.length) {
            list.append(textElement("p", "No phenotype scope saved yet. The first gene-prioritization analysis of a phenotype saves one.",
                "settings-field-help"));
            return;
        }
        for (const scope of scopes) list.append(phenotypeScopeCard(scope));
    }

    async function refreshPhenotypeScopes() {
        const list = byId("phenotype-scope-list");
        if (!list) return;
        try {
            const response = await fetch("/api/v1/phenotype-scopes", { cache: "no-store" });
            if (!response.ok) throw new Error("Phenotype scopes unavailable");
            renderPhenotypeScopes((await response.json()).scopes);
        } catch (_) {
            list.textContent = "Saved phenotype scopes could not be loaded. Reopen API Settings to retry.";
        }
    }

    async function submitPhenotypeScope(targetId, action, relatedIds, status) {
        if (phenotypeScopeBusy) return;
        phenotypeScopeBusy = true;
        const endpoint = `/api/v1/phenotype-scopes/${encodeURIComponent(targetId)}`;
        const options = action === "reset" ? { method: "DELETE" } : {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ related_ids: relatedIds })
        };
        status.textContent = action === "reset" ? "Removing the saved scope…" : "Saving related phenotypes…";
        try {
            const response = await fetch(endpoint, options);
            const result = await response.json();
            if (!response.ok || !result.success) {
                status.textContent = "The saved phenotype scope could not be updated.";
                return;
            }
            renderPhenotypeScopes(result.scopes);
        } catch (_) {
            status.textContent = "The local service could not update the phenotype scope. Retry after checking the connection.";
        } finally {
            phenotypeScopeBusy = false;
        }
    }

    window.GWASMindPhenotypeScopes = { refresh: refreshPhenotypeScopes, submit: submitPhenotypeScope };
}());

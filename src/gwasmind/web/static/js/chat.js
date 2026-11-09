/**
 * GWASMind chat controller.
 *
 * This keeps the original cards, bubbles, progress affordances and plot linkage,
 * while routing all work through the local single-agent service.  It never
 * displays hidden chain-of-thought and never stores credentials in the browser.
 */
(function () {
    "use strict";

    let chatFontSize = 14;
    let chatHistory = [];
    let openTargetsAvailable = null;
    let currentHighlightedIndex = null;
    let lastClickTime = 0;
    let activeChatController = null;
    let chatRequestGeneration = 0;
    let currentRegionCardScope = 0;

    const MAX_CHAT_HISTORY = globalThis.GWASMindConfig.get('web.chat_history_limit');
    const MAX_CHAT_HISTORY_ITEM_CHARS = globalThis.GWASMindConfig.get('web.chat_history_item_max_chars');
    const MAX_CHAT_VARIANTS = globalThis.GWASMindConfig.get('execution.max_variants_per_query');
    const MAX_CHAT_PROGRESS_LINES = 80;
    const MAX_PUBLIC_PROGRESS_METRIC = 10_000_000;
    const MAX_PUBLIC_ELAPSED_SECONDS = 7 * 24 * 60 * 60;
    const EVIDENCE_RECOVERY_SUMMARY = "Verified evidence ready · interpretation incomplete";
    const LEGACY_RANKING_FAILURE_DETAIL = 'Model ranking did not complete. Retrieved evidence remains available; no rule-based ranking was substituted.';
    const RETAINED_RANKING_DETAIL = 'Model ranking retained; identifier, value and citation checks require review.';
    const REPORT_SECTION_TITLES = new Set([
        "Summary",
        "Evidence",
        "Details",
        "Limitations",
        "Limitations & Uncertainty",
        "References"
    ]);
    const CHAT_PROGRESS_STAGES = Object.freeze([
        Object.freeze({ id: "plan", title: "Define Analysis Scope" }),
        Object.freeze({ id: "retrieve", title: "Retrieve Genetic Evidence" }),
        Object.freeze({ id: "safety", title: "Safety & Sufficiency Review" }),
        Object.freeze({ id: "response", title: "Interpret & Verify Findings" })
    ]);
    const CHAT_PROGRESS_STAGE_IDS = new Set(CHAT_PROGRESS_STAGES.map((stage) => stage.id));
    const liveProgressCards = new Set();
    let progressCardSequence = 0;
    const RETRIEVAL_PHASE_LABELS = Object.freeze({
        ld_source_discovery: 'Matching selected variants to the reference genome',
        ld_identity_batch: 'Checking variant identifiers and reference alleles',
        ld_source_identity: 'Validating selected variants for LD analysis', ld_proxy_lookup: 'Retrieving population-specific LD proxies',
        ld_proxy_identity: 'Validating LD proxy variant identities', ld_population_identity: 'Checking variants in the LD reference panel',
        ld_expansion: 'Expanding the locus with LD proxy variants', ld_reference_request: 'Checking reference-genome annotations',
        ld_recovery: 'Completing LD proxy validation',
        catalog_request: 'Searching published GWAS associations', semantic_variant_batch: 'Retrieving variant annotations',
        semantic_tool: 'Retrieving genetic and functional evidence', retrieval_recovery: 'Completing evidence retrieval',
        result_processing: 'Organizing retrieved genetic evidence',
        retrieval_phase: 'Retrieving genetic and functional evidence',
        v2g_retrieval: 'Retrieving Open Targets evidence'
    });
    const V2G_RETRIEVAL_LABELS = Object.freeze({
        phenotype: 'Verifying Open Targets phenotype identities',
        candidates: 'Identifying candidate genes', gene_profiles: 'Retrieving molQTL, expression and gene function',
        variants: 'Retrieving variant consequences and E2G', colocalisation: 'Retrieving GWAS–QTL colocalisation',
        associations: 'Retrieving gene–trait associations', cards: 'Preparing evidence cards'
    });
    const RETRIEVAL_TOOL_LABELS = Object.freeze({
        ot_get_capabilities: 'Checking available Open Targets datasets', ot_resolve_entities: 'Matching variant and gene identifiers',
        ot_get_locus_context: 'Retrieving GWAS locus annotations', ot_get_locus_gene_evidence: 'Retrieving variant-to-gene evidence',
        ot_get_target_profiles: 'Retrieving gene function and expression', ot_get_target_disease_evidence: 'Retrieving gene–trait associations',
        ot_get_disease_profiles: 'Retrieving disease and phenotype annotations', ot_get_drug_evidence: 'Retrieving drug–target evidence',
        ot_analyze_gene_set: 'Retrieving gene-set evidence'
    });
    const RETRIEVAL_STATES = new Set(['started', 'running', 'waiting', 'completed', 'incomplete', 'failed', 'unavailable', 'blocked', 'not_run', 'retrying', 'cached', 'splitting']);
    const OT_IDENTITY_PHASES = new Set(['ld_reference_request', 'ld_identity_batch', 'ld_source_identity',
        'ld_source_discovery', 'ld_proxy_identity', 'ld_population_identity']);
    const RETRIEVAL_PHASE_NAMES = Object.freeze({ discovery: 'Candidate-gene discovery', shortlist: 'Candidate-gene shortlist',
        colocalisation_details: 'Shortlisted-gene evidence' });
    const RETRIEVAL_MODULE_NAMES = Object.freeze({ 'locus.colocalisation': 'Colocalization evidence',
        'runtime.capabilities': 'Data-service capabilities', 'schema.discovery': 'Data-service capabilities', 'entity.resolve': 'Entity identifiers',
        'locus.qtl': 'Molecular QTL (molQTL) associations', 'locus.enhancer_to_gene': 'Regulatory element-to-gene (E2G) links',
        'locus.variant': 'Variant annotations', 'locus.variant_consequence': 'Variant consequences',
        'locus.study': 'GWAS study evidence', 'locus.credible_set': 'Credible-set evidence',
        'locus.fine_mapping': 'Fine-mapping evidence', 'locus.l2g': 'Locus-to-gene predictions',
        'target.core': 'Gene annotations', 'target.function': 'Gene function',
        'target.go': 'Gene ontology', 'target.pathway': 'Pathway evidence',
        'target.expression': 'Tissue and cell-type expression', 'target.safety': 'Target safety evidence',
        'target.tractability': 'Target tractability', 'evidence.europepmc': 'Literature evidence',
        'association.datasource': 'Associations by data source', 'association.overall': 'Gene–disease associations',
        'association.raw_evidence': 'Association evidence records', 'disease.core': 'Disease annotations',
        'disease.hpo': 'Disease HPO annotations', 'disease.ontology': 'Disease ontology', 'disease.phenotype': 'Disease phenotypes',
        'drug.core': 'Drug annotations', 'drug.indication': 'Drug indications', 'drug.mechanism': 'Drug mechanisms',
        'drug.clinical_target': 'Clinical target evidence', 'drug.adverse_event': 'Drug adverse events',
        'drug.pharmacogenomics': 'Pharmacogenomic evidence', 'drug.warning': 'Drug warnings',
        'gene_set.evidence_coverage': 'Gene-set evidence coverage', 'gene_set.go': 'Gene-set ontology',
        'gene_set.pathway': 'Gene-set pathways', 'gene_set.tissue_coverage': 'Gene-set tissue coverage' });
    const RETRIEVAL_OUTCOMES = Object.freeze({ RUNNING: 'In progress', SUCCESS_DATA: 'Matching records retrieved', SUCCESS_NO_DATA: 'Query complete · no matching records',
        PARTIAL: 'Partial evidence coverage', FAILED: 'Query failed · coverage unknown', BLOCKED: 'Blocked · not performed', NOT_RUN: 'Not performed', NOT_REQUESTED: 'Not requested', NOT_APPLICABLE: 'Not applicable' });
    const SCIENTIFIC_TASKS = new Set(['general', 'describe_variants', 'gene_associations', 'prioritize_genes',
        'target_assessment', 'disease_associations', 'drug_repurposing']);
    const RESPONSE_SUBSTAGES = new Set(['inference', 'fact_verification']);
    // The shared renderer consumes one task profile and real operation events.
    // Adding a task or provider must not add a new set of clocks/disclosures.
    const TASK_PROGRESS_PRESENTATION = Object.freeze({
        general: { retrievalTitle: 'Review Question Context', scope: 'none', phases: {} },
        describe_variants: { retrievalTitle: 'Annotate Selected Variants', scope: 'variants', phases: { discovery: 'Variant annotations' } },
        gene_associations: { retrievalTitle: 'Retrieve Variant-to-Gene Evidence', scope: 'genes', phases: {
            discovery: 'Variant-to-gene evidence', shortlist: 'Gene-evidence assessment', colocalisation_details: 'Molecular and regulatory evidence' } },
        prioritize_genes: { retrievalTitle: 'Retrieve Variant-to-Gene Evidence', scope: 'ranking', phases: RETRIEVAL_PHASE_NAMES },
        target_assessment: { retrievalTitle: 'Retrieve Target Evidence', scope: 'variants', phases: {} },
        disease_associations: { retrievalTitle: 'Retrieve Genetic Associations', scope: 'variants', phases: {} },
        drug_repurposing: { retrievalTitle: 'Retrieve Target & Drug Evidence', scope: 'variants', phases: {} }
    });
    const DEFAULT_PROGRESS_PRESENTATION = Object.freeze({ retrievalTitle: 'Retrieve Genetic Evidence', scope: 'ranking', phases: {
        discovery: 'Evidence discovery', shortlist: 'Evidence selection', colocalisation_details: 'Detailed evidence' } });
    function taskPresentation(task) { return Object.hasOwn(TASK_PROGRESS_PRESENTATION, task) ? TASK_PROGRESS_PRESENTATION[task] : DEFAULT_PROGRESS_PRESENTATION; }
    const RETRIEVAL_COUNTS = ['source_count', 'completed_count', 'total_count', 'validated_count', 'pending_count',
        'batch_index', 'batch_count', 'variant_count', 'total_analysis_variants', 'records', 'elapsed_seconds', 'step', 'attempt', 'page_size', 'offset', 'wait_seconds',
        'network_requests', 'cache_hits', 'failed_cache_hits', 'completed_pages', 'failed_pages', 'failed_request_pages', 'unique_request_pages',
        'successful_requests', 'failed_items', 'partial_items',
        'unique_credible_sets', 'unique_studies', 'unique_connections', 'returned_count',
        'verified_proxy_count', 'previous_validated_count', 'newly_validated_count', 'proxy_count', 'candidate_count'];
    const SCOPE_LABELS = Object.freeze({ source_variant_count: 'Selected GWAS variants', verified_proxy_count: 'Validated LD proxies',
        analysis_variant_count: 'Variants analyzed', candidate_pool_count: 'Candidate genes', shortlist_count: 'Shortlisted genes',
        ranked_count: 'Prioritized genes', displayed_ranked_count: 'Displayed genes' });
    const RETRIEVAL_REASON_LABELS = Object.freeze({ ld_credential_unavailable: 'LDlink API connection is not configured', source_identity_unavailable: 'Source variant identity could not be verified', source_identity_unresolved: 'Source variant identity could not be verified', external_acquisition_unavailable: 'External evidence service unavailable', service_error: 'Service temporarily unavailable', rate_limited: 'Service rate limit',
        request_timeout: 'Request timed out', invalid_response: 'Service response could not be used', connection_error: 'Connection interrupted',
        query_too_expensive: 'Splitting a complex query into smaller requests', partial_page: 'Completing a partial page', request_budget: 'Recovery limit reached',
        continuation_budget_exhausted: 'Recovery limit reached', invalid_or_unavailable_response: 'Incomplete service response',
        aggregate_request_too_large: 'Splitting a large request into smaller requests',
        response_size_limit: 'Splitting an oversized response into smaller requests',
        reference_batch_unavailable: 'The reference batch did not finish; using smaller requests',
        cooldown_exceeds_budget: 'The requested service wait exceeds the remaining retrieval time',
        transport_or_deadline_gap: 'Reusing completed LD lookups and retrying unfinished checks' });

    function stageTitle(id, task = null) {
        if (id === 'retrieve') return taskPresentation(task).retrievalTitle;
        return CHAT_PROGRESS_STAGES.find(stage => stage.id === id)?.title || 'Analysis progress';
    }

    function recordedDuration(value) {
        if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > MAX_PUBLIC_ELAPSED_SECONDS) return null;
        if (value < 1) return '<1 s';
        return `${value.toLocaleString('en-US', { maximumFractionDigits: 3 })} s`;
    }

    function renderHistoricalProvenance(progress, stage, value, state) {
        const labels = { request_interpretation: 'Phenotype and analysis scope', evidence_retrieval: 'Genetic evidence retrieval',
            scientific_review: 'Evidence sufficiency review', gene_ranking: 'Candidate-gene prioritization',
            report_generation: 'Scientific report generation', report_revalidation: 'Report fact revalidation' };
        if (!progress.historical || !value || typeof value.separate_executions !== 'boolean'
                || !Array.isArray(value.components) || value.components.length > 6 || !value.components.length) return;
        const components = [];
        for (const component of value.components) {
            if (!component || !Object.hasOwn(labels, component.kind) || typeof component.reused !== 'boolean') return;
            if (component.duration_seconds !== undefined && recordedDuration(component.duration_seconds) === null) return;
            if (component.version !== undefined && !/^v\d{1,5}$/.test(component.version)) return;
            if (component.new_model_calls !== undefined && boundedPublicInteger(component.new_model_calls) === null) return;
            components.push(component);
        }
        stage.historyProvenance?.remove();
        const block = document.createElement('div'); block.className = 'progress-history-provenance';
        const title = document.createElement('strong');
        title.textContent = value.separate_executions ? 'Recorded runs · durations shown separately' : 'Recorded analysis source';
        block.appendChild(title);
        for (const component of components) {
            const row = document.createElement('div'); row.className = 'progress-recorded-component';
            const label = document.createElement('span'); label.textContent = labels[component.kind];
            const duration = document.createElement('span'); duration.className = 'progress-recorded-duration';
            duration.textContent = recordedDuration(component.duration_seconds) || 'Time not recorded';
            row.append(label, duration);
            const note = document.createElement('small');
            note.textContent = component.reused ? 'Reused from a saved analysis' : 'Recorded for this review';
            if (component.new_model_calls === 0) note.textContent += ' · no new LLM call';
            row.appendChild(note); block.appendChild(row);
        }
        stage.details.appendChild(block); stage.historyProvenance = block;
        stage.separateRecordedRuns = value.separate_executions;
        if (stage.id === 'response' && ['complete', 'warning'].includes(state)
                && components.some(component => ['gene_ranking', 'report_generation'].includes(component.kind))) {
            stage.phaseStates.set('inference', state === 'complete' ? 'SUCCESS_DATA' : 'PARTIAL');
        }
        renderProgressClock(stage);
    }

    function abortError() {
        const error = new Error("Request cancelled");
        error.name = "AbortError";
        return error;
    }

    function isAbortError(error) {
        return Boolean(error && error.name === "AbortError");
    }

    function beginChatRequest() {
        if (activeChatController) activeChatController.abort();
        disposeChatProgress(document.getElementById?.("chat-messages"));
        const request = {
            controller: new AbortController(),
            generation: ++chatRequestGeneration
        };
        activeChatController = request.controller;
        return request;
    }

    function isCurrentChatRequest(request) {
        return request.generation === chatRequestGeneration
            && request.controller === activeChatController
            && !request.controller.signal.aborted;
    }

    function finishChatRequest(request) {
        if (request.controller === activeChatController) activeChatController = null;
    }

    function abortActiveChatRequest() {
        chatRequestGeneration += 1;
        if (activeChatController) activeChatController.abort();
        activeChatController = null;
        disposeChatProgress(document.getElementById?.("chat-messages"));
    }

    async function checkMcpData() {
        // Cache a healthy local bridge, but retry a transient startup failure
        // instead of disabling Chat until the page is reloaded.
        if (openTargetsAvailable === true) return true;
        try {
            const response = await fetch("/api/db-status");
            const value = await response.json();
            openTargetsAvailable = Boolean(response.ok && value.bridge_available);
        } catch (error) {
            openTargetsAvailable = false;
        }
        return openTargetsAvailable;
    }

    function showMcpMissingPopup() {
        if (document.getElementById("mcp-popup-overlay")) return;
        const overlay = document.createElement("div");
        overlay.id = "mcp-popup-overlay";
        overlay.style.cssText = "position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.45);display:flex;align-items:center;justify-content:center;";
        const panel = document.createElement("div");
        panel.style.cssText = "background:#fff;border-radius:12px;padding:32px 36px;max-width:440px;width:90%;box-shadow:0 8px 32px rgba(0,0,0,.18);text-align:center;";
        const icon = document.createElement("div");
        icon.style.cssText = "font-size:44px;margin-bottom:12px;";
        icon.textContent = "⚠️";
        const title = document.createElement("h3");
        title.style.cssText = "margin:0 0 10px;font-size:18px;color:#333;";
        title.textContent = "Local evidence bridge unavailable";
        const message = document.createElement("p");
        message.style.cssText = "color:#555;font-size:14px;line-height:1.6;margin:0 0 20px;";
        message.textContent = "The local GWASMind evidence bridge is unavailable. Restart the local service, then retry; live upstream errors are reported by the analysis itself.";
        const close = document.createElement("button");
        close.style.cssText = "padding:10px 24px;background:#e0e0e0;color:#333;border:none;border-radius:6px;cursor:pointer;font-size:14px;";
        close.textContent = "Close";
        close.addEventListener("click", () => overlay.remove());
        panel.append(icon, title, message, close);
        overlay.appendChild(panel);
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) overlay.remove();
        });
        document.body.appendChild(overlay);
    }

    function getChatHistory() {
        return MAX_CHAT_HISTORY === 0 ? [] : chatHistory.slice(-MAX_CHAT_HISTORY);
    }

    function addToChatHistory(role, content) {
        window.recordWorkspaceChatTurn?.(role, content);
        if (MAX_CHAT_HISTORY === 0) { chatHistory = []; return; }
        chatHistory.push({ role, content: String(content || "").slice(0, MAX_CHAT_HISTORY_ITEM_CHARS) });
        if (chatHistory.length > MAX_CHAT_HISTORY) {
            chatHistory = chatHistory.slice(-MAX_CHAT_HISTORY);
        }
    }

    function createRegionResponseCard(region, index, scope = currentRegionCardScope) {
        const card = document.createElement("div");
        card.className = "region-response-card";
        card.id = `response-${scope}-${index}`;
        card.dataset.regionIndex = String(index);

        const header = document.createElement("div");
        header.className = "region-card-header";
        header.dataset.regionIndex = String(index);
        header.onclick = event => {
            event.stopPropagation();
            // Older cards refer to their own original locus, not the current
            // selection's ordinal after another question or lasso operation.
            window.ManhattanPlot?.highlightPointsOnly?.(index, region);
            document.querySelectorAll?.('.region-response-card').forEach(node => node.classList.remove('active'));
            card.classList?.add('active');
        };
        const badge = document.createElement("span");
        badge.className = "region-badge";
        badge.textContent = `chr${region.chromosome}`;
        const count = document.createElement("span");
        count.className = "region-info";
        count.textContent = `${region.snps.length} SNPs`;
        const range = document.createElement("span");
        range.className = "region-range";
        range.textContent = `${Number(region.start).toLocaleString()}-${Number(region.end).toLocaleString()}`;
        header.append(badge, count, range);

        const body = document.createElement("div");
        body.className = "region-card-body";
        const loading = document.createElement("div");
        loading.className = "card-loading";
        loading.id = `loading-${scope}-${index}`;
        // The shared stage panel carries progress; do not animate a second placeholder.
        loading.hidden = true;
        const content = document.createElement("div");
        content.className = "card-content";
        content.id = `content-${scope}-${index}`;
        content.style.display = "none";
        body.append(loading, content);
        card.append(header, body);
        return card;
    }

    function highlightAll(index) {
        const now = Date.now();
        if (index === currentHighlightedIndex && now - lastClickTime < 300) {
            clearAllHighlights();
            currentHighlightedIndex = null;
            lastClickTime = 0;
            return;
        }
        currentHighlightedIndex = index;
        lastClickTime = now;
        document.querySelectorAll(".multi-region-item").forEach((item, itemIndex) => {
            item.classList.toggle("highlighted", itemIndex === index);
        });
        if (typeof ManhattanPlot !== "undefined" && ManhattanPlot.highlightRegionPoints) {
            ManhattanPlot.highlightRegionPoints(index);
        }
        document.querySelectorAll(".region-response-card").forEach((card) => card.classList.remove("active"));
        const card = document.getElementById(`response-${currentRegionCardScope}-${index}`);
        if (card) {
            card.classList.add("active");
            card.scrollIntoView({ behavior: "smooth", block: "center" });
        }
    }

    function clearAllHighlights() {
        document.querySelectorAll(".multi-region-item").forEach((item) => item.classList.remove("highlighted"));
        document.querySelectorAll(".region-response-card").forEach((card) => card.classList.remove("active"));
        if (typeof ManhattanPlot !== "undefined" && ManhattanPlot.clearHighlight) {
            ManhattanPlot.clearHighlight();
        }
    }

    function highlightCardFromLeft(index) { highlightAll(index); }
    function highlightFromCard(index) { highlightAll(index); }

    function rankingDisplaySummary(value) {
        if (!value || !Number.isSafeInteger(value.ranked_count) || !Number.isSafeInteger(value.displayed_ranked_count)
                || value.ranked_count < 1 || value.displayed_ranked_count < 0
                || value.displayed_ranked_count >= value.ranked_count) return '';
        return `${value.ranked_count.toLocaleString('en-US')} ranked · ${value.displayed_ranked_count.toLocaleString('en-US')} displayed`;
    }

    function publicRankingEvidence(value) {
        if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
        if (typeof value.run_id !== 'string' || !/^[a-f0-9]{32}$/.test(value.run_id)) return null;
        for (const key of ['snapshot_id', 'model_input_sha256', 'messages_sha256']) {
            if (typeof value[key] !== 'string' || !/^[a-f0-9]{64}$/.test(value[key])) return null;
        }
        for (const key of ['query_index', 'candidate_count']) {
            if (!Number.isSafeInteger(value[key]) || value[key] < 0) return null;
        }
        return Object.fromEntries(['run_id', 'query_index', 'snapshot_id', 'model_input_sha256',
            'messages_sha256', 'candidate_count'].map(key => [key, value[key]]));
    }

    function renderRankingEvidence(target, value) {
        if (!target) return false;
        const incoming = Array.isArray(value) ? value : value ? [value] : target.rankingEvidence || [];
        const entries = new Map();
        for (const item of incoming.slice(0, 256)) {
            const checked = publicRankingEvidence(item);
            if (checked) entries.set(`${checked.run_id}:${checked.query_index}`, checked);
        }
        target.rankingEvidence = [...entries.values()];
        const cardEntry = target.querySelector?.('.gene-evidence-cards');
        cardEntry?.remove();
        target.querySelector?.(':scope > .ranking-evidence-downloads')?.remove();
        if (!entries.size) {
            if (cardEntry) target.prepend(cardEntry);
            return false;
        }
        const toolbar = document.createElement('div');
        toolbar.className = 'ranking-evidence-downloads';
        toolbar.setAttribute('aria-label', 'Gene ranking evidence downloads');
        if (cardEntry) {
            toolbar.classList.add('has-gene-card');
            toolbar.appendChild(cardEntry);
        }
        for (const [index, item] of target.rankingEvidence.entries()) {
            const group = document.createElement('div'); group.className = 'ranking-evidence-download-group';
            if (entries.size > 1) {
                const label = document.createElement('span'); label.className = 'ranking-evidence-region-label';
                label.textContent = `Analysis ${index + 1}`; group.appendChild(label);
            }
            for (const [suffix, text] of [['ranking-evidence', 'Download evidence card'], ['ranking-input', 'Download LLM input']]) {
                const link = document.createElement('a'); link.className = 'ranking-evidence-download';
                link.textContent = text;
                link.setAttribute('href', `/api/v1/runs/${item.run_id}/artifact/${item.query_index}/${suffix}?format=html`);
                link.setAttribute('download', '');
                group.appendChild(link);
            }
            toolbar.appendChild(group);
        }
        target.prepend(toolbar);
        return true;
    }

    const GENE_CARD_CATEGORIES = Object.freeze([
        ['coding', 'Variant consequence'], ['qtl', 'Molecular QTL'], ['coloc', 'GWAS–QTL colocalisation'],
        ['e2g', 'Enhancer-to-gene'], ['expression', 'Expression'], ['association', 'Trait association'],
        ['constraint', 'Constraint'], ['function', 'Function']]);
    const GENE_CARD_STATUS_LABELS = Object.freeze({ observed: 'Observed', not_observed: 'Not observed',
        partial: 'Partial', failed: 'Failed', not_requested: 'Not requested' });

    function publicQtlRows(value) {
        if (!Array.isArray(value)) return [];
        const text = (item, max = 2000) => typeof item === 'string' && item.length <= max ? item : '';
        const variant = item => /^(?:[1-9]|1\d|2[0-2]|X|Y|MT)_\d+_[ACGT]+_[ACGT]+$/.test(item || '') ? item : '';
        return value.filter(row => row && typeof row === 'object'
            && (!row.measurement_scope || (row.measurement_scope === 'credible_set_lead'
                && variant(row.cs_lead_variant_id) && row.cs_lead_variant_id === row.variant_id))).map(row => {
            const leadScope = row.measurement_scope === 'credible_set_lead'
                && variant(row.cs_lead_variant_id) && row.cs_lead_variant_id === row.variant_id;
            const witnesses = leadScope && Array.isArray(row.locus_anchor_witnesses) ? row.locus_anchor_witnesses
                .filter(item => item && variant(item.variant_id) && variant(item.locus_variant_id))
                .map(item => ({ variant_id: item.variant_id, locus_variant_id: item.locus_variant_id,
                    variant_origin: ['source_selected', 'ld_proxy'].includes(item.variant_origin) ? item.variant_origin : 'unknown' })) : [];
            let origin = ['source_selected', 'ld_proxy'].includes(row.variant_origin) ? row.variant_origin : 'unknown';
            if (leadScope && !witnesses.some(item => item.variant_id === row.variant_id && item.variant_origin === origin)) origin = 'unknown';
            const measurement = {};
            for (const key of ['pValueMantissa', 'pValueExponent', 'beta', 'standardError']) {
                if (typeof row.measurement?.[key] === 'number' && Number.isFinite(row.measurement[key])) measurement[key] = row.measurement[key];
            }
            const links = Array.isArray(row.ld_relations) ? row.ld_relations.filter(link => link
                && variant(link.selected_variant_id) && variant(link.proxy_variant_id)
                && (leadScope ? witnesses.some(item => item.variant_id === link.proxy_variant_id) : link.proxy_variant_id === variant(row.variant_id))
                && typeof link.r2 === 'number' && Number.isFinite(link.r2) && link.r2 >= 0 && link.r2 <= 1
                && ['AFR', 'AMR', 'EAS', 'EUR', 'SAS'].includes(link.population) && link.genome_build === 'GRCh38')
                .map(link => ({ selected_variant_id: link.selected_variant_id, proxy_variant_id: link.proxy_variant_id,
                    r2: link.r2, population: link.population, genome_build: link.genome_build })) : [];
            return { variant_id: variant(row.variant_id), locus_variant_id: variant(row.locus_variant_id), variant_origin: origin,
                variant_display: text(row.variant_display, 300) || variant(row.variant_id),
                ...(/^rs\d+$/.test(row.rsid || '') ? { rsid: row.rsid } : {}),
                ...(row.genome_build === 'GRCh38' ? { genome_build: 'GRCh38' } : {}),
                variant_label: leadScope ? 'Credible-set lead' : { source_selected: 'User-selected variant', ld_proxy: 'LD-linked variant' }[origin] || 'Analyzed variant',
                qtl_type: text(row.qtl_type, 30), biosample: text(row.biosample, 500), biosample_id: text(row.biosample_id, 200),
                p_value: text(row.p_value, 80), beta: text(row.beta, 80), study_id: text(row.study_id),
                study_locus_id: text(row.study_locus_id), evidence_id: text(row.evidence_id), project: text(row.project, 200),
                ...(leadScope ? { measurement_scope: 'credible_set_lead', cs_lead_variant_id: row.variant_id,
                    locus_anchor_witnesses: witnesses } : { context_matched: row.context_matched === true }),
                summary: text(row.summary), measurement,
                legacy_display: row.legacy_display === true,
                ld_relations: links, ld_reference_panel: text(row.ld_reference_panel, 200) };
        });
    }

    function legacyQtlRows(items) {
        return (Array.isArray(items) ? items : []).flatMap(item => {
            if (typeof item !== 'string') return [];
            const parts = item.split(' · ');
            if (!/^(?:sc-)?(?:e|s|p|tu|mol)?QTL$/i.test(parts[0])
                || !/^P=/.test(parts[3] || '') || !/^beta=/i.test(parts[4] || '')) return [];
            return [{ qtl_type: parts[0], biosample: parts[1], variant_id: parts[2], locus_variant_id: parts[2],
                p_value: parts[3].slice(2), beta: parts[4].replace(/^beta=/i, ''), study_id: parts[5],
                context_matched: parts[6] === 'Context matched', variant_origin: 'unknown', legacy_display: true }];
        });
    }

    function createQtlDisplayRow(row, { citations = [], provenance = [] } = {}) {
        const wrapper = document.createElement('div'); wrapper.className = 'qtl-compact-row';
        const line = document.createElement('div'); line.className = 'qtl-compact-line';
        const separate = () => {
            if (!line.children.length) return;
            const separator = document.createElement('span'); separator.className = 'qtl-separator'; separator.textContent = ' · ';
            line.appendChild(separator);
        };
        const add = (tag, className, text) => {
            separate();
            const node = document.createElement(tag); node.className = className; node.textContent = text; line.appendChild(node); return node;
        };
        const variant = add('code', 'qtl-measured-variant', row.variant_display || row.variant_id || 'Variant unavailable');
        if (/^rs[1-9][0-9]*$/.test(row.rsid || '') && row.variant_display === row.rsid) {
            const link = document.createElement('a');
            link.className = 'variant-reference'; link.textContent = row.rsid;
            link.setAttribute('href', `https://www.ncbi.nlm.nih.gov/snp/${row.rsid}`);
            link.target = '_blank'; link.rel = 'noopener noreferrer';
            link.title = `Measured variant: ${row.variant_id}${row.genome_build ? ` (${row.genome_build})` : ''}`;
            variant.textContent = ''; variant.appendChild(link);
        }
        add('span', 'qtl-assay', row.qtl_type || 'QTL');
        add('span', 'qtl-statistic', row.p_value ? `P=${row.p_value}` : 'P unavailable');
        add('span', 'qtl-statistic', row.beta ? `β=${row.beta}` : 'β unavailable');
        for (const citation of citations) { separate(); line.appendChild(citation); }
        const details = document.createElement('details'); details.className = 'qtl-source-details';
        const title = document.createElement('summary'); title.textContent = 'Source'; details.appendChild(title);
        const body = document.createElement('div'); body.className = 'qtl-source-body';
        const detail = (label, value) => {
            if (value === null || value === undefined || value === '') return;
            const entry = document.createElement('p'); entry.textContent = `${label}: ${value}`; body.appendChild(entry);
        };
        detail('Study', row.study_id); detail('Study locus', row.study_locus_id); detail('Project', row.project);
        detail('Measured variant', row.variant_id);
        detail('Genome build', row.genome_build);
        detail('Measurement scope', row.measurement_scope === 'credible_set_lead' ? 'Credible-set lead' : row.variant_label);
        detail('rsID', row.rsid); detail('Biosample name', row.biosample);
        for (const witness of row.locus_anchor_witnesses || []) {
            const role = { source_selected: 'User-selected variant', ld_proxy: 'LD-linked variant' }[witness.variant_origin] || 'Analyzed variant';
            detail('Locus overlap', `${role} ${witness.variant_id}`);
            if (witness.locus_variant_id !== witness.variant_id) detail('Input locus identity', witness.locus_variant_id);
        }
        if (row.locus_variant_id && row.locus_variant_id !== row.variant_id) detail('Input locus identity', row.locus_variant_id);
        detail('Biosample', row.biosample_id); detail('Evidence record', row.evidence_id);
        if (typeof row.context_matched === 'boolean') detail('Context match', row.context_matched ? 'Matched' : 'Not established');
        if (typeof row.measurement?.pValueMantissa === 'number' && typeof row.measurement?.pValueExponent === 'number')
            detail('P as reported', `${row.measurement.pValueMantissa}e${row.measurement.pValueExponent}`);
        detail('Beta as reported', row.measurement?.beta); detail('Standard error', row.measurement?.standardError);
        for (const link of row.ld_relations || [])
            detail('LD relation', `${link.selected_variant_id} ↔ ${link.proxy_variant_id}; r²=${link.r2}; ${link.population}; ${link.genome_build}`);
        if (row.ld_relations?.length) detail('LD reference', row.ld_reference_panel);
        for (const text of provenance) {
            if (!text || /^Source SNPs:\s*none\b/i.test(text) || /^Proxy evidence is indirect/i.test(text)) continue;
            const entry = document.createElement('p'); entry.textContent = text.replace(/^Source SNPs:/, 'User-selected variants:')
                .replace(/^Verified LD proxies:/, 'LD-linked variants:'); body.appendChild(entry);
        }
        details.appendChild(body); separate(); line.appendChild(details); wrapper.appendChild(line);
        return wrapper;
    }

    function publicGeneEvidenceCards(value) {
        if (!Array.isArray(value)) return null;
        const text = (item, limit) => typeof item === 'string' ? item.slice(0, limit) : '';
        const list = (items, count, limit) => Array.isArray(items)
            ? items.filter(item => typeof item === 'string').slice(0, count).map(item => item.slice(0, limit)) : [];
        const count = item => Number.isSafeInteger(item) && item >= 0 ? item : null;
        const cards = [];
        for (const card of value) {
            if (!card || typeof card !== 'object' || !/^ENSG\d{11}$/.test(String(card.gene_id || ''))
                    || (card.tier !== undefined && !['A', 'B', 'C'].includes(card.tier))) continue;
            const categories = {};
            for (const [key] of GENE_CARD_CATEGORIES) {
                const summary = card.categories?.[key];
                if (!summary || typeof summary !== 'object') continue;
                categories[key] = { status: Object.hasOwn(GENE_CARD_STATUS_LABELS, summary.status) ? summary.status : 'not_requested',
                    count: count(summary.count), items: list(summary.items, 4, 300) };
                if (Array.isArray(summary.tags)) categories[key].tags = list(summary.tags, 5, 40)
                    .filter(tag => ['Coding', 'UTR', 'Synonymous', 'Coloc (trait)', 'Coloc (other)'].includes(tag));
                if (key === 'qtl') {
                    const rows = publicQtlRows(Array.isArray(summary.rows) ? summary.rows : legacyQtlRows(summary.items));
                    if (!Array.isArray(summary.rows)) {
                        for (const legacy of publicQtlRows(legacyQtlRows(card.key_evidence))) {
                            if (!rows.some(row => ['variant_id', 'qtl_type', 'biosample', 'study_id', 'p_value', 'beta']
                                .every(field => row[field] === legacy[field]))) rows.push(legacy);
                        }
                    }
                    if (rows.length) categories[key].rows = rows;
                }
                if (['coloc', 'e2g', 'expression', 'function'].includes(key) && Array.isArray(summary.preview)) {
                    categories[key].preview = summary.preview.filter(row => row && row.kind === key
                        && /^sha256:[a-f0-9]{64}$/.test(row.source_ref || '')).slice(0, 3).map(row => ({
                            kind: key, summary: text(row.summary, 1200), source: list(row.source, 32, 1200),
                            evidence_id: text(row.evidence_id, 300), source_ref: row.source_ref,
                            ...(row.group ? { group: text(row.group, 300) } : {}),
                            ...(row.detail ? { detail: text(row.detail, 1200) } : {}) }));
                }
                if (key === 'association' && Array.isArray(summary.scope_source))
                    categories[key].scope_source = list(summary.scope_source, 9, 1200);
            }
            cards.push({ gene_id: card.gene_id, symbol: text(card.symbol, 60) || card.gene_id,
                rank: Number.isSafeInteger(card.rank) && card.rank > 0 ? card.rank : null,
                ...(Number.isSafeInteger(card.shared_rank) && card.shared_rank > 0 && Number.isSafeInteger(card.rank)
                    && card.shared_rank <= card.rank ? { shared_rank: card.shared_rank, tied: card.tied === true } : {}),
                ...(card.tier ? { tier: card.tier } : {}),
                tier_basis: list(card.tier_basis, 5, 300), tss_distance_bp: count(card.tss_distance_bp),
                distance_rank: count(card.distance_rank), key_evidence: list(card.key_evidence, 5, 300),
                rationale: text(card.rationale, 1200), categories,
                model_selection: ['selected', 'not_selected', 'unresolved'].includes(card.model_selection) ? card.model_selection : null,
                selection_reason: text(card.selection_reason, 1200),
                selection_evidence_ids: list(card.selection_evidence_ids, 256, 300) });
        }
        return cards.length ? cards : null;
    }

    function geneCardRankLabel(card) {
        if (card.rank === null) return '—';
        return card.tied ? `${card.shared_rank}=` : String(card.shared_rank ?? card.rank);
    }

    function geneCardDistance(bp) {
        if (bp === null) return '—';
        return bp >= 1000 ? `${(bp / 1000).toFixed(1)} kb` : `${bp} bp`;
    }

    function geneCardTags(card) {
        const linked = key => ['observed', 'partial'].includes(card.categories[key]?.status) && card.categories[key]?.count;
        return [...(card.categories.coding?.tags || (linked('coding') ? ['Consequence'] : [])),
            linked('qtl') && `QTL ${card.categories.qtl.count}`,
            ...(card.categories.coloc?.tags || (linked('coloc') ? ['Coloc'] : [])), linked('e2g') && 'E2G'].filter(Boolean);
    }

    function geneCardDetail(card) {
        const detail = document.createElement('div');
        detail.className = 'gene-card-detail';
        const explanation = card.rationale || card.selection_reason;
        if (explanation) {
            const label = document.createElement('div');
            label.className = 'gene-card-rationale-label';
            label.textContent = 'LLM rationale';
            if (card.rank !== null && card.model_selection && card.model_selection !== 'selected') {
                label.textContent += ' · exploratory ranking';
            }
            const rationale = document.createElement('p');
            rationale.className = 'gene-card-rationale';
            rationale.textContent = explanation;
            detail.append(label, rationale);
        }
        const grid = document.createElement('div');
        grid.className = 'gene-card-categories';
        for (const [key, title] of GENE_CARD_CATEGORIES) {
            const summary = card.categories[key];
            if (!summary) continue;
            const block = document.createElement('div');
            block.className = `gene-card-category gene-card-status-${summary.status}` + (key === 'qtl' ? ' gene-card-qtl' : '');
            const heading = document.createElement('div');
            heading.className = 'gene-card-category-title';
            const name = document.createElement('span');
            name.textContent = title + (summary.count !== null ? ` (${summary.count})` : '');
            const status = document.createElement('span');
            status.className = 'gene-card-status';
            status.textContent = GENE_CARD_STATUS_LABELS[summary.status];
            heading.append(name, status);
            block.appendChild(heading);
            if (key === 'qtl' && summary.rows?.length) {
                const rows = document.createElement('div'); rows.className = 'qtl-visible-rows';
                let displayed = 0;
                const appendRows = size => {
                    const end = Math.min(displayed + size, summary.rows.length);
                    while (displayed < end) rows.appendChild(createQtlDisplayRow(summary.rows[displayed++]));
                };
                appendRows(3);
                block.appendChild(rows);
                if (summary.rows.length > 3) {
                    const controls = document.createElement('p'); controls.className = 'qtl-coverage qtl-row-controls';
                    const count = document.createElement('span'); count.className = 'qtl-row-count';
                    count.setAttribute('aria-live', 'polite');
                    const more = document.createElement('button'); more.type = 'button';
                    more.className = 'qtl-show-more evidence-preview-control';
                    more.setAttribute('aria-label', 'Show up to 50 more molecular QTL records');
                    const collapse = document.createElement('button'); collapse.type = 'button';
                    collapse.className = 'qtl-collapse evidence-preview-control'; collapse.textContent = 'Collapse';
                    const update = () => {
                        count.textContent = `${displayed}/${summary.rows.length} shown`;
                        more.textContent = `Show ${Math.min(50, summary.rows.length - displayed)} more`;
                        more.hidden = displayed === summary.rows.length;
                        collapse.hidden = displayed <= 3;
                    };
                    more.onclick = () => { appendRows(50); update(); };
                    collapse.onclick = () => {
                        while (rows.children.length > 3) rows.children[rows.children.length - 1].remove();
                        displayed = Math.min(3, summary.rows.length); update();
                    };
                    controls.append(count, more, collapse); update(); block.appendChild(controls);
                }
            } else if (summary.preview?.length) {
                let previousGroup = '';
                for (const record of summary.preview) {
                    if (record.group && record.group !== previousGroup) {
                        const group = document.createElement('p'); group.className = 'evidence-preview-group';
                        group.textContent = record.group; block.appendChild(group); previousGroup = record.group;
                    }
                    const entry = document.createElement('div'); entry.className = 'evidence-preview-row';
                    const line = document.createElement('span'); line.textContent = record.summary; entry.appendChild(line);
                    if (record.detail) {
                        const second = document.createElement('p'); second.className = 'evidence-preview-detail';
                        second.textContent = record.detail; entry.appendChild(second);
                    }
                    const source = document.createElement('details'); source.className = 'qtl-source-details';
                    const sourceTitle = document.createElement('summary'); sourceTitle.textContent = 'Source'; source.appendChild(sourceTitle);
                    const sourceBody = document.createElement('div'); sourceBody.className = 'qtl-source-body';
                    const sources = [...record.source,
                        ...(record.evidence_id ? [`Evidence record: ${record.evidence_id}`] : []),
                        `Source record: ${record.source_ref}`];
                    for (const text of sources) { const p = document.createElement('p'); p.textContent = text; sourceBody.appendChild(p); }
                    source.appendChild(sourceBody); entry.appendChild(source); block.appendChild(entry);
                }
                if (summary.count > summary.preview.length) {
                    const note = document.createElement('p'); note.className = 'qtl-coverage';
                    note.textContent = `${summary.preview.length}/${summary.count} previewed · complete records retained`;
                    block.appendChild(note);
                }
            } else if (summary.items.length) {
                const items = document.createElement('ul');
                for (const item of summary.items) {
                    if (key === 'qtl' && /QTL credible sets overlap the locus$/.test(item)) continue;
                    const entry = document.createElement('li');
                    entry.textContent = key === 'coloc' && item === 'No retained GWAS-QTL result in the allowed phenotype and frozen biosample context.'
                        ? 'No qualifying GWAS–QTL result retained.' : item;
                    items.appendChild(entry);
                }
                block.appendChild(items);
            }
            if (summary.scope_source?.length) {
                const source = document.createElement('details'); source.className = 'qtl-source-details';
                const label = document.createElement('summary'); label.textContent = 'Source'; source.appendChild(label);
                const body = document.createElement('div'); body.className = 'qtl-source-body';
                for (const line of summary.scope_source) {
                    const p = document.createElement('p'); p.textContent = line; body.appendChild(p);
                }
                source.appendChild(body); block.appendChild(source);
            }
            grid.appendChild(block);
        }
        detail.appendChild(grid);
        return detail;
    }

    function geneCardsSummary(cards) {
        if (cards.some(card => card.model_selection)) {
            const selected = cards.filter(card => card.model_selection === 'selected').length;
            return `${cards.length} candidate ${cards.length === 1 ? 'gene' : 'genes'} · ${selected} shortlisted`;
        }
        const ranked = cards.some(card => card.rank !== null);
        return `${cards.length} ${cards.length === 1 ? 'gene' : 'genes'}`
            + (ranked ? ' · ranked candidates' : ' · listed by distance, not ranked');
    }

    function geneEvidenceTable(cards) {
        const table = document.createElement('table');
        table.className = 'gene-evidence-table';
        const head = document.createElement('tr');
        for (const title of ['Rank', 'Gene', 'TSS', 'Locus links']) {
            const cell = document.createElement('th');
            cell.textContent = title;
            head.appendChild(cell);
        }
        table.appendChild(head);
        for (const card of cards) {
            const row = document.createElement('tr');
            row.className = 'gene-card-row';
            const values = [geneCardRankLabel(card), card.symbol, geneCardDistance(card.tss_distance_bp),
                geneCardTags(card)];
            values.forEach((item, index) => {
                const cell = document.createElement('td');
                if (index === 3) {
                    if (!item.length) cell.textContent = 'Position only';
                    for (const tag of item) {
                        const label = document.createElement('span');
                        label.className = 'gene-card-tag';
                        label.textContent = tag;
                        cell.appendChild(label);
                    }
                } else if (index === 1) {
                    const toggle = document.createElement('button');
                    toggle.type = 'button';
                    toggle.className = 'gene-card-toggle';
                    toggle.textContent = String(item);
                    toggle.setAttribute('aria-expanded', 'false');
                    cell.appendChild(toggle);
                    if (card.model_selection === 'selected') {
                        const badge = document.createElement('span'); badge.className = 'gene-card-tag gene-card-shortlist';
                        badge.textContent = 'Shortlist'; cell.appendChild(badge);
                    }
                } else {
                    cell.textContent = String(item);
                }
                row.appendChild(cell);
            });
            const detailRow = document.createElement('tr');
            detailRow.className = 'gene-card-detail-row';
            detailRow.hidden = true;
            const detailCell = document.createElement('td');
            detailCell.setAttribute('colspan', '4');
            detailCell.appendChild(geneCardDetail(card));
            detailRow.appendChild(detailCell);
            row.addEventListener?.('click', () => {
                detailRow.hidden = !detailRow.hidden;
                row.querySelector?.('.gene-card-toggle')?.setAttribute('aria-expanded', String(!detailRow.hidden));
            });
            table.append(row, detailRow);
        }
        return table;
    }

    function openGeneEvidenceCardsModal(cards, variantLabels = null) {
        if (!cards?.length) return null;
        document.getElementById('gene-cards-modal')?.remove();
        const overlay = document.createElement('div');
        overlay.id = 'gene-cards-modal';
        overlay.className = 'settings-overlay gene-cards-overlay';
        const panel = document.createElement('div');
        panel.className = 'settings-panel gene-cards-panel';
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        panel.setAttribute('aria-labelledby', 'gene-cards-title');
        const heading = document.createElement('div');
        heading.className = 'settings-heading';
        const titles = document.createElement('div');
        const title = document.createElement('h2');
        title.id = 'gene-cards-title';
        title.textContent = 'Gene evidence card';
        const subtitle = document.createElement('p');
        subtitle.className = 'gene-cards-subtitle';
        subtitle.textContent = `${geneCardsSummary(cards)}. Select a gene to show its evidence card.`;
        titles.append(title, subtitle);
        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'settings-close';
        close.setAttribute('aria-label', 'Close gene evidence cards');
        close.textContent = '×';
        heading.append(titles, close);
        panel.append(heading, geneEvidenceTable(cards));
        if (cards.some(card => card.categories.qtl?.rows?.some(row => row.variant_origin === 'ld_proxy'
            || row.locus_anchor_witnesses?.some(item => item.variant_origin === 'ld_proxy')))) {
            const legend = document.createElement('p'); legend.className = 'qtl-origin-legend';
            legend.textContent = 'LD-linked variant: an additional variant included through LD with a user-selected variant.';
            panel.appendChild(legend);
        }
        overlay.appendChild(panel);
        const dismiss = () => {
            overlay.remove();
            document.removeEventListener?.('keydown', onKey);
        };
        const onKey = event => { if (event.key === 'Escape') dismiss(); };
        close.addEventListener?.('click', dismiss);
        overlay.addEventListener?.('click', event => { if (event.target === overlay) dismiss(); });
        document.addEventListener?.('keydown', onKey);
        document.body.appendChild(overlay);
        window.GWASMindVariants?.decorate(panel, variantLabels);
        return overlay;
    }

    function renderGeneEvidenceCards(target, value, variantLabels = undefined) {
        if (!target) return false;
        if (variantLabels !== undefined) target.variantLabels = window.GWASMindVariants?.cleanLabels(variantLabels) || [];
        const cards = value === undefined ? target.geneEvidenceCards : publicGeneEvidenceCards(value);
        target.geneEvidenceCards = cards || null;
        window.decorateQtlEvidence?.(target, { cards });
        const aliases = new Map((publicVariantNormalizations(target.reportVariantNormalizations) || []).map(row => [row.original_id, row.canonical_id]));
        const measured = [...(target.querySelectorAll?.('li') || [])].map(item => item.qtlDisplaySource).filter(Boolean);
        for (const card of cards || []) for (const row of card.categories.qtl?.rows || []) {
            if (!row.legacy_display) continue;
            const origins = new Set(measured.filter(item => item.variant_id === row.locus_variant_id).map(item => item.variant_origin));
            if (aliases.has(row.locus_variant_id)) { row.variant_id = aliases.get(row.locus_variant_id); row.variant_origin = 'source_selected'; }
            else if (origins.size === 1) row.variant_origin = [...origins][0];
            row.variant_label = { source_selected: 'User-selected variant', ld_proxy: 'LD-linked variant' }[row.variant_origin] || 'Analyzed variant';
        }
        target.querySelector?.('.gene-evidence-cards')?.remove();
        const toolbar = target.querySelector?.(':scope > .ranking-evidence-downloads');
        toolbar?.classList.remove('has-gene-card');
        if (!cards?.length) return false;
        const entry = document.createElement('button');
        entry.type = 'button';
        entry.className = 'gene-evidence-cards';
        entry.setAttribute('aria-haspopup', 'dialog');
        entry.textContent = 'Gene evidence card';
        entry.addEventListener?.('click', () => openGeneEvidenceCardsModal(target.geneEvidenceCards, target.variantLabels));
        if (toolbar) {
            toolbar.classList.add('has-gene-card');
            toolbar.prepend(entry);
        } else {
            target.prepend(entry);
        }
        window.GWASMindVariants?.decorate(target, target.variantLabels);
        return true;
    }

    function renderMarkdown(target, content, cursor, reviewUpdate = null) {
        const source = String(content || "");
        const safeDom = window.GWASMindSafe;
        const markdown = safeDom && typeof safeDom.renderMarkdown === "function"
            ? safeDom.renderMarkdown(source)
            : (safeDom?.escapeText(source) || source.replace(/[&<>"']/g, (character) => ({
                "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
            }[character]))).replace(/\n/g, "<br>");
        target.innerHTML = markdown;
        decorateRenderedReport(target);
        if (reviewUpdate) target.scientificReviewReceipt = publicReviewUpdate(reviewUpdate);
        window.decorateReportWarnings?.(target, source, cursor ? null : target.scientificReviewReceipt);
        if (!cursor) window.decorateReportStatus?.(target);
        if (cursor) {
            const marker = document.createElement("span");
            marker.className = "cursor-blink";
            marker.textContent = "▋";
            target.appendChild(marker);
        } else if (typeof window.attachReportActions === 'function') {
            window.attachReportActions(target, source);
        }
        renderRankingEvidence(target);
        renderGeneEvidenceCards(target);
        window.GWASMindVariants?.decorate(target, target.variantLabels);
    }

    function decorateRenderedReport(target) {
        if (!target || typeof target.querySelectorAll !== "function") return;
        target.querySelectorAll("p, h1, h2, h3, h4, h5, h6").forEach((element) => {
            const text = String(element.textContent || "").trim();
            const sectionTitle = text.replace(/:$/, "");
            if (REPORT_SECTION_TITLES.has(sectionTitle)) {
                element.classList.add("report-section-heading");
                return;
            }
            const quality = text.match(
                /^(?:Validation status|Report quality):\s*(PASS_WITH_WARNINGS|PASS|STOPPED)$/i
            );
            if (!quality) return;
            const status = quality[1].toUpperCase();
            // Keep rolling-upgrade responses visually consistent with the new label.
            element.textContent = `Report quality: ${status}`;
            element.classList.add("report-quality", `report-quality-${status.toLowerCase()}`);
        });
    }

    function updateCardContentStreaming(index, content, scope = currentRegionCardScope, reviewUpdate = null) {
        const loading = document.getElementById(`loading-${scope}-${index}`);
        const target = document.getElementById(`content-${scope}-${index}`);
        if (loading) loading.style.display = "none";
        if (!target) return;
        target.style.display = "block";
        renderMarkdown(target, content, true, reviewUpdate);
    }

    function updateCardContent(index, content, scope = currentRegionCardScope, reviewUpdate = null) {
        const loading = document.getElementById(`loading-${scope}-${index}`);
        const target = document.getElementById(`content-${scope}-${index}`);
        if (loading) loading.style.display = "none";
        if (!target) return;
        target.style.display = "block";
        renderMarkdown(target, content || "No evidence summary was returned.", false, reviewUpdate);
    }

    function updateCardError(index, error, scope = currentRegionCardScope) {
        const loading = document.getElementById(`loading-${scope}-${index}`);
        const target = document.getElementById(`content-${scope}-${index}`);
        if (loading) loading.style.display = "none";
        if (!target) return;
        target.style.display = "block";
        target.replaceChildren();
        const message = document.createElement("div");
        message.className = "card-error";
        message.textContent = `❌ ${error}`;
        target.appendChild(message);
        renderRankingEvidence(target);
    }

    function variantIdForSnp(snp) {
        if (!snp || typeof snp !== "object") return "";
        const chromosome = String(snp.chromosome || "").replace(/^chr/i, "");
        const ref = String(snp.ref || "").toUpperCase();
        const alt = String(snp.alt || "").toUpperCase();
        if (chromosome && Number.isFinite(Number(snp.position)) && Number(snp.position) > 0
                && ref && alt && ref !== "N" && alt !== "N") {
            return `${chromosome}_${Number(snp.position)}_${ref}_${alt}`;
        }
        return String(snp.snp_id || "");
    }

    function effectivePValue(snp) {
        const rawPValue = snp?.pvalue;
        if (rawPValue !== null && rawPValue !== undefined && String(rawPValue).trim() !== "") {
            const pvalue = Number(rawPValue);
            if (Number.isFinite(pvalue) && pvalue >= 0 && pvalue <= 1) {
                return pvalue === 0 ? Number.MIN_VALUE : pvalue;
            }
        }
        const rawNegLogP = snp?.neg_log_p;
        if (rawNegLogP !== null && rawNegLogP !== undefined && String(rawNegLogP).trim() !== "") {
            const negLogP = Number(rawNegLogP);
            if (Number.isFinite(negLogP) && negLogP >= 0) {
                return Math.pow(10, -Math.min(negLogP, 323));
            }
        }
        return Number.POSITIVE_INFINITY;
    }

    function selectVariantsForRegion(region, limit = MAX_CHAT_VARIANTS) {
        const snps = Array.isArray(region?.snps) ? region.snps : [];
        const ranked = snps.map((snp, index) => ({ snp, index, pvalue: effectivePValue(snp) }))
            .sort((left, right) => left.pvalue - right.pvalue || left.index - right.index);
        const variantIds = [];
        const seen = new Set();
        for (const item of ranked) {
            const id = variantIdForSnp(item.snp);
            if (!id || seen.has(id)) continue;
            seen.add(id);
            variantIds.push(id);
            if (variantIds.length >= Math.max(0, Number(limit) || 0)) break;
        }
        return {
            variantIds,
            totalCount: snps.length,
            selectedCount: variantIds.length,
            truncated: snps.length > variantIds.length
        };
    }

    function selectionDisclosure(selection) {
        if (!selection?.totalCount) return "No selected SNPs were supplied.";
        if (selection.truncated) {
            return `Analyzing ${selection.selectedCount.toLocaleString()} of ${selection.totalCount.toLocaleString()} selected SNPs, ranked by p-value.`;
        }
        return `Analyzing all ${selection.selectedCount.toLocaleString()} selected SNPs.`;
    }

    function multiRegionCompletionMessage(successCount, failureCount, totalCount) {
        if (failureCount === 0) return `✅ All ${totalCount} regions queried`;
        if (successCount === 0) return `❌ All ${totalCount} regions failed`;
        return `⚠️ ${successCount} regions succeeded; ${failureCount} failed`;
    }

    function scopedCardContent(selection, content) {
        const scope = selectionDisclosure(selection);
        return `**Evidence scope:** ${scope}\n\n${content || "No evidence summary was returned."}`;
    }

    async function consumeSse(response, onEvent, signal) {
        if (!response.ok || !response.body) {
            let message = `HTTP ${response.status}`;
            try {
                const error = await response.json();
                message = error.error || message;
            } catch (ignored) {}
            throw new Error(message);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let sawDone = false;
        let sawComplete = false;
        try {
        while (true) {
            if (signal?.aborted) throw abortError();
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            buffer = buffer.replace(/\r\n/g, "\n");
            const blocks = buffer.split("\n\n");
            buffer = blocks.pop() || "";
            for (const block of blocks) {
                const data = block.split("\n")
                    .filter((line) => line.startsWith("data:"))
                    .map((line) => line.slice(5).trimStart())
                    .join("\n");
                if (!data) continue;
                if (data === "[DONE]") {
                    sawDone = true;
                    continue;
                }
                let parsed;
                try {
                    parsed = JSON.parse(data);
                } catch (error) {
                    throw new Error("The response stream contained invalid JSON");
                }
                if (parsed.type === "complete") sawComplete = true;
                onEvent(parsed);
            }
            if (done) break;
        }
        if (buffer.trim()) throw new Error("The response stream ended mid-event");
        if (!sawDone || !sawComplete) {
            throw new Error("The response stream ended before completion");
        }
        } catch (error) {
            try { await reader.cancel?.(); } catch (_cancelError) {}
            throw error;
        } finally {
            reader.releaseLock?.();
        }
    }

    async function requestAnswer(body, onEvent, request) {
        let streamRunId = null;
        const response = await fetch("/api/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: request.controller.signal
        });
        await consumeSse(response, (event) => {
            if (!isCurrentChatRequest(request)) throw abortError();
            if (event.type === 'run' && typeof event.content === 'string' && /^[a-f0-9]{32}$/.test(event.content)) {
                streamRunId = event.content;
            }
            if (event.type === 'ranking_evidence') {
                const snapshot = publicRankingEvidence(event);
                if (!snapshot || snapshot.run_id !== streamRunId) return;
                event = { type: 'ranking_evidence', ...snapshot };
            }
            onEvent(event);
            if (event.type === "run" && /^[a-f0-9]{32}$/.test(String(event.content || ""))) {
                window.recordWorkspaceChatEvent?.({ type: "run", run_id: event.content });
            } else if (event.type === 'ranking_evidence') {
                window.recordWorkspaceChatEvent?.(event);
            } else if (event.type === 'gene_evidence_cards') {
                const cards = publicGeneEvidenceCards(event.cards);
                if (cards) window.recordWorkspaceChatEvent?.({ type: 'gene_evidence_cards', cards,
                    ...(Array.isArray(event.variant_labels)
                        ? { variant_labels: window.GWASMindVariants?.cleanLabels(event.variant_labels) || [] } : {}) });
            } else if (event.type === "review_update") {
                const update = publicReviewUpdate(event);
                if (update) window.recordWorkspaceChatEvent?.(update);
            } else if (event.type === "analysis_status") {
                window.recordWorkspaceChatEvent?.(event);
            } else if (event.type === "variant_normalization") {
                const normalizations = publicVariantNormalizations(event.normalizations);
                if (normalizations) window.recordWorkspaceChatEvent?.({ type: event.type, normalizations });
            } else if (event.type === "stage" && CHAT_PROGRESS_STAGE_IDS.has(event.stage)) {
                const states = { pending: "pending", running: "active", passed: "complete",
                    warning: "warning", stopped: "stopped", failed: "error" };
                const state = states[event.state] || "active";
                const metrics = {};
                for (const key of ["total_records", "candidate_genes", "elapsed_seconds",
                    "stage_elapsed_seconds", "ranking_elapsed_seconds", "estimated_output_tokens",
                    "total_tokens", "output_tokens", "received_chars", "warning_count"]) {
                    const count = boundedPublicInteger(event.metrics?.[key]);
                    if (count !== null) metrics[key] = count;
                }
                window.recordWorkspaceChatEvent?.({ type: "stage", stage: event.stage,
                    ...(SCIENTIFIC_TASKS.has(event.requested_task) ? { requested_task: event.requested_task } : {}),
                    ...(event.stage === 'response' && RESPONSE_SUBSTAGES.has(event.substage) ? { substage: event.substage } : {}),
                    ...(event.stage === 'response' && event.ranking_unavailable === true ? { ranking_unavailable: true } : {}),
                    ...(publicModelUsage(event.model_usage) ? { model_usage: publicModelUsage(event.model_usage) } : {}),
                    state: Object.hasOwn(states, event.state) ? event.state : "running",
                    summary: publicStructuredStageSummary(event.stage, event.summary, state),
                    detail: event.detail ? publicChatStatus(event.detail) : "", metrics,
                    operation_timings: operationTimings(request.progress?.stages.get(event.stage)),
                    ...(publicRetrievalProgress(event.progress) ? { progress: publicRetrievalProgress(event.progress) } : {}) });
            } else if (event.type === 'complete') {
                for (const stage of request.progress?.stages.values() || []) {
                    window.recordWorkspaceChatEvent?.({ type: 'operation_timings', stage: stage.id,
                        operation_timings: operationTimings(stage) });
                }
            }
        }, request.controller.signal);
    }

    async function querySingleRegion(question, region, index, request, onStatus = null, history = getChatHistory()) {
        const selection = selectVariantsForRegion(region);
        let content = "";
        let reviewUpdate = null;
        try {
            await requestAnswer({
                question,
                region: {
                    chromosome: region.chromosome,
                    start: region.start,
                    end: region.end
                },
                variants: selection.variantIds,
                history
            }, (event) => {
                if (event.type === "status" || event.type === "stage" || event.type === "review_update" || event.type === "variant_normalization" || event.type === 'analysis_status') {
                    if (event.type === 'review_update') reviewUpdate = publicReviewUpdate(event) || reviewUpdate;
                    if (event.type === 'variant_normalization') {
                        const target = document.getElementById(`content-${request.generation}-${index}`);
                        if (target) target.reportVariantNormalizations = publicVariantNormalizations(event.normalizations);
                    }
                    if (typeof onStatus === "function") {
                        onStatus(event);
                    }
                } else if (event.type === 'ranking_evidence') {
                    const target = document.getElementById(`content-${request.generation}-${index}`);
                    if (target) {
                        target.style.display = 'block';
                        renderRankingEvidence(target, [...(target.rankingEvidence || []), event]);
                    }
                } else if (event.type === 'gene_evidence_cards') {
                    const target = document.getElementById(`content-${request.generation}-${index}`);
                    if (target) {
                        target.style.display = 'block';
                        renderGeneEvidenceCards(target, event.cards, event.variant_labels);
                    }
                } else if (event.type === "answer_chunk") {
                    content += String(event.content || "");
                    updateCardContentStreaming(index, scopedCardContent(selection, content), request.generation, reviewUpdate);
                } else if (event.type === "complete") {
                    if (!content.trim()) throw new Error("No evidence summary was returned");
                    updateCardContent(index, scopedCardContent(selection, content), request.generation, reviewUpdate);
                } else if (event.type === "error") {
                    throw new Error(event.content || "Analysis failed");
                }
            }, request);
            if (!isCurrentChatRequest(request)) throw abortError();
            if (!content.trim()) throw new Error("No evidence summary was returned");
            updateCardContent(index, scopedCardContent(selection, content), request.generation, reviewUpdate);
            return { ok: true, content, selection };
        } catch (error) {
            if (isAbortError(error) || !isCurrentChatRequest(request)) {
                updateCardError(index, "Request cancelled", request.generation);
                throw abortError();
            }
            updateCardError(index, error.message, request.generation);
            return { ok: false, error: error.message, selection };
        }
    }

    async function queryMultipleRegions(question, request) {
        const regions = ManhattanPlot.getMultiSelectRegions();
        if (!regions.length) {
            addMessage("system", "Please select one or more regions first.");
            return;
        }
        addMessage("user", question);
        const priorHistory = getChatHistory();
        addToChatHistory("user", question);
        currentRegionCardScope = request.generation;
        const messages = document.getElementById("chat-messages");
        const progress = createTransientChatStatus(
            `Preparing ${regions.length} selected regions...`
        );
        request.progress = progress;
        const cards = document.createElement("div");
        cards.className = "responses-container";
        regions.forEach((region, index) => cards.appendChild(createRegionResponseCard(region, index)));
        messages.appendChild(cards);

        let successCount = 0;
        let failureCount = 0;
        const summaries = [];
        try {
            for (let index = 0; index < regions.length; index += 1) {
                if (!isCurrentChatRequest(request)) throw abortError();
                if (index > 0) restartProgressForRegion(progress);
                const selection = selectVariantsForRegion(regions[index]);
                appendChatProgressLine(
                    progress,
                    `Region ${index + 1}/${regions.length} · ${selectionDisclosure(selection)}`
                );
                const result = await querySingleRegion(
                    question,
                    regions[index],
                    index,
                    request,
                    (event) => {
                        if (event.type === "stage" || event.type === "review_update" || event.type === "variant_normalization" || event.type === 'analysis_status') {
                            appendPublicChatStageEvent(progress, event);
                            return;
                        }
                        const safe = publicChatStatus(event.content);
                        appendChatProgressLine(
                            progress,
                            `Region ${index + 1}/${regions.length} · ${safe}`,
                            inferProgressStage(safe)
                        );
                    },
                    priorHistory
                );
                if (result.ok) {
                    successCount += 1;
                    summaries.push(`Region ${index + 1}: ${result.content}`);
                } else {
                    failureCount += 1;
                    summaries.push(`Region ${index + 1}: No report was generated. ${result.error}`);
                }
            }
            if (!isCurrentChatRequest(request)) throw abortError();
        } catch (error) {
            if (isAbortError(error) || !isCurrentChatRequest(request)) {
                progress.root.remove();
                throw abortError();
            }
            throw error;
        }
        appendChatProgressLine(
            progress,
            multiRegionCompletionMessage(successCount, failureCount, regions.length)
        );
        completeChatProgress(progress);
        if (summaries.length) addToChatHistory("assistant", summaries.join("\n\n"));
    }

    function createTransientChatStatus(initialText, destination, options = {}) {
        const messages = destination || document.getElementById("chat-messages");
        const status = document.createElement("div");
        status.className = "shared-thinking analysis-progress";
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        status.setAttribute("aria-label", "Analysis progress");
        const log = document.createElement("div");
        log.className = "thinking-log progress-stage-list";
        status.appendChild(log);
        messages.appendChild(status);
        const progress = {
            root: status,
            log,
            messages,
            stages: new Map(),
            activeStageId: null,
            detailCount: 0,
            complete: false,
            historical: Boolean(options.historical),
            restoring: Boolean(options.restoring),
            requestedTask: SCIENTIFIC_TASKS.has(options.requestedTask) ? options.requestedTask : null,
            timer: null
        };
        const cardId = ++progressCardSequence;
        CHAT_PROGRESS_STAGES.forEach((definition, index) => {
            const section = document.createElement("section");
            section.className = "progress-stage progress-stage-pending";
            section.setAttribute("data-stage", definition.id);

            const header = document.createElement("button");
            header.className = "progress-stage-header";
            header.setAttribute("type", "button");
            header.setAttribute("aria-expanded", "false");
            const marker = document.createElement("span");
            marker.className = "progress-stage-marker";
            marker.textContent = String(index + 1);
            marker.setAttribute("aria-hidden", "true");
            const heading = document.createElement("span");
            heading.className = "progress-stage-heading";
            const title = document.createElement("span");
            title.className = "progress-stage-title";
            title.textContent = stageTitle(definition.id, progress.requestedTask);
            const duration = document.createElement("span");
            duration.className = "progress-stage-duration";
            duration.id = `progress-${cardId}-${definition.id}-duration`;
            duration.setAttribute("aria-live", "off");
            const summary = document.createElement("span");
            summary.className = "progress-stage-summary";
            summary.id = `progress-${cardId}-${definition.id}-summary`;
            summary.textContent = "Waiting";
            header.setAttribute('aria-describedby', `${summary.id} ${duration.id}`);
            heading.append(title, duration, summary);
            const tokens = document.createElement("span");
            tokens.className = "progress-stage-token-count";
            tokens.id = `progress-${cardId}-${definition.id}-tokens`;
            tokens.setAttribute("aria-live", "off");
            tokens.hidden = true;
            if (definition.id === 'response') {
                heading.replaceChildren(title, duration, tokens, summary);
                header.setAttribute('aria-describedby', `${summary.id} ${duration.id} ${tokens.id}`);
            }
            const chevron = document.createElement("span");
            chevron.className = "progress-stage-toggle-label";
            chevron.textContent = "Details";
            chevron.setAttribute("aria-hidden", "true");
            header.append(marker, heading, chevron);

            const body = document.createElement('div');
            body.className = 'progress-stage-body';
            body.id = `progress-${cardId}-${definition.id}-body`;
            body.hidden = true;
            const details = document.createElement("div");
            details.className = "progress-stage-details";
            details.id = `progress-${cardId}-${definition.id}-details`;
            header.setAttribute('aria-controls', body.id);
            details.hidden = true;
            const operationList = document.createElement('div');
            operationList.className = 'progress-operation-list';
            const diagnostics = document.createElement('details');
            diagnostics.className = 'progress-diagnostics';
            diagnostics.hidden = true;
            const diagnosticTitle = document.createElement('summary');
            diagnosticTitle.textContent = 'Technical details';
            diagnosticTitle.title = definition.id === 'response' ? 'LLM request and retry details for troubleshooting'
                : 'Data requests, retries, cache use and transfer checks for troubleshooting';
            const diagnosticBody = document.createElement('div');
            diagnostics.append(diagnosticTitle, diagnosticBody);
            const scope = document.createElement('div');
            scope.className = 'progress-scope'; scope.hidden = true;
            const phaseRail = document.createElement('div');
            phaseRail.className = 'progress-phase-rail'; phaseRail.hidden = true;
            phaseRail.setAttribute('aria-label', definition.id === 'retrieve' ? 'Evidence retrieval phases' : 'Interpretation phases');
            const activity = document.createElement('span');
            activity.className = 'progress-stage-activity';
            activity.setAttribute('aria-live', 'off');
            activity.hidden = true;
            if (definition.id === 'retrieve') body.appendChild(activity);
            header.onclick = () => {
                const stage = progress.stages.get(definition.id);
                if (stage) stage.userExpanded = body.hidden;
                setProgressStageExpanded(
                    stage,
                    body.hidden
                );
            };
            body.append(scope, phaseRail, details);
            if (['retrieve', 'response'].includes(definition.id)) body.appendChild(diagnostics);
            section.append(header, body);
            log.appendChild(section);
            progress.stages.set(definition.id, {
                ...definition,
                titleElement: title,
                index,
                section,
                header,
                body, toggleLabel: chevron,
                marker,
                duration,
                tokens,
                summary,
                activity,
                details,
                operationList, operationRows: new Map(),
                diagnostics, diagnosticBody, scope, phaseRail, phaseStates: new Map(),
                retrievalSteps: new Map(),
                currentRetrieval: null,
                scientificRetrieval: null,
                currentSubstage: null,
                seen: new Set(),
                telemetry: null,
                responseMetrics: {},
                userExpanded: null,
                state: "pending",
                progress,
                startedAt: null,
                durationSeconds: null,
                clockAnchor: null,
                clockBaseSeconds: 0,
                clockFrozen: Boolean(options.historical)
            });
        });
        if (!progress.historical) {
            appendChatProgressLine(progress, initialText, "plan");
            liveProgressCards.add(progress);
            scheduleProgressClock(progress);
        }
        return progress;
    }

    function progressNow() {
        return typeof performance !== "undefined" && typeof performance.now === "function"
            ? performance.now() : Date.now();
    }

    function restartProgressForRegion(progress) {
        // One question may submit several independent loci. Keep the same four
        // rows and their prior details, but do not mistake the next locus's
        // planning events for delayed events from the completed previous locus.
        if (!progress || progress.historical || progress.complete) return;
        stopProgressClock(progress);
        progress.activeStageId = null;
        progress.scopeCounts = {}; progress.explicitScopeCounts = new Set();
        progress.ldUnavailable = false;
        progress.retainedRankingProof = null;
        for (const stage of progress.stages.values()) {
            stage.startedAt = null;
            stage.durationSeconds = null;
            stage.clockAnchor = null;
            stage.clockBaseSeconds = 0;
            stage.clockFrozen = false;
            stage.hasServerClock = false;
            stage.holdClockForServer = false;
            stage.responseMetrics = {};
            stage.modelUsage = null; stage.modelCalls?.remove(); stage.modelCalls = null;
            stage.evidenceOnlyRecovery = false;
            stage.rankingUnavailable = false;
            stage.retrievalFailed = false; stage.retrievalBlocked = false;
            stage.safetyWarnings = [];
            stage.currentRetrieval = null;
            stage.scientificRetrieval = null;
            stage.currentSubstage = null; stage.section.removeAttribute?.('data-current-substage');
            stage.scope.textContent = ''; stage.scope.hidden = true;
            stage.retrievalSteps.clear();
            stage.operationRows.clear(); stage.operationList.replaceChildren(); stage.operationList.remove(); stage.currentOperationKey = null;
            stage.providerGroups = new Map(); stage.ldSessionOpen = false;
            stage.phaseStates.clear(); stage.phaseRail.replaceChildren(); stage.phaseRail.hidden = true;
            stage.gateReview?.remove(); stage.methodNotes?.remove(); stage.operationReview?.remove();
            stage.scientificReview = null;
            stage.details.classList.remove('progress-scientific-details');
            stage.activity.hidden = true;
            stage.summary.classList.remove("progress-safety-warnings");
            stage.seen.clear();
            stage.summary.textContent = 'Waiting';
            stage.tokens.hidden = true;
            setProgressStageState(stage, 'pending');
            setProgressStageExpanded(stage, false);
        }
        liveProgressCards.add(progress);
        scheduleProgressClock(progress);
    }

    function elapsedStageSeconds(stage) {
        if (stage.clockFrozen) return stage.durationSeconds;
        if (stage.clockAnchor === null) return stage.durationSeconds;
        return Math.min(MAX_PUBLIC_ELAPSED_SECONDS, stage.clockBaseSeconds
            + Math.max(0, Math.floor((progressNow() - stage.clockAnchor) / 1000)));
    }

    const MODEL_TASK_LABELS = Object.freeze({ candidate_ranking: 'Gene prioritization', report_generation: 'Report preparation', report_repair: 'Report correction' });
    const MODEL_USAGE_COUNTS = ['output_tokens', 'estimated_output_tokens', 'reasoning_tokens', 'content_tokens', 'input_tokens', 'output_budget_tokens'];
    const MODEL_REVIEW_LABELS = Object.freeze({ pending: 'pending', passed: 'passed', corrected: 'corrected',
        needs_review: 'needs review', not_assessed: 'not assessed' });
    function publicModelUsage(value) {
        if (!value || typeof value !== 'object' || !Object.hasOwn(MODEL_TASK_LABELS, value.task)
                || typeof value.call_id !== 'string' || !/^[A-Za-z0-9_.:-]{1,100}$/.test(value.call_id)
                || !Number.isSafeInteger(value.attempt) || value.attempt < 1 || value.attempt > 24
                || !['running', 'checking', 'completed', 'failed'].includes(value.status)) return null;
        const clean = Object.fromEntries(['call_id', 'task', 'attempt', 'status'].map(key => [key, value[key]]));
        if (Object.hasOwn(MODEL_REVIEW_LABELS, value.review_status)) clean.review_status = value.review_status;
        for (const key of MODEL_USAGE_COUNTS) if (typeof value[key] === 'number'
                && boundedPublicInteger(value[key]) !== null) clean[key] = value[key];
        if (clean.output_tokens !== undefined) delete clean.estimated_output_tokens;
        if (clean.output_tokens !== undefined && (clean.reasoning_tokens > clean.output_tokens
                || clean.content_tokens > clean.output_tokens
                || (clean.reasoning_tokens ?? 0) + (clean.content_tokens ?? 0) > clean.output_tokens)) {
            delete clean.reasoning_tokens; delete clean.content_tokens;
        }
        if (Array.isArray(value.calls)) clean.calls = value.calls.slice(0, 8).map(item => item && typeof item === 'object'
            ? publicModelUsage(Object.fromEntries(Object.entries(item).filter(([key]) => key !== 'calls'))) : null).filter(Boolean);
        return clean;
    }
    function updateModelUsage(stage, value) {
        const usage = publicModelUsage(value);
        if (!stage || !usage) return;
        stage.modelUsage = usage;
        if (!stage.modelCalls) {
            const block = document.createElement('details'); block.className = 'model-call-details';
            block.appendChild(document.createElement('summary'));
            block.appendChild(document.createElement('div'));
            stage.body.appendChild(block); stage.modelCalls = block;
            stage.modelCallRows = new Map();
        }
        const calls = [...new Map((usage.calls?.length ? usage.calls : [usage]).map(item => [item.call_id, item])).values()];
        stage.modelCalls.children[0].textContent = `LLM calls · ${calls.length}`;
        const body = stage.modelCalls.children[1];
        const visible = new Set(calls.map(call => call.call_id));
        for (const [id, row] of stage.modelCallRows) if (!visible.has(id)) {
            row.root.remove(); stage.modelCallRows.delete(id);
        }
        const rankingCalls = calls.filter(call => call.task === 'candidate_ranking');
        for (const call of calls) {
            let row = stage.modelCallRows.get(call.call_id);
            if (!row) {
                const root = document.createElement('p'), title = document.createElement('strong');
                const generation = document.createElement('span'), review = document.createElement('span'), counts = document.createElement('span');
                root.className = 'model-call-row'; root.setAttribute('data-call-id', call.call_id);
                generation.className = 'model-call-generation'; review.className = 'model-call-review'; counts.className = 'model-call-counts';
                root.append(title, generation, review, counts); body.appendChild(root);
                row = { root, title, generation, review, counts }; stage.modelCallRows.set(call.call_id, row);
            }
            const { title, generation, review, counts } = row;
            title.textContent = `${MODEL_TASK_LABELS[call.task]} · attempt ${call.attempt}`;
            const parts = [];
            if (call.output_tokens !== undefined) parts.push(`${call.output_tokens.toLocaleString('en-US')} output tokens`);
            else if (call.estimated_output_tokens !== undefined) parts.push(`~${call.estimated_output_tokens.toLocaleString('en-US')} output tokens (estimate)`);
            if (call.reasoning_tokens !== undefined) parts.push(`${call.reasoning_tokens.toLocaleString('en-US')} reasoning tokens`);
            if (call.content_tokens !== undefined) parts.push(`${call.content_tokens.toLocaleString('en-US')} response tokens`);
            if (call.input_tokens !== undefined) parts.push(`${call.input_tokens.toLocaleString('en-US')} input tokens`);
            if (call.status === 'running' && call.output_budget_tokens !== undefined) parts.push(`output limit ${call.output_budget_tokens.toLocaleString('en-US')}`);
            const retained = call.status === 'failed' && call.review_status === undefined
                && call.task === 'candidate_ranking' && rankingCalls.length === 1
                && stage.progress.retainedRankingProof && call.content_tokens > 0 && call.output_tokens > 0;
            const generated = retained ? 'completed' : call.status;
            generation.textContent = ({ running: 'Generating', checking: 'Generation completed', completed: 'Generation completed', failed: 'Call failed' })[generated];
            const reviewStatus = call.review_status || (retained ? 'needs_review' : call.status === 'checking' ? 'pending' : 'not_assessed');
            review.textContent = `Fact review: ${MODEL_REVIEW_LABELS[reviewStatus]}`;
            row.root.setAttribute('data-generation-state', generated);
            row.root.setAttribute('data-review-state', reviewStatus);
            row.root.title = retained ? 'A complete LLM ranking is retained in this result; the older call status included fact-review issues.' : '';
            counts.textContent = parts.join(' · ');
        }
    }

    function updateRetainedRankingProof(progress, event) {
        const count = boundedPublicInteger(event?.computed_facts?.ranked_count);
        if (event?.analysis_status?.ranking !== 'model' || count === null || count < 1) return false;
        progress.retainedRankingProof = { count, method: 'model' };
        updateProgressScope(progress, event.computed_facts);
        const stage = progress.stages.get('response');
        stage.rankingUnavailable = false;
        for (const row of stage.details.children) if (row.textContent === LEGACY_RANKING_FAILURE_DETAIL
                || row.getAttribute('data-status-source') === LEGACY_RANKING_FAILURE_DETAIL) {
            row.textContent = geneticStatusLabel(RETAINED_RANKING_DETAIL);
        }
        if (stage.modelUsage) updateModelUsage(stage, stage.modelUsage);
        if (/(?:Model|LLM) ranking incomplete|ranking did not complete/i.test(stage.summary.textContent)) {
            stage.summary.textContent = `LLM ranking retained · ${count} genes · fact review recorded`;
        }
        return true;
    }

    function renderProgressClock(stage) {
        const seconds = elapsedStageSeconds(stage);
        const missing = stage.state !== 'pending';
        stage.duration.textContent = stage.separateRecordedRuns ? ' · separate recorded runs'
            : seconds === null ? (missing ? ' · time not recorded' : '')
            : seconds === 0 && (stage.progress.historical || stage.state !== 'active') ? ' · <1s' : ` · ${seconds.toLocaleString("en-US")}s`;
        stage.duration.setAttribute("aria-label", stage.separateRecordedRuns ? 'Durations belong to separate recorded runs'
            : seconds === null ? (missing ? 'Elapsed time was not recorded' : '') : `${seconds} seconds`);
        renderRetrievalActivity(stage);
        for (const row of stage.operationRows?.values() || []) renderOperationClock(stage, row);
        if (stage.id !== "response" || stage.state === "pending") return;
        const metrics = stage.responseMetrics || {};
        let text = "";
        if (Number.isSafeInteger(metrics.estimated_output_tokens)) {
            text = `~${metrics.estimated_output_tokens.toLocaleString("en-US")} ${stage.modelUsage ? 'total output ' : ''}tokens received (estimate)`;
        } else if (Number.isSafeInteger(metrics.output_tokens)) {
            text = `${metrics.output_tokens.toLocaleString("en-US")} ${stage.modelUsage ? 'total ' : ''}output tokens received`;
        } else if (Number.isSafeInteger(metrics.total_tokens)) {
            text = `${metrics.total_tokens.toLocaleString("en-US")} provider total tokens (input + output)`;
        } else if (["complete", "warning", "stopped", "error"].includes(stage.state)) {
            text = "Token count unavailable";
        }
        if (stage.modelUsage) text += `${text ? ' · ' : ''}${MODEL_TASK_LABELS[stage.modelUsage.task]} · attempt ${stage.modelUsage.attempt}`;
        stage.tokens.textContent = text;
        stage.tokens.hidden = !text;
    }

    function freezeProgressStage(stage) {
        if (!stage) return;
        if (stage.id === 'retrieve') finishProviderGroups(stage);
        if (!stage.clockFrozen) stage.durationSeconds = elapsedStageSeconds(stage);
        for (const row of stage.operationRows?.values() || []) {
            row.baseSeconds = operationSeconds(stage, row); row.anchor = progressNow();
            if (row.state === 'RUNNING') {
                const optional = optionalLdOperation(stage, row.lastValue);
                row.sourceState = 'PARTIAL'; row.state = optional ? 'OPTIONAL' : 'PARTIAL';
                row.executionState = 'FAILED'; row.executionIncomplete = true;
                row.root.setAttribute('data-state', 'FAILED');
                row.marker.textContent = '×'; row.stateLabel.textContent = 'Error';
                if (!stage.progress.historical && row.observedStartedMs !== undefined) row.observedEndedMs ??= Math.floor(progressNow());
            }
        }
        stage.clockFrozen = true;
        renderProgressClock(stage);
        renderRetrievalExecution(stage);
    }

    function stopProgressClock(progress) {
        if (progress.timer !== null) clearTimeout(progress.timer);
        progress.timer = null;
        liveProgressCards.delete(progress);
    }

    function disposeProgress(progress) {
        if (!progress) return;
        for (const stage of progress.stages.values()) freezeProgressStage(stage);
        stopProgressClock(progress);
        progress.complete = true;
    }

    function disposeChatProgress(destination) {
        for (const progress of liveProgressCards) {
            if (!destination || progress.messages === destination) disposeProgress(progress);
        }
    }

    function scheduleProgressClock(progress) {
        // Detached/historical cards have no live clocks. Browser lifecycle cleanup
        // is also checked every tick when a parent panel is removed externally.
        if (progress.complete || progress.historical || progress.root.isConnected !== true) return;
        progress.timer = setTimeout(() => {
            progress.timer = null;
            if (progress.root.isConnected === false || progress.complete) {
                disposeProgress(progress);
                return;
            }
            for (const stage of progress.stages.values()) renderProgressClock(stage);
            scheduleProgressClock(progress);
        }, 1000);
    }

    window.addEventListener?.("pagehide", () => disposeChatProgress());

    function setProgressStageExpanded(stage, expanded) {
        if (!stage) return;
        renderRetrievalExecution(stage);
        const hasContent = Boolean(stage.details.children.length || !stage.scope.hidden || !stage.phaseRail.hidden
            || !stage.diagnostics.hidden || stage.methodNotes?.children.length || stage.operationReview
            || stage.progress.normalizationNotice?.parentNode === stage.body);
        const value = Boolean(expanded && hasContent);
        stage.body.hidden = !value;
        stage.details.hidden = !value;
        stage.header.setAttribute("aria-expanded", String(value));
        stage.header.setAttribute('aria-label', `${value ? 'Hide' : 'Show'} ${stage.titleElement.textContent} details`);
        stage.toggleLabel.textContent = value ? 'Hide' : 'Details';
        stage.section.classList[value ? "add" : "remove"]("progress-stage-expanded");
    }

    function setProgressStageState(stage, state) {
        if (!stage) return;
        const allowed = new Set(["pending", "active", "complete", "warning", "unassessed", "stopped", "error"]);
        const next = stage.id === 'retrieve' && stage.retrievalFailed ? 'error'
            : stage.id === 'retrieve' && stage.retrievalBlocked ? 'stopped' : allowed.has(state) ? state : "active";
        if (next === 'active' && stage.progress.restoring && !stage.hasServerClock) stage.holdClockForServer = true;
        if (next === 'active' && stage.startedAt === null && !stage.progress.historical && !stage.holdClockForServer) {
            stage.startedAt = progressNow();
            if (stage.durationSeconds === null) stage.durationSeconds = 0;
            if (!stage.clockFrozen) stage.clockAnchor = stage.startedAt;
        }
        ["pending", "active", "complete", "warning", "unassessed", "stopped", "error"].forEach((name) => {
            stage.section.classList.remove(`progress-stage-${name}`);
        });
        stage.section.classList.add(`progress-stage-${next}`);
        stage.state = next;
        if (["complete", "warning", "unassessed", "stopped", "error"].includes(next)) {
            freezeProgressStage(stage);
        }
        renderProgressClock(stage);
        if (next === "active") {
            stage.marker.className = "progress-stage-marker progress-stage-spinner";
            stage.marker.textContent = "";
        } else if (next === "complete") {
            stage.marker.className = "progress-stage-marker progress-stage-check";
            stage.marker.textContent = "✓";
        } else if (next === "warning") {
            stage.marker.className = "progress-stage-marker progress-stage-warning-marker";
            stage.marker.textContent = "!";
        } else if (next === "unassessed") {
            stage.marker.className = "progress-stage-marker progress-stage-unassessed-marker";
            stage.marker.textContent = "–";
        } else if (next === "stopped" || next === "error") {
            stage.marker.className = "progress-stage-marker progress-stage-error-marker";
            stage.marker.textContent = "!";
        } else {
            stage.marker.className = "progress-stage-marker";
            stage.marker.textContent = String(stage.index + 1);
        }
        renderRetrievalExecution(stage);
    }

    function activateProgressStage(progress, stageId, state = "active") {
        if (!progress || !CHAT_PROGRESS_STAGE_IDS.has(stageId)) return null;
        const stage = progress.stages.get(stageId);
        if (!stage) return null;
        const priorActive = progress.stages.get(progress.activeStageId);
        // Late legacy events may arrive after the workflow has advanced. Keep
        // them in their historical detail panel without moving the active row
        // backwards or reopening a completed stage.
        if (priorActive && stage.index < priorActive.index) {
            if (state !== 'active' && state !== 'pending') setProgressStageState(stage, state);
            return stage;
        }
        if (!progress.historical && priorActive && priorActive.id !== stageId && priorActive.index < stage.index
                && priorActive.state === "active") {
            if (priorActive.id === 'retrieve') priorActive.executionIncomplete = true;
            setProgressStageState(priorActive, 'warning');
            priorActive.summary.textContent = 'Completion not recorded';
            setProgressStageExpanded(priorActive, false);
        }
        for (const candidate of progress.stages.values()) {
            if (!progress.historical && candidate.index < stage.index && candidate.state === "pending") {
                if (candidate.summary.textContent === "Waiting") {
                    candidate.summary.textContent = 'Stage not recorded';
                }
            }
        }
        if (!priorActive || stage.index >= priorActive.index || priorActive.state !== "active") {
            progress.activeStageId = stageId;
        }
        setProgressStageState(stage, state);
        setProgressStageExpanded(
            stage,
            state === "active" && progress.activeStageId === stageId
                && stage.userExpanded !== false
        );
        return stage;
    }

    function inferProgressStage(text) {
        const value = String(text || "").trim();
        if (/^(?:Safety gate|Safety review)/i.test(value)) return "safety";
        if (/^(?:Candidate ranking|Prioritizing candidate genes|Preparing the (?:safe |final )?|Generating |Formatting |Validating the response|Response ready)/i.test(value)) {
            return "response";
        }
        if (/^(?:Analyzing |Identifying |Intent identified:|Defining phenotype|Phenotype scope:|Biological context defined:|Planned evidence:|Planning the evidence)/i.test(value)) {
            return "plan";
        }
        return "retrieve";
    }

    function boundedPublicInteger(value, maximum = MAX_PUBLIC_PROGRESS_METRIC) {
        if (value === null || value === undefined || typeof value === 'boolean'
                || (typeof value === 'string' && !value.trim())) return null;
        const number = Number(value);
        return Number.isSafeInteger(number) && number >= 0 && number <= maximum
            ? number
            : null;
    }

    function publicRetrievalProgress(value) {
        if (!value || typeof value !== 'object' || Array.isArray(value)
                || !Object.hasOwn(RETRIEVAL_PHASE_LABELS, value.phase) || !RETRIEVAL_STATES.has(value.status)) return null;
        if (value.tool !== undefined && !Object.hasOwn(RETRIEVAL_TOOL_LABELS, value.tool)) return null;
        if (value.reason !== undefined && !Object.hasOwn(RETRIEVAL_REASON_LABELS, value.reason)) return null;
        const clean = { phase: value.phase, status: value.status };
        if (value.phase === 'v2g_retrieval') {
            if (!Object.hasOwn(V2G_RETRIEVAL_LABELS, value.name)) return null;
            clean.name = value.name;
            if (value.timing_basis === 'backend_monotonic') clean.timing_basis = value.timing_basis;
        }
        if (typeof value.operation_id === 'string' && /^[a-f0-9]{32}$/.test(value.operation_id)) clean.operation_id = value.operation_id;
        if (value.phase === 'semantic_tool' && value.status === 'completed'
                && value.tool === 'ot_get_target_profiles' && value.coverage_note === 'bounded_literature') {
            clean.coverage_note = 'bounded_literature';
        }
        const ldPolicy = publicLdReviewPolicy(value.ld_review_policy);
        if (ldPolicy) clean.ld_review_policy = ldPolicy;
        if (Object.hasOwn(RETRIEVAL_MODULE_NAMES, value.module)) clean.module = value.module;
        if (Array.isArray(value.modules)) clean.modules = [...new Set(value.modules
            .filter(module => Object.hasOwn(RETRIEVAL_MODULE_NAMES, module)))].slice(0, 40);
        if (value.module_statuses && typeof value.module_statuses === 'object' && !Array.isArray(value.module_statuses)) {
            clean.module_statuses = Object.fromEntries(Object.entries(value.module_statuses).filter(([module, state]) =>
                Object.hasOwn(RETRIEVAL_MODULE_NAMES, module) && Object.hasOwn(RETRIEVAL_OUTCOMES, state)).slice(0, 40));
        }
        if (value.module_counts && typeof value.module_counts === 'object' && !Array.isArray(value.module_counts)) {
            clean.module_counts = Object.fromEntries(Object.entries(value.module_counts).filter(([module, count]) =>
                Object.hasOwn(RETRIEVAL_MODULE_NAMES, module) && typeof count === 'number' && boundedPublicInteger(count) !== null).slice(0, 40));
        }
        if (value.reference_provider !== undefined) {
            if (!['ldlink', 'ensembl', 'opentargets'].includes(value.reference_provider)) return null;
            if (value.reference_provider === 'opentargets' && !OT_IDENTITY_PHASES.has(value.phase)) return null;
            clean.reference_provider = value.reference_provider;
        }
        if (value.provider_version !== undefined) {
            if (!clean.reference_provider || typeof value.provider_version !== 'string'
                    || !/^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$/.test(value.provider_version)) return null;
            clean.provider_version = value.provider_version;
        }
        if (value.phase === 'retrieval_phase') {
            if (!Object.hasOwn(RETRIEVAL_PHASE_NAMES, value.name) || !Object.hasOwn(RETRIEVAL_OUTCOMES, value.state)) return null;
            clean.name = value.name; clean.state = value.state;
            for (const key of ['scope_sha256', 'context_sha256']) {
                if (typeof value[key] === 'string' && /^[a-f0-9]{64}$/.test(value[key])) clean[key] = value[key];
            }
        }
        if (['source', 'proxy'].includes(value.identity_scope)) clean.identity_scope = value.identity_scope;
        if (['organizing', 'saving', 'verifying', 'loading', 'saving_raw', 'normalizing', 'checking_policy', 'projecting', 'saving_response'].includes(value.operation)) clean.operation = value.operation;
        if (Number.isSafeInteger(value.bytes_written) && value.bytes_written >= 0) clean.bytes_written = value.bytes_written;
        if (value.populations_requested !== undefined) {
            if (typeof value.populations_requested !== 'boolean') return null;
            clean.populations_requested = value.populations_requested;
        }
        if (['expression', 'colocalisation', 'transport', 'credible_sets', 'entities'].includes(value.channel)) clean.channel = value.channel;
        if (['genes', 'records', 'pages', 'entities', 'requests', 'variants', 'phenotypes', 'credible_sets'].includes(value.unit)) clean.unit = value.unit;
        if (value.tool) clean.tool = value.tool;
        if (value.reason) clean.reason = value.reason;
        for (const key of RETRIEVAL_COUNTS) {
            if (value[key] === undefined) continue;
            // The structured contract is typed JSON; strings/booleans are not counts.
            if (typeof value[key] !== 'number') return null;
            const count = boundedPublicInteger(value[key], key === 'elapsed_seconds' ? MAX_PUBLIC_ELAPSED_SECONDS : MAX_PUBLIC_PROGRESS_METRIC);
            if (count === null) return null;
            clean[key] = count;
        }
        if (clean.completed_count !== undefined && clean.total_count !== undefined && clean.completed_count > clean.total_count) return null;
        if (clean.batch_index !== undefined && clean.batch_count !== undefined && clean.batch_index > clean.batch_count) return null;
        return clean;
    }

    function retrievalStepKey(value) {
        if (value.phase === 'v2g_retrieval') return `v2g:${value.name}`;
        if (value.operation_id) return `call:${value.operation_id}:${value.phase}:${value.channel || ''}`;
        if (value.phase === 'ld_reference_request') return `${value.phase}:${value.reference_provider || 'unknown'}`;
        if (value.phase === 'retrieval_phase') return `${value.phase}:${value.name}`;
        if (value.phase === 'ld_identity_batch') return `${value.phase}:${value.reference_provider || 'unknown'}:${value.identity_scope || ''}:${value.populations_requested ?? ''}`;
        if (value.phase === 'result_processing') return `${value.phase}:${value.operation || ''}`;
        if (value.phase === 'retrieval_recovery' && ['transport', 'credible_sets', 'entities'].includes(value.channel)) {
            return `${value.phase}:${value.channel}:${value.tool || ''}`;
        }
        return [value.phase, value.channel || '', value.tool || '', value.batch_index ?? '', value.step ?? ''].join(':');
    }

    function showRetrievalWorkload(value) {
        // A successful MCP receipt reports returned records. Its pool offset
        // may come from a preceding wait and cannot prove completed gene work.
        return !(value.phase === 'semantic_tool' && value.tool && ['completed', 'cached'].includes(value.status));
    }

    function retrievalStepSummary(value, diagnostic = false, requestedTask = null) {
        if (value.phase === 'v2g_retrieval') return V2G_RETRIEVAL_LABELS[value.name];
        let label = value.phase === 'retrieval_recovery' ? ({ expression: 'Retrieving tissue and cell-type expression', colocalisation: 'Retrieving colocalization evidence',
            credible_sets: 'Retrieving fine-mapping credible sets', entities: 'Retrieving study evidence',
            transport: value.status === 'retrying' ? 'Retrying external evidence request' : 'Retrieving external evidence' }[value.channel] || RETRIEVAL_PHASE_LABELS[value.phase])
            : ['semantic_tool', 'semantic_variant_batch'].includes(value.phase) && value.tool
                ? RETRIEVAL_TOOL_LABELS[value.tool] : RETRIEVAL_PHASE_LABELS[value.phase];
        if (value.phase === 'ld_reference_request' && value.reference_provider === 'ldlink') label = 'Retrieving population-specific LD · LDlink';
        if (value.phase === 'ld_reference_request' && value.reference_provider === 'ensembl') label += ' · Ensembl';
        if (value.phase === 'ld_identity_batch' && typeof value.populations_requested === 'boolean') {
            label = value.populations_requested
                ? (value.identity_scope === 'proxy' ? 'Checking proxy variants in the LD reference panel'
                    : 'Checking selected variants in the LD reference panel')
                : (value.identity_scope === 'proxy' ? 'Validating LD proxy identifiers and REF/ALT' : 'Validating selected variant identifiers and REF/ALT');
        }
        if (value.reference_provider === 'opentargets') label = 'Validating variant identifiers · Open Targets';
        if (value.phase === 'result_processing') label = ({ organizing: 'Organizing locus and gene evidence',
            saving: 'Saving the analysis evidence', verifying: 'Checking retrieved data integrity',
            loading: 'Loading retrieved annotations', saving_raw: 'Saving source annotations',
            normalizing: 'Harmonizing annotation records', checking_policy: 'Checking evidence relevance to the analysis',
            projecting: 'Preparing evidence for interpretation', saving_response: 'Saving evidence for scientific review' })[value.operation] || label;
        if (diagnostic && value.bytes_written !== undefined) label += ` · ${(value.bytes_written / (1024 * 1024)).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB written`;
        const parts = [label];
        const count = number => number.toLocaleString('en-US');
        if (value.phase === 'retrieval_phase') {
            const parts = [taskPresentation(requestedTask).phases[value.name] || 'Evidence retrieval'];
            if (value.module) parts.push(RETRIEVAL_MODULE_NAMES[value.module]);
            if (value.source_count !== undefined) parts.push(`${count(value.source_count)} selected variants`);
            if (value.proxy_count !== undefined) parts.push(`${count(value.proxy_count)} validated LD proxies`);
            if (value.candidate_count !== undefined && ['genes', 'ranking'].includes(taskPresentation(requestedTask).scope)) {
                const geneLabel = requestedTask === 'gene_associations' ? 'genes assessed'
                    : value.name === 'discovery' ? 'candidate genes' : 'shortlisted genes';
                parts.push(`${count(value.candidate_count)} ${geneLabel}`);
            }
            if (value.records !== undefined) parts.push(`${count(value.records)} annotation records`);
            return parts.join(' · ');
        }
        if (value.phase === 'ld_recovery') {
            return parts.join(' · ');
        }
        if (diagnostic && value.batch_index !== undefined && value.batch_count !== undefined) parts.push(`batch ${count(value.batch_index)}/${count(value.batch_count)}`);
        const unit = ({ genes: 'genes', records: 'annotation records', entities: diagnostic ? 'entities' : 'study records',
            pages: diagnostic ? 'pages' : 'retrieval steps', requests: diagnostic ? 'requests' : 'retrieval steps' })[value.unit];
        if (showRetrievalWorkload(value)) {
            if (value.completed_count !== undefined && value.total_count !== undefined) parts.push(`${count(value.completed_count)}/${count(value.total_count)}${unit ? ' ' + unit : ''} checked`);
            else if (value.total_count !== undefined) parts.push(`${count(value.total_count)} to check`);
        }
        if (value.records !== undefined && value.tool !== 'ot_get_capabilities'
                && !(value.phase === 'result_processing' && value.completed_count !== undefined && value.total_count !== undefined)) {
            parts.push(`${count(value.records)} annotation records`);
        }
        if (value.variant_count !== undefined && value.phase === 'semantic_variant_batch') parts.push(`${count(value.variant_count)} variants`);
        if (diagnostic) for (const [key, text] of [['network_requests', 'network requests'], ['cache_hits', 'cache hits'],
            ['failed_cache_hits', 'retained failed requests'], ['completed_pages', 'completed pages'], ['failed_pages', 'failed requests'],
            ['failed_request_pages', 'failed requests'],
            ['unique_request_pages', 'distinct request pages'], ['unique_credible_sets', 'distinct credible sets'],
            ['unique_studies', 'distinct studies'], ['unique_connections', 'distinct connections'],
            ['returned_count', 'reference metadata records returned']]) {
            if (key === 'failed_pages' && value.failed_request_pages !== undefined) continue;
            if (value[key] !== undefined) parts.push(`${count(value[key])} ${text}`);
        }
        return parts.join(' · ');
    }

    function renderRetrievalActivity(stage) {
        // Waiting belongs to the persistent operation row. A separate transient
        // line used to shift the entire body whenever requests interleaved.
        if (!stage?.activity) return;
        stage.activity.textContent = ''; stage.activity.hidden = true;
    }

    const OPERATION_STATES = Object.freeze({ PENDING: 'Waiting', RUNNING: 'In progress', SUCCESS_DATA: 'Completed',
        SUCCESS_NO_DATA: 'Completed · no matching records', PARTIAL: 'Partial coverage', FAILED: 'Failed · coverage unknown',
        BLOCKED: 'Blocked · not performed', OPTIONAL: 'Optional retrieval incomplete', NOT_RUN: 'Not performed', NOT_REQUESTED: 'Not requested', NOT_APPLICABLE: 'Not applicable', CACHED: 'Cached result reused' });
    const EXECUTION_ERROR_REASONS = new Set(['connection_error', 'request_timeout', 'service_error',
        'rate_limited', 'invalid_response', 'invalid_or_unavailable_response', 'external_acquisition_unavailable',
        'ld_credential_unavailable', 'source_identity_unavailable', 'reference_unavailable']);
    function operationExecutionState(value, state, unfinished = false) {
        if (unfinished) return 'FAILED';
        if (state === 'RUNNING' || state === 'PENDING') return 'RUNNING';
        const ot = value.phase === 'v2g_retrieval' || value.phase === 'semantic_tool'
            || value.phase === 'semantic_variant_batch';
        const legacyPartialData = value.phase === 'v2g_retrieval' && value.status === 'incomplete'
            && Number.isInteger(value.failed_pages) && value.completed_count > value.failed_pages;
        // Completion describes obtaining a response. Scientific coverage is
        // retained separately, including failed continuations and row limits.
        if (ot && (value.records > 0 || value.successful_requests > 0 || legacyPartialData)) return 'SUCCESS_DATA';
        const capabilities = value.tool === 'ot_get_capabilities'
            || ['schema.discovery', 'runtime.capabilities'].includes(value.module);
        // A returned capabilities catalogue describes supported modules; its
        // unsupported entries are not failures of the catalogue request.
        if (!capabilities && Object.values(value.module_statuses || {}).includes('FAILED')) return 'FAILED';
        if (['SUCCESS_DATA', 'SUCCESS_NO_DATA', 'CACHED'].includes(state)) return 'SUCCESS_DATA';
        if (state === 'FAILED' || EXECUTION_ERROR_REASONS.has(value.reason)
                || value.failed_request_pages > 0 || value.failed_pages > 0 || value.failed_items > 0) return 'FAILED';
        if (['NOT_RUN', 'NOT_REQUESTED', 'NOT_APPLICABLE', 'BLOCKED'].includes(state)) return null;
        return 'SUCCESS_DATA';
    }
    function renderRetrievalExecution(stage) {
        if (!stage || stage.id !== 'retrieve' || stage.state === 'pending') return;
        const error = ['error', 'stopped'].includes(stage.state) || stage.retrievalFailed || stage.retrievalBlocked
            || stage.executionFailureDeclared || stage.executionIncomplete
            || [...(stage.operationRows?.values() || [])].some(row => row.executionState === 'FAILED');
        const state = stage.state === 'active' ? 'active' : error ? 'error' : 'complete';
        stage.executionState = state;
        for (const name of ['pending', 'active', 'complete', 'warning', 'unassessed', 'stopped', 'error']) {
            stage.section.classList.remove(`progress-stage-${name}`);
        }
        stage.section.classList.add(`progress-stage-${state}`);
        stage.marker.className = 'progress-stage-marker ' + (state === 'active' ? 'progress-stage-spinner'
            : state === 'error' ? 'progress-stage-error-marker' : 'progress-stage-check');
        stage.marker.textContent = state === 'active' ? '' : state === 'error' ? '×' : '✓';
        const count = stage.retrievalRecordCount;
        stage.summary.textContent = (state === 'active' ? 'In progress' : state === 'error' ? 'Error' : 'Completed')
            + (Number.isSafeInteger(count) ? ` · ${count.toLocaleString('en-US')} records` : '');
    }
    const METADATA_OPERATION_KEYS = new Set(['tool:ot_get_capabilities', 'module:schema.discovery', 'module:runtime.capabilities']);
    function publicLdReviewPolicy(value) {
        if (!value || typeof value !== 'object' || Array.isArray(value)
                || value.proxy_expansion !== 'optional' || typeof value.selected_source_identities_validated !== 'boolean') return null;
        return { proxy_expansion: 'optional', selected_source_identities_validated: value.selected_source_identities_validated };
    }
    function optionalLdOperation(stage, value) {
        if (stage?.progress?.ldReviewPolicy?.selected_source_identities_validated !== true) return false;
        if (!value?.phase?.startsWith('ld_') || ['ld_source_identity', 'ld_source_discovery'].includes(value.phase)) return false;
        return value.identity_scope !== 'source';
    }
    function applyLdReviewPolicy(progress, policy) {
        const clean = publicLdReviewPolicy(policy);
        if (!clean) return;
        progress.ldReviewPolicy = clean;
        const stage = progress.stages.get('retrieve');
        for (const [key, row] of stage?.operationRows || []) {
            if (['PARTIAL', 'FAILED', 'BLOCKED', 'NOT_RUN', 'OPTIONAL'].includes(row.state) && row.lastValue) {
                updateOperationRow(stage, key, row.title.textContent, row.lastValue, row.sourceState || row.state,
                    { ignoreElapsed: true, noClock: true });
            }
        }
        if (stage?.state === 'warning' && stage.optionalLdOnlyWarning
                && ![...stage.operationRows.values()].some(row => ['PARTIAL', 'FAILED', 'BLOCKED'].includes(row.state))) {
            setProgressStageState(stage, 'complete');
            renderRetrievalExecution(stage);
            stage.optionalLdOnlyWarning = false;
        }
    }
    function operationSeconds(stage, row) {
        if (row.baseSeconds === null) return null;
        const ticking = row.state === 'RUNNING' && !stage.clockFrozen && !stage.progress.historical;
        return Math.min(MAX_PUBLIC_ELAPSED_SECONDS, row.baseSeconds + (ticking ? Math.max(0, (progressNow() - row.anchor) / 1000) : 0));
    }
    function renderOperationClock(stage, row) {
        const seconds = operationSeconds(stage, row);
        row.timing.textContent = seconds === null ? '' : seconds < 1 ? '<1s' : `${Math.floor(seconds).toLocaleString('en-US')}s`;
        row.timing.setAttribute('aria-label', seconds === null ? 'Duration not recorded' : seconds < 1 ? 'Less than 1 second' : `${Math.floor(seconds)} seconds`);
    }
    function publicOperationTimings(value) {
        if (!Array.isArray(value) || value.length > 48) return [];
        const rows = [], seen = new Set();
        for (const item of value) {
            if (!item || item.basis !== 'observed_operation' || typeof item.operation_key !== 'string'
                    || !/^(?:ld|tool|module|phase|processing|recovery|catalog):[A-Za-z0-9_.:-]{1,100}$/.test(item.operation_key)
                    || seen.has(item.operation_key) || typeof item.elapsed_seconds !== 'number'
                    || boundedPublicInteger(item.elapsed_seconds, MAX_PUBLIC_ELAPSED_SECONDS) === null) continue;
            const row = { operation_key: item.operation_key, basis: item.basis, elapsed_seconds: item.elapsed_seconds };
            for (const key of ['started_ms', 'ended_ms']) {
                if (typeof item[key] === 'number' && boundedPublicInteger(item[key], Number.MAX_SAFE_INTEGER) !== null) row[key] = item[key];
            }
            if (row.ended_ms !== undefined && (row.started_ms === undefined || row.ended_ms < row.started_ms)) continue;
            seen.add(row.operation_key); rows.push(row);
        }
        return rows;
    }
    function operationTimings(stage) {
        const rows = [];
        for (const [key, row] of stage?.operationRows || []) {
            const seconds = operationSeconds(stage, row);
            if (seconds === null || row.observedStartedMs === undefined) continue;
            rows.push({ operation_key: key, basis: 'observed_operation', elapsed_seconds: Math.floor(seconds),
                started_ms: row.observedStartedMs,
                ...(row.observedEndedMs !== undefined ? { ended_ms: row.observedEndedMs } : {}) });
        }
        return publicOperationTimings(rows);
    }
    function updateOperationRow(stage, key, title, value, requestedState, options = {}) {
        let row = stage.operationRows.get(key);
        if (!row) {
            if (stage.operationRows.size >= 48) return null;
            const root = document.createElement('div'), marker = document.createElement('span');
            const titleNode = document.createElement('span'), stateLabel = document.createElement('span');
            const timing = document.createElement('span'), detail = document.createElement('span');
            root.className = 'progress-operation'; root.setAttribute('data-operation-key', key);
            marker.className = 'progress-operation-marker'; marker.setAttribute('aria-hidden', 'true');
            titleNode.className = 'progress-operation-title'; stateLabel.className = 'progress-operation-state';
            timing.className = 'progress-operation-time'; detail.className = 'progress-operation-detail';
            root.append(marker, titleNode, stateLabel, timing, detail);
            if (!stage.operationList.parentNode) stage.details.appendChild(stage.operationList);
            stage.operationList.appendChild(root);
            row = { root, marker, title: titleNode, stateLabel, timing, detail, state: 'PENDING', baseSeconds: null, anchor: progressNow(), lastValue: null };
            stage.operationRows.set(key, row);
        }
        const metadata = METADATA_OPERATION_KEYS.has(key) || key.startsWith('tool:ot_get_capabilities:');
        const unfinished = requestedState === 'RUNNING' && stage.clockFrozen && !['active', 'pending'].includes(stage.state);
        // Capability/schema discovery returns metadata, not scientific evidence
        // rows. A successful probe must not imply that an evidence query was empty.
        const displayState = metadata && requestedState === 'SUCCESS_NO_DATA' ? 'SUCCESS_DATA' : requestedState;
        const sourceState = unfinished ? 'PARTIAL' : Object.hasOwn(OPERATION_STATES, displayState) ? displayState : 'RUNNING';
        // Optional proxy admission has no bearing on the identity of selected
        // variants. Keep its original receipt and counts without an alarm state.
        const optional = optionalLdOperation(stage, value) && ['PARTIAL', 'FAILED', 'BLOCKED', 'NOT_RUN'].includes(sourceState);
        const state = optional ? 'OPTIONAL' : sourceState;
        const terminal = !['PENDING', 'RUNNING'].includes(row.state);
        if (terminal && requestedState === 'RUNNING' && !options.newInvocation) return row; // Delayed heartbeats cannot reopen completed work.
        const now = progressNow(), elapsed = operationSeconds(stage, row);
        const recorded = stage.recordedOperationTimings?.get(key);
        // Old LD receipts stored the last network request's elapsed seconds.
        // They cannot establish the duration of the whole LD operation.
        const unmeasuredHistory = stage.progress.historical && key.startsWith('ld:') && !recorded;
        const incoming = options.ignoreElapsed || unmeasuredHistory ? null : boundedPublicInteger(value.elapsed_seconds, MAX_PUBLIC_ELAPSED_SECONDS);
        if (!options.noClock && (incoming !== null || elapsed !== null || (state === 'RUNNING' && !stage.progress.historical))) {
            row.baseSeconds = value.timing_basis === 'backend_monotonic' && incoming !== null
                ? incoming : Math.max(elapsed || 0, incoming || 0); row.anchor = now;
        }
        if (!stage.progress.historical && state === 'RUNNING' && row.observedStartedMs === undefined) row.observedStartedMs = Math.floor(now);
        if (!stage.progress.historical && row.observedStartedMs !== undefined) {
            if (state === 'RUNNING') delete row.observedEndedMs;
            else row.observedEndedMs ??= Math.floor(now);
        }
        if (stage.progress.historical && recorded && value.timing_basis !== 'backend_monotonic') {
            row.baseSeconds = recorded.elapsed_seconds; row.anchor = now;
        }
        row.state = state;
        row.sourceState = sourceState;
        if (!options.transport) row.lastValue = value;
        row.title.textContent = title;
        row.title.title = title;
        if (['completed', 'cached', 'incomplete', 'failed', 'unavailable', 'blocked', 'not_run'].includes(value.status)) row.executionIncomplete = false;
        row.executionFailed = options.executionFailed ?? row.executionFailed ?? false;
        const execution = row.executionFailed && sourceState !== 'RUNNING' ? 'FAILED'
            : operationExecutionState(value, sourceState, unfinished || row.executionIncomplete);
        row.executionState = execution;
        row.root.hidden = execution === null;
        row.root.setAttribute('data-state', execution || '');
        row.marker.textContent = execution === 'SUCCESS_DATA' ? '✓' : execution === 'FAILED' ? '×' : '';
        row.stateLabel.textContent = ({ RUNNING: 'In progress', SUCCESS_DATA: 'Completed', FAILED: 'Error' })[execution] || '';
        const parts = [];
        if (state === 'RUNNING' && value.status === 'waiting') parts.push('Waiting for the data service');
        else if (state === 'RUNNING' && value.status === 'retrying') parts.push('Retrying the request');
        else if (state === 'RUNNING' && value.status === 'splitting') parts.push('Splitting the request');
        if (execution === 'FAILED' && EXECUTION_ERROR_REASONS.has(value.reason)) parts.push(RETRIEVAL_REASON_LABELS[value.reason]);
        if (showRetrievalWorkload(value) && value.completed_count !== undefined && value.total_count !== undefined) {
            const unit = value.channel === 'credible_sets' ? 'credible sets' : value.channel === 'entities' ? 'study records'
                : value.unit === 'credible_sets' ? 'credible sets'
                : ['genes', 'records', 'pages', 'requests', 'variants', 'phenotypes'].includes(value.unit) ? value.unit : '';
            parts.push(`${value.completed_count}/${value.total_count}${unit ? ' ' + unit : ''}`);
        }
        if (value.records !== undefined && !metadata) parts.push(`${value.records.toLocaleString('en-US')} records`);
        if (value.source_count !== undefined) parts.push(`${value.source_count} selected variants`);
        if (value.proxy_count !== undefined && !stage.progress.ldUnavailable) parts.push(`${value.proxy_count} validated LD proxies`);
        if (value.attempt !== undefined && value.status === 'retrying') parts.push(`attempt ${value.attempt}`);
        row.detail.textContent = parts.join(' · ');
        row.detail.title = row.detail.textContent;
        renderOperationClock(stage, row);
        renderRetrievalExecution(stage);
        return row;
    }
    function updateOperationProgress(stage, value) {
        if (value.phase === 'ld_source_identity' && value.status === 'started') stage.ldSessionOpen = true;
        if (value.phase === 'ld_expansion' && !['started', 'running', 'waiting', 'retrying'].includes(value.status)) {
            stage.ldSessionOpen = false;
            finishProviderGroups(stage);
        }
        let state = value.state || ({ completed: value.records === 0 ? 'SUCCESS_NO_DATA' : 'SUCCESS_DATA',
            incomplete: 'PARTIAL', failed: 'FAILED', unavailable: 'FAILED', blocked: 'BLOCKED', not_run: 'NOT_RUN', cached: 'CACHED' })[value.status] || 'RUNNING';
        // A completed zero-sized LD workload is not a successful query. Older
        // receipts used "completed" even when source identity prevented lookup.
        const identity = ['ld_source_identity', 'ld_proxy_identity', 'ld_population_identity', 'ld_identity_batch'].includes(value.phase);
        if (value.status === 'completed' && !value.state) {
            if ((identity || value.phase === 'ld_proxy_lookup') && value.total_count === 0) state = 'NOT_RUN';
            else if (identity && value.total_count > 0 && value.validated_count !== undefined
                    && value.validated_count < value.total_count) state = 'PARTIAL';
        }
        if (state === 'NOT_RUN' && ['ld_proxy_lookup', 'ld_proxy_identity', 'ld_population_identity'].includes(value.phase)) {
            const source = stage.operationRows.get('ld:ld_source_identity');
            const lookup = stage.operationRows.get('ld:ld_proxy_lookup');
            if (['source_identity_unresolved', 'source_identity_unavailable'].includes(value.reason)
                    || ['FAILED', 'BLOCKED'].includes(source?.state)
                    || (source?.lastValue?.total_count > 0 && source.lastValue.validated_count === 0)
                    || (value.phase !== 'ld_proxy_lookup' && ['FAILED', 'BLOCKED'].includes(lookup?.state))) state = 'BLOCKED';
        }
        // Modules are coverage inside an actual MCP call, not independent
        // invocations. Never create planned Waiting cards for them.
        if (value.phase === 'retrieval_phase') return;
        if (!value.tool && value.module) {
            updateOperationRow(stage, `module:${value.module}`, RETRIEVAL_MODULE_NAMES[value.module], value, state, { noClock: true });
            return;
        }
        const transport = value.phase === 'retrieval_recovery' && value.channel === 'transport';
        if (transport || value.phase === 'retrieval_recovery') {
            const operationKey = value.operation_id && value.tool ? `tool:${value.tool}:${value.operation_id}` : stage.currentOperationKey;
            const active = stage.operationRows.get(operationKey);
            if (active?.state === 'RUNNING' && ['waiting', 'retrying', 'splitting'].includes(value.status)) {
                updateOperationRow(stage, operationKey, active.title.textContent,
                    { ...active.lastValue, status: value.status, reason: value.reason, attempt: value.attempt }, 'RUNNING', { ignoreElapsed: true, transport: true });
            } else if (active?.state === 'RUNNING' && value.status === 'completed' && active.lastValue) {
                updateOperationRow(stage, operationKey, active.title.textContent,
                    active.lastValue, 'RUNNING', { ignoreElapsed: true, transport: true });
            }
            return;
        }
        let key, title, nextState = state;
        if (value.phase === 'v2g_retrieval') {
            key = `phase:v2g_${value.name}`; title = V2G_RETRIEVAL_LABELS[value.name];
        } else if (value.tool) {
            key = `tool:${value.tool}${value.operation_id ? ':' + value.operation_id : ''}`;
            title = RETRIEVAL_TOOL_LABELS[value.tool].replace(/^Retrieving /, '');
            title = title.charAt(0).toUpperCase() + title.slice(1);
            if (value.operation_id) {
                const previous = stage.operationRows.get(key);
                const ordinal = previous?.callOrdinal || 1 + [...stage.operationRows.keys()]
                    .filter(item => item.startsWith(`tool:${value.tool}:`)).length;
                title += ` · call ${ordinal}`;
            }
            if (value.phase === 'semantic_variant_batch') nextState = 'RUNNING'; // A page or batch is not the whole tool.
        } else if (value.phase === 'retrieval_phase') {
            key = `phase:${value.name}`; title = taskPresentation(stage.progress.requestedTask).phases[value.name];
        } else if (value.phase.startsWith('ld_')) {
            // Reference calls can interleave across providers. Keep their
            // identities and outcomes separate, and never infer Ensembl from
            // a generic LD phase when no provider was recorded.
            const providerOperation = ['ld_reference_request', 'ld_identity_batch'].includes(value.phase);
            key = `ld:${value.phase}${providerOperation && value.reference_provider ? ':' + value.reference_provider : ''}`;
            title = retrievalStepSummary({ phase: value.phase, status: value.status,
                reference_provider: value.reference_provider, identity_scope: value.identity_scope,
                populations_requested: value.populations_requested });
        } else if (value.phase === 'catalog_request') {
            key = 'catalog:associations'; title = 'Published GWAS associations';
        } else if (value.phase === 'retrieval_recovery' && value.channel) {
            key = `recovery:${value.channel}`; title = retrievalStepSummary({ phase: value.phase, status: value.status, channel: value.channel });
        } else if (value.phase === 'result_processing') {
            key = `processing:${value.operation || 'organizing'}`;
            title = retrievalStepSummary({ phase: value.phase, status: value.status, operation: value.operation });
        }
        if (!key || !title) return;
        if (stage.progress.ldUnavailable && value.phase === 'ld_expansion') {
            nextState = 'NOT_APPLICABLE'; title = 'Retrieving population-specific LD · LDlink';
        }
        if (['ld_reference_request', 'ld_identity_batch'].includes(value.phase) && value.reference_provider) {
            updateProviderGroup(stage, key, title, value, nextState);
            return;
        }
        const row = updateOperationRow(stage, key, title, value, nextState, { ignoreElapsed: value.phase === 'semantic_variant_batch',
            newInvocation: !value.operation_id && ['semantic_tool', 'ld_reference_request', 'ld_identity_batch'].includes(value.phase) && value.status === 'started' });
        if (row && value.operation_id) row.callOrdinal ??= Number(title.match(/call (\d+)$/)?.[1]) || 1;
        if (row && value.module_statuses) {
            const coverage = Object.keys(value.module_statuses).map(module =>
                !['schema.discovery', 'runtime.capabilities'].includes(module) && value.module_counts?.[module] !== undefined
                    ? `${RETRIEVAL_MODULE_NAMES[module]}: ${value.module_counts[module]} records` : '').filter(Boolean);
            row.detail.textContent = [row.detail.textContent, ...coverage].filter(Boolean).join(' · ');
            row.detail.title = row.detail.textContent;
        }
        if (row?.state === 'RUNNING') stage.currentOperationKey = key;
    }

    function updateProviderGroup(stage, key, title, value, state) {
        stage.providerGroups ??= new Map();
        let group = stage.providerGroups.get(key);
        if (!group) {
            group = { key, title, calls: new Map(), sequence: 0, current: null };
            stage.providerGroups.set(key, group);
        }
        let identity = value.operation_id;
        if (!identity) {
            if (value.status === 'started') group.current = `observed-${++group.sequence}`;
            identity = group.current || 'legacy-receipt';
        }
        const previous = group.calls.get(identity);
        // Retries remain one actual request; a delayed callback cannot change
        // that request's terminal outcome. Other requests remain independent.
        if (previous && previous.state !== 'RUNNING' && state === 'RUNNING') return;
        group.calls.set(identity, { state, value });
        group.value = value;
        renderProviderGroup(stage, group);
    }

    function renderProviderGroup(stage, group) {
        const calls = [...group.calls.values()];
        const active = calls.filter(call => call.state === 'RUNNING').length;
        const failed = calls.filter(call => ['FAILED', 'PARTIAL', 'BLOCKED', 'NOT_RUN'].includes(call.state)).length;
        const finished = calls.length - active;
        const running = active > 0 || stage.ldSessionOpen === true;
        const state = running ? 'RUNNING' : calls.every(call => call.state === 'FAILED') ? 'FAILED'
            : failed ? 'PARTIAL' : calls.every(call => call.state === 'CACHED') ? 'CACHED' : 'SUCCESS_DATA';
        const executionFailures = calls.filter(call => operationExecutionState(call.value, call.state) === 'FAILED');
        const row = updateOperationRow(stage, group.key, group.title, group.value, state,
            { ignoreElapsed: true, newInvocation: true, executionFailed: executionFailures.length > 0 });
        if (!row) return;
        const legacy = calls.length === 1 && group.calls.has('legacy-receipt');
        const parts = [legacy ? ''
            : `${finished}/${calls.length} requests finished`,
            active ? `${active} active` : running ? 'Other LD checks in progress' : '',
            ...[...new Set(executionFailures
                .map(call => RETRIEVAL_REASON_LABELS[call.value.reason]).filter(Boolean))],
            active && group.value.status === 'retrying' ? `Retrying request · attempt ${group.value.attempt || 1}` : ''];
        row.detail.textContent = parts.filter(Boolean).join(' · ');
        row.detail.title = row.detail.textContent;
    }

    function finishProviderGroups(stage) {
        stage.ldSessionOpen = false;
        for (const group of stage.providerGroups?.values() || []) renderProviderGroup(stage, group);
    }

    function updateRetrievalStep(progress, value, state) {
        const stage = progress.stages.get('retrieve');
        if (!stage) return false;
        if (value.ld_review_policy) applyLdReviewPolicy(progress, value.ld_review_policy);
        // Live source validation is sufficient proof for the optional scope.
        // Old saved histories require an explicit, re-evaluated review policy.
        if (!progress.historical && !progress.ldReviewPolicy && value.phase === 'ld_source_identity'
                && value.status === 'completed' && value.total_count > 0 && value.validated_count === value.total_count) {
            applyLdReviewPolicy(progress, { proxy_expansion: 'optional', selected_source_identities_validated: true });
        }
        if (value.phase === 'ld_expansion' && value.reason === 'ld_credential_unavailable') {
            progress.ldUnavailable = true;
            if (progress.scopeCounts) delete progress.scopeCounts.verified_proxy_count;
            updateProgressScope(progress, {});
        }
        if (value.phase === 'retrieval_phase' && !taskPresentation(progress.requestedTask).phases[value.name]) return true;
        activateProgressStage(progress, 'retrieve', state);
        updateOperationProgress(stage, value);
        if (value.phase === 'retrieval_phase') {
            stage.currentSubstage = value.name;
            stage.section.setAttribute('data-current-substage', value.name);
        }
        const key = retrievalStepKey(value);
        let row = stage.retrievalSteps.get(key);
        if (!row && progress.detailCount < MAX_CHAT_PROGRESS_LINES) {
            row = document.createElement('div');
            row.className = 'progress-stage-detail progress-retrieval-step';
            stage.diagnosticBody.appendChild(row);
            stage.diagnostics.hidden = false;
            stage.retrievalSteps.set(key, row);
            progress.detailCount += 1;
        }
        const candidateKey = value.name === 'discovery' ? 'candidate_pool_count' : 'shortlist_count';
        let displayed = value.phase === 'retrieval_phase' && progress.explicitScopeCounts?.has(candidateKey)
            ? { ...value, candidate_count: progress.scopeCounts[candidateKey] } : value;
        if (progress.ldUnavailable && value.phase === 'retrieval_phase') {
            displayed = { ...displayed }; delete displayed.proxy_count;
        }
        const summary = retrievalStepSummary(displayed, false, progress.requestedTask);
        if (row && value.phase === 'retrieval_phase') {
            const text = document.createElement('span'); text.textContent = summary;
            const badge = document.createElement('span'); badge.className = 'retrieval-state-badge';
            const execution = operationExecutionState(value, value.state);
            badge.setAttribute('data-state', execution || '');
            badge.textContent = ({ RUNNING: 'In progress', SUCCESS_DATA: 'Completed', FAILED: 'Error' })[execution] || '';
            badge.hidden = !execution || execution === 'RUNNING';
            row.replaceChildren(text, badge); row.setAttribute('data-retrieval-state', value.state);
            stage.phaseStates.set(value.name, value.state);
            renderPhaseRail(stage, retrievalPhaseLabels(progress));
            updateProgressScope(progress, { source_variant_count: value.source_count, verified_proxy_count: value.proxy_count,
                [value.name === 'discovery' ? 'candidate_pool_count' : 'shortlist_count']: value.candidate_count }, 'phase');
        } else if (row) {
            const rawState = ({ completed: 'SUCCESS_DATA', cached: 'CACHED', incomplete: 'PARTIAL',
                failed: 'FAILED', unavailable: 'FAILED', not_run: 'NOT_RUN', blocked: 'BLOCKED' })[value.status] || 'RUNNING';
            const execution = operationExecutionState(value, rawState);
            row.hidden = execution === null;
            row.textContent = retrievalStepSummary(value, row.parentNode === stage.diagnosticBody, progress.requestedTask)
                + (execution === 'FAILED' && EXECUTION_ERROR_REASONS.has(value.reason) ? ` · ${RETRIEVAL_REASON_LABELS[value.reason]}` : '')
                + (execution ? ' · ' + ({ RUNNING: 'In progress', SUCCESS_DATA: 'Completed', FAILED: 'Error' })[execution] : '');
        }
        if (value.tool === 'ot_get_capabilities'
                || (value.status === 'cached' && value.phase !== 'ld_identity_batch')) return true;
        const transport = value.phase === 'retrieval_recovery' && value.channel === 'transport';
        // Individual network completions are not completion of a scientific
        // workload (for example, a variant batch still awaiting other pages).
        // Keep their counters above, and let their own waits/errors remain visible.
        if (transport && ['started', 'running', 'completed'].includes(value.status) && stage.scientificRetrieval) {
            stage.currentRetrieval = stage.scientificRetrieval;
        } else {
            stage.currentRetrieval = { value, receivedAt: progressNow(), summary };
            if (!transport) stage.scientificRetrieval = stage.currentRetrieval;
        }
        stage.summary.textContent = substageSummary(stage, stage.currentRetrieval.summary, state);
        renderRetrievalActivity(stage);
        setProgressStageExpanded(stage, stage.userExpanded === true);
        return true;
    }

    function renderPhaseRail(stage, labels) {
        if (!stage?.phaseRail) return;
        stage.phaseRail.replaceChildren();
        for (const [id, label] of Object.entries(labels)) {
            const item = document.createElement('span'); item.className = 'progress-phase-chip';
            const state = stage.phaseStates.get(id) || 'PENDING';
            item.setAttribute('data-state', state);
            item.textContent = label;
            const outcome = stage.id === 'response'
                ? ({ SUCCESS_DATA: 'Completed', PARTIAL: 'Incomplete', FAILED: 'Failed', NOT_APPLICABLE: 'Not assessed' })[state]
                : RETRIEVAL_OUTCOMES[state];
            item.setAttribute('aria-label', `${label}: ${outcome || (state === 'RUNNING' ? 'In progress' : 'Not yet recorded')}`);
            stage.phaseRail.appendChild(item);
        }
        // Substages are announced in the live status line. Retain their bounded
        // state records for History without displaying a second navigation rail.
        stage.phaseRail.hidden = true;
    }

    function retrievalPhaseLabels(progress) {
        const observed = progress.stages.get('retrieve')?.phaseStates;
        return Object.fromEntries(Object.entries(taskPresentation(progress.requestedTask).phases).filter(([id]) => observed?.has(id)));
    }

    function interpretationPhaseLabels(stage) {
        return { inference: 'Scientific interpretation', fact_verification: 'Objective fact verification' };
    }

    function substageSummary(stage, summary, state = stage.state) {
        if (state !== 'active') return summary;
        if (stage.id === 'retrieve') {
            if (stage.phaseStates.get(stage.currentSubstage) !== 'RUNNING') return summary;
            const label = taskPresentation(stage.progress.requestedTask).phases[stage.currentSubstage];
            if (label && stage.currentRetrieval?.value.phase === 'retrieval_phase') return label;
            return !label || summary.startsWith(label) ? summary : `${label} · ${summary}`;
        }
        if (stage.id !== 'response' || stage.progress.requestedTask === 'general' || stage.progress.scientificBlocking) return summary;
        const label = interpretationPhaseLabels(stage)[stage.currentSubstage || 'inference'];
        if (!label || summary.startsWith(label)) return summary;
        // A generic stream/heartbeat message must not replace a known check.
        if (stage.currentSubstage === 'fact_verification') return label;
        const generation = /^(?:Generating response|Preparing response|Validating response)(?: · (.+))?$/.exec(summary);
        if (generation) return generation[1] ? `${label} · ${generation[1]}` : label;
        return `${label} · ${summary}`;
    }

    function updateRequestedTask(progress, event) {
        if (SCIENTIFIC_TASKS.has(event.requested_task)) progress.requestedTask = event.requested_task;
        const labels = { 'general conversation': 'general', 'variant description': 'describe_variants',
            'variant-to-gene associations': 'gene_associations', 'gene prioritization': 'prioritize_genes',
            'target assessment': 'target_assessment', 'disease associations': 'disease_associations', 'drug repurposing': 'drug_repurposing' };
        if (event.stage === 'plan') for (const [label, task] of Object.entries(labels)) {
            if (String(event.summary || '').split(' · ').includes(label)) progress.requestedTask = task;
        }
        for (const stage of progress.stages.values()) stage.titleElement.textContent = stageTitle(stage.id, progress.requestedTask);
        renderPhaseRail(progress.stages.get('retrieve'), retrievalPhaseLabels(progress));
    }

    function responseProgressMetrics(stage, metrics = {}) {
        if (!stage) return {};
        const next = { ...(stage.responseMetrics || {}) };
        const elapsed = boundedPublicInteger(
            metrics.elapsed_seconds,
            MAX_PUBLIC_ELAPSED_SECONDS
        );
        const totalTokens = boundedPublicInteger(metrics.total_tokens);
        const outputTokens = boundedPublicInteger(metrics.output_tokens);
        const receivedChars = boundedPublicInteger(metrics.received_chars);
        const estimatedTokens = boundedPublicInteger(metrics.estimated_output_tokens);
        if (elapsed !== null) next.elapsed_seconds = elapsed;
        if (totalTokens !== null) next.total_tokens = totalTokens;
        if (outputTokens !== null) {
            next.output_tokens = outputTokens;
            delete next.estimated_output_tokens;
        } else if (estimatedTokens !== null) {
            next.estimated_output_tokens = estimatedTokens;
            delete next.output_tokens;
        }
        if (receivedChars !== null) next.received_chars = receivedChars;
        stage.responseMetrics = next;
        renderProgressClock(stage);
        return next;
    }

    function formatResponseProgress(stage, state = "active") {
        const metrics = stage?.responseMetrics || {};
        const items = [];
        if (!stage?.separateRecordedRuns && !stage?.hasServerClock && Number.isSafeInteger(metrics.elapsed_seconds)) {
            items.push(`${metrics.elapsed_seconds.toLocaleString("en-US")}s elapsed`);
        }
        const prefix = ['complete', 'warning'].includes(state) && stage?.rankingUnavailable
            ? 'LLM ranking incomplete · evidence report available'
            : state === "warning" && stage?.evidenceOnlyRecovery
            ? EVIDENCE_RECOVERY_SUMMARY
            : state === "complete" || state === "warning"
            ? responseCompletionSummary(stage)
            : state === "stopped"
                ? "Response stopped"
                : state === "error" ? "Response failed" : "Generating response";
        return substageSummary(stage, items.length ? `${prefix} · ${items.join(" · ")}` : prefix, state);
    }

    function responseCompletionSummary(stage) {
        if (stage?.objectiveReviewStatus !== 'PASS_WITH_WARNINGS') return 'Response ready';
        const specific = factIssueSummary(stage.objectiveFactIssues);
        if (specific) return `Response ready · ${specific}`;
        return stage.objectiveReviewWarnings?.some(message => message.includes(
            'Some reported measurements could not be verified and are labelled in place.'))
            ? 'Response ready · some measurements unverified'
            : 'Response ready · fact review needs attention';
    }

    function rememberObjectiveReview(stage, status, warnings = [], factIssues = null) {
        stage.objectiveReviewStatus = status;
        stage.objectiveReviewWarnings = warnings;
        stage.objectiveFactIssues = factIssues;
        // A final review may arrive after generation has completed. It must
        // explain the output check without inheriting scientific Gate 1 warnings.
        if (['complete', 'warning'].includes(stage.state) && !stage.rankingUnavailable
                && !stage.evidenceOnlyRecovery && stage.summary.textContent.startsWith('Response ready')) {
            if (status === 'PASS_WITH_WARNINGS') setProgressStageState(stage, 'warning');
            else if (status === 'PASS') setProgressStageState(stage, 'complete');
            stage.summary.textContent = formatResponseProgress(stage, stage.state);
        }
    }

    function retrievalCoverageLabel(stage) {
        const incomplete = [...(stage?.operationRows || [])].filter(([, row]) =>
            ['FAILED', 'BLOCKED', 'PARTIAL'].includes(row.state));
        return incomplete.length && incomplete.every(([key]) => key.startsWith('ld:'))
            ? 'LD coverage limited' : 'partial coverage';
    }

    function updateResponseProgress(progress, metrics = {}, state = "active") {
        const stage = progress?.stages?.get("response");
        if (!stage) return false;
        if (state === "active" && ["complete", "warning", "unassessed", "stopped", "error"].includes(stage.state)) {
            state = stage.state;
        }
        responseProgressMetrics(stage, metrics);
        const summary = formatResponseProgress(stage, state);
        stage.summary.textContent = summary;
        if (state === "active" && progress.activeStageId === "response"
                && stage.userExpanded !== false) {
            setProgressStageExpanded(stage, true);
        }
        return true;
    }

    function recordResponseOutput(progress, chunk) {
        const value = String(chunk || "");
        if (!value || progress?.complete) return false;
        const stage = progress?.stages?.get("response");
        if (!stage) return false;
        // Rendered answer chunks can contain program-added evidence/reference
        // text. They are not a second provider completion or token counter.
        if (stage.hasServerClock) return false;
        const received = boundedPublicInteger(stage.responseMetrics?.received_chars) || 0;
        return updateResponseProgress(
            progress,
            { received_chars: Math.min(MAX_PUBLIC_PROGRESS_METRIC, received + value.length) },
            "active"
        );
    }

    function appendChatProgressLine(progress, text, explicitStageId = null, options = {}) {
        let value = String(text || "").trim();
        if (!progress || progress.complete || !value) return false;
        if (progress.retainedRankingProof && value === LEGACY_RANKING_FAILURE_DETAIL) value = RETAINED_RANKING_DETAIL;
        const stageId = CHAT_PROGRESS_STAGE_IDS.has(explicitStageId)
            ? explicitStageId
            : inferProgressStage(value);
        const stage = progress.stages.get(stageId);
        if (!stage || stage.seen.has(value)) return false;
        const reviewOnly = stageId === 'retrieve'
            && /(?:partial coverage|coverage (?:limited|unknown|incomplete)|optional (?:expansion|retrieval)|not admitted|verified proxies retained|no proxies admitted)/i.test(value);
        const retainedWarning = stageId === 'safety' && stage.safetyWarnings?.includes(value) && publicSafetyWarning(value);
        // Ordinary retrieval log volume must not hide an already validated,
        // bounded safety reason. Its own 24-reason limit still applies.
        if (progress.detailCount >= MAX_CHAT_PROGRESS_LINES && !retainedWarning) return false;
        if (stageId === 'retrieve' && stage.currentRetrieval
                && /^(?:Retrieving evidence(?:\.\.\.)?|Planning the evidence retrieval\.\.\.|Validating the selected variants and data access\.\.\.|Preparing the requested Open Targets evidence queries\.\.\.)$/.test(value)) return false;

        let state = options.state || "active";
        if (/^Safety gate stopped:/i.test(value)) state = "stopped";
        if (/PASS_WITH_WARNINGS/i.test(value)) state = "warning";
        if (stageId === "response" && state === "active"
                && ["complete", "warning", "unassessed", "stopped", "error"].includes(stage.state)) {
            return false;
        }
        let summary = String(options.summary || value);
        if (stageId === "safety" && stage.state === "warning"
                && /^Safety gate passed(?::|;)/i.test(value)) {
            // A later legacy hard-gate success message cannot replace the
            // specific nonfatal reasons already reported by the safety audit.
            return false;
        }
        if (stage.seen.has(value)) return false;
        const active = activateProgressStage(progress, stageId, state);
        if (!active) return false;
        if (reviewOnly) return true;
        const row = document.createElement("div");
        row.className = "progress-stage-detail";
        row.textContent = geneticStatusLabel(value, progress.requestedTask);
        row.setAttribute('data-status-source', value);
        const diagnostic = /^(?:Retrieved 0 evidence records\.|Candidate ranking completed .*model output tokens\.|Retrying candidate ranking .*attempt )/.test(value);
        (diagnostic ? stage.diagnosticBody : stage.details).appendChild(row);
        if (diagnostic) stage.diagnostics.hidden = false;
        stage.seen.add(value);
        progress.detailCount += 1;
        if (!diagnostic) stage.summary.textContent = substageSummary(stage, geneticStatusLabel(summary, progress.requestedTask), state);
        setProgressStageExpanded(
            stage,
            state === "active" && progress.activeStageId === stageId
                && stage.userExpanded !== false
        );
        // Preserve the user's reading position while progress details arrive.
        return true;
    }

    function appendPublicChatProgress(progress, content) {
        return appendChatProgressLine(progress, publicChatStatus(content));
    }

    // Reword only already validated public status text. Legacy receipts remain
    // readable without admitting provider messages or changing scientific facts.
    function geneticStatusLabel(value, task = null) {
        if (value === "Building the candidate-gene shortlist from the selected variants' credible-set/LD context and L2G evidence...") {
            return task === 'prioritize_genes' ? 'Discovering genes with eligible molecular or functional evidence…'
                : 'Assessing variant-to-gene links for the selected locus…';
        }
        const fixed = {
            'Analyzing the question and selected variants...': 'Defining the phenotype, selected variants and analysis task…',
            'Identifying the requested analysis workflow...': 'Identifying the genetic analysis task…',
            'Planning the evidence retrieval...': 'Selecting genetic and functional evidence relevant to the question…',
            'Validating the selected variants and data access...': 'Checking variant identifiers, genome build and reference alleles…',
            'Preparing the requested Open Targets evidence queries...': 'Preparing Open Targets genetic evidence queries…',
            'Organizing and validating the retrieved evidence...': 'Reviewing evidence relevance, coverage and scientific limitations…',
            'Preparing the final evidence-grounded response...': 'Interpreting the retrieved genetic and functional evidence…',
            'Checking ranking claims against the retrieved evidence...': 'Verifying gene identifiers, evidence values and citations in the ranking…',
            'Safety gate passed; preparing the evidence-grounded response.': 'Evidence review completed; preparing the scientific interpretation.',
            'Preparing the fixed Auto Analysis workflow...': 'Preparing Known / Novel screening…',
            'Fixed workflow: checking prior phenotype-specific GWAS reports.': 'Checking published GWAS associations for this phenotype.',
            'Fixed workflow: prioritizing genes for loci with no prior report retrieved.': 'Prioritizing genes at loci with no matching prior GWAS report retrieved.'
        };
        if (Object.hasOwn(fixed, value)) return fixed[value];
        const ranking = /^Evidence ready: [\d,]+ audited records\. Prioritizing ([\d,]+) shortlisted genes? from a compact deduplicated set\.\.\.$/.exec(value);
        if (ranking) return `Comparing ${ranking[1]} shortlisted genes using retrieved locus-to-gene evidence…`;
        return value.replace(/^Safety review · PASS_WITH_WARNINGS/, 'Evidence review · scientific limitations')
            .replace(/^Safety review · PASS\b/, 'Evidence review · passed')
            .replace(/^Safety review · STOPPED/, 'Evidence review · stopped')
            .replace(/^(Retrieved [\d,]+ )supporting publications?\.$/, '$1publication records.')
            .replace(/^Independent datasets: /, 'Distinct source datasets: ')
            .replace(/\bmodels\b/gi, 'LLMs').replace(/\bmodel(?!-organism)\b/gi, 'LLM');
    }

    /*
     * Preferred SSE contract (legacy `status` strings remain supported):
     * {type:"stage", stage:"plan|retrieve|safety|response",
     *  state:"pending|running|passed|warning|stopped|failed",
     *  summary:<bounded public summary>, detail:<public status>, metrics:{<allowlisted key>:<int>}}
     * Only allowlisted fields are rendered; arbitrary model text and hidden reasoning are discarded.
     */
    const GATE_GROUP_LABELS = Object.freeze({
        input_identity: 'Input identity', required_data_coverage: 'Required data coverage',
        evidence_coverage: 'Evidence coverage', ranking_prerequisites: 'Requirements for ranking',
        scientific_limitations: 'Scientific limitations', source_gwas_signal: 'Uploaded GWAS signal',
        output_validation: 'Output verification', review_execution: 'Review execution', analysis_method: 'Analysis method',
        scientific_answerability: 'Scientific answerability'
    });
    const GATE_STATUS_LABELS = Object.freeze({ PASS: 'Passed', PASS_WITH_WARNINGS: 'Warning',
        STOPPED: 'Stopped', NOT_RUN: 'Not run' });
    const GATE_CHECK_IDS = new Set(['variant_integrity', 'build_integrity', 'allele_integrity',
        'evidence_availability', 'required_modules', 'optional_evidence_coverage',
        'variant_resolution_coverage', 'candidate_discovery_coverage', 'locus_genes_1mb',
        'direct_locus_gene_evidence', 'current_gwas_signal', 'biological_context_coverage',
        'evidence_independence', 'evidence_conflicts', 'complex_region', 'literature_coverage',
        'novelty_coverage', 'ld_proxy_scope', 'ld_proxy_coverage', 'task_evidence_requirements',
        'ranking_not_attempted', 'ranking_unavailable', 'model_output_grounding', 'report_output_grounding', 'model_review_execution']);
    const ACADEMIC_WARNING_LABELS = Object.freeze({
        W1: 'Task-critical evidence is insufficient', W2: 'Source GWAS significance is not established',
        W3: 'Biological context is not matched', W4: 'Declared retrieval scope is incomplete',
        W5: 'Evidence independence is not established', W6: 'Comparable evidence conflicts',
        W7: 'A selected source SNP identity could not be resolved', W8: 'Complex genomic region',
        W9: 'Original-study provenance is incomplete'
    });
    const REVIEW_METHOD_LABELS = Object.freeze({
        LD_PROXY_METHOD: 'Verified LD proxy evidence', W1_METHOD: 'Evidence for the requested task',
        W2_METHOD: 'Source GWAS significance', W3_METHOD: 'Tissue and cell-type context',
        W4_METHOD: 'Declared retrieval scope', W5_METHOD: 'Study and sample independence',
        W6_METHOD: 'Evidence comparability', W7_METHOD: 'Selected source variant identity',
        W8_METHOD: 'Complex genomic regions', W9_METHOD: 'Source traceability'
    });
    const REVIEW_ASSESSMENT_LABELS = Object.freeze({
        note: 'Method note', not_assessed: 'Not assessed', not_applicable: 'Not applicable'
    });
    const REVIEW_AFFECTED_LABELS = Object.freeze({
        variant_ids: 'Variants', gene_ids: 'Genes', study_ids: 'Studies', modules: 'Retrieval modules'
    });
    const ANSWERABILITY_LABELS = Object.freeze({ sufficient: 'Sufficient', limited: 'Limited', insufficient: 'Insufficient' });
    const OPERATION_LABELS = Object.freeze({ S1: 'No valid source variants', S2: 'Required retrieval did not complete',
        SYSTEM_ERROR: 'Analysis operation failed', MODEL_REVIEW_UNAVAILABLE: 'Additional review unavailable',
        source_variants_excluded: 'Source variants excluded', IDENTITY_UNRESOLVED: 'Source identity verification incomplete' });

    function reviewText(value, limit = 1600) {
        return typeof value === 'string' && value.trim() && value.length <= limit && !/[\r\n\x00-\x08\x0b\x0c\x0e-\x1f]/.test(value)
            ? value.trim() : null;
    }

    function reviewStrings(value, limit = 24) {
        if (!Array.isArray(value) || value.length > limit || value.some(item => !reviewText(item))) return null;
        return [...new Set(value.map(item => item.trim()))];
    }

    function reviewReferences(value = []) {
        if (!Array.isArray(value) || value.length > 32) return null;
        const refs = [];
        for (const item of value) {
            // Older receipts may hold an identifier alone. Do not guess that
            // such an identifier is a PMID, study or retrieved evidence row.
            const ref = typeof item === 'string' ? { type: 'reference', id: item } : item;
            if (!ref || !reviewText(ref.type, 80) || !reviewText(ref.id, 500)) return null;
            refs.push({ type: ref.type.trim(), id: ref.id.trim() });
        }
        return [...new Map(refs.map(ref => [`${ref.type}:${ref.id}`, ref])).values()];
    }

    function reviewScopeDetails(item) {
        const refs = reviewReferences(item.refs);
        if (!refs || (item.impact !== undefined && !reviewText(item.impact))) return null;
        const affected = {};
        if (item.affected !== undefined) {
            if (!item.affected || typeof item.affected !== 'object' || Array.isArray(item.affected)) return null;
            for (const key of Object.keys(REVIEW_AFFECTED_LABELS)) {
                const values = reviewStrings(item.affected[key] || [], 256);
                if (!values) return null;
                if (values.length) affected[key] = values;
            }
        }
        const counts = {};
        if (item.affected_count !== undefined) {
            if (!item.affected_count || typeof item.affected_count !== 'object' || Array.isArray(item.affected_count)) return null;
            for (const key of Object.keys(REVIEW_AFFECTED_LABELS)) {
                if (!Object.hasOwn(item.affected_count, key)) continue;
                const count = item.affected_count[key];
                if (!Number.isSafeInteger(count) || count < 0) return null;
                counts[key] = count;
            }
        }
        return { refs, ...(item.impact ? { impact: item.impact.trim() } : {}),
            ...(Object.keys(affected).length ? { affected } : {}),
            ...(Object.keys(counts).length ? { affected_count: counts } : {}) };
    }

    function mergeReviewDetails(previous, incoming) {
        const affected = {};
        for (const key of Object.keys(REVIEW_AFFECTED_LABELS)) {
            const values = [...new Set([...(previous?.affected?.[key] || []), ...(incoming.affected?.[key] || [])])];
            if (values.length) affected[key] = values;
        }
        const impacts = [...new Set([previous?.impact, incoming.impact].filter(Boolean))];
        const counts = {};
        for (const key of Object.keys(REVIEW_AFFECTED_LABELS)) {
            if (previous?.affected_count?.[key] === undefined && incoming.affected_count?.[key] === undefined) continue;
            counts[key] = Math.max(previous?.affected_count?.[key] || 0, incoming.affected_count?.[key] || 0, affected[key]?.length || 0);
        }
        return { refs: [...new Map([...(previous?.refs || []), ...incoming.refs]
            .map(ref => [`${ref.type}:${ref.id}`, ref])).values()],
            ...(impacts.length ? { impact: impacts.join(' ') } : {}),
            ...(Object.keys(affected).length ? { affected } : {}),
            ...(Object.keys(counts).length ? { affected_count: counts } : {}) };
    }

    function publicRuleCodes(item, previous) {
        const codes = Array.isArray(item?.rule_codes)
            ? item.rule_codes.filter(code => typeof code === 'string' && /^W[1-9](?:\.[1-9])?$/.test(code)) : [];
        return [...new Set([...(previous?.rule_codes || []), ...codes])].slice(0, 8);
    }

    function warningCodeLabel(item) {
        return item.rule_codes?.length ? item.rule_codes.join(', ') : item.code;
    }

    function publicScientificReview(event) {
        if (!Object.hasOwn(event, 'academic_warnings')) return null;
        if (!Object.hasOwn(ANSWERABILITY_LABELS, event.answerability?.level)
                || !Array.isArray(event.academic_warnings) || event.academic_warnings.length > 32) return null;
        const warnings = new Map();
        for (const item of event.academic_warnings) {
            if (!item || !Object.hasOwn(ACADEMIC_WARNING_LABELS, item.code)) return null;
            const facts = reviewStrings(item.facts), reasons = reviewStrings(item.reason_codes),
                checks = reviewStrings(item.check_ids), impacts = reviewStrings(item.impact_domains);
            const scope = reviewScopeDetails(item);
            if (!facts || !reasons || !checks || !impacts || !scope) return null;
            const previous = warnings.get(item.code);
            warnings.set(item.code, { code: item.code, label: ACADEMIC_WARNING_LABELS[item.code],
                rule_codes: publicRuleCodes(item, previous),
                facts: [...new Set([...(previous?.facts || []), ...facts])],
                reason_codes: [...new Set([...(previous?.reason_codes || []), ...reasons])],
                check_ids: [...new Set([...(previous?.check_ids || []), ...checks])],
                impact_domains: [...new Set([...(previous?.impact_domains || []), ...impacts])],
                ...mergeReviewDetails(previous, scope) });
        }
        const methods = new Map();
        if (!Array.isArray(event.method_notes || []) || (event.method_notes || []).length > 32) return null;
        for (const item of event.method_notes || []) {
            if (!item || !Object.hasOwn(REVIEW_METHOD_LABELS, item.code)) return null;
            const facts = reviewStrings(item.facts), checks = reviewStrings(item.check_ids), reasons = reviewStrings(item.reason_codes || []);
            const scope = reviewScopeDetails(item), state = item.assessment_state ?? 'note';
            if (!facts || !checks || !reasons || !scope || !Object.hasOwn(REVIEW_ASSESSMENT_LABELS, state)) return null;
            const previous = methods.get(item.code);
            const states = [previous?.assessment_state, state];
            methods.set(item.code, { code: item.code, label: REVIEW_METHOD_LABELS[item.code],
                assessment_state: states.includes('not_assessed') ? 'not_assessed' : states.includes('note') ? 'note' : 'not_applicable',
                facts: [...new Set([...(previous?.facts || []), ...facts])],
                reason_codes: [...new Set([...(previous?.reason_codes || []), ...reasons])],
                check_ids: [...new Set([...(previous?.check_ids || []), ...checks])], ...mergeReviewDetails(previous, scope) });
        }
        const operations = [];
        if (!Array.isArray(event.operation_issues || []) || (event.operation_issues || []).length > 16) return null;
        for (const item of event.operation_issues || []) {
            // Diagnostic extensions do not invalidate known operational receipts.
            // Unknown messages remain private; only these fixed codes are rendered.
            if (item && !Object.hasOwn(OPERATION_LABELS, item.code) && item.blocking === false) continue;
            if (!item || !Object.hasOwn(OPERATION_LABELS, item.code) || !reviewText(item.message)
                    || !['input', 'retrieval', 'review'].includes(item.scope) || typeof item.blocking !== 'boolean'
                    || !['input_invalid', 'input_excluded', 'service_unavailable', 'system_error', 'review_unavailable'].includes(item.category)) return null;
            const reasons = reviewStrings(item.reason_codes);
            if (!reasons) return null;
            operations.push({ code: item.code, category: item.category, scope: item.scope,
                message: item.message, blocking: item.blocking, reason_codes: reasons });
        }
        const reasons = reviewStrings(event.answerability.reason_codes);
        if (!reasons) return null;
        return { answerability: { level: event.answerability.level, reason_codes: reasons },
            academic_warnings: [...warnings.values()].sort((a, b) => a.code.localeCompare(b.code)),
            operation_issues: [...new Map(operations.map(item => [item.code, item])).values()],
            method_notes: [...methods.values()] };
    }

    function appendReviewFacts(container, item) {
        if (item.facts.length) {
            const label = document.createElement('strong'); label.className = 'review-detail-label'; label.textContent = 'Observed facts';
            const list = document.createElement('ul'); list.className = 'review-facts';
            for (const fact of item.facts) { const row = document.createElement('li'); row.textContent = fact; list.appendChild(row); }
            container.append(label, list);
        }
        if (item.impact) {
            const impact = document.createElement('p'); impact.className = 'review-impact';
            const label = document.createElement('strong'); label.textContent = 'What this limits: ';
            const text = document.createElement('span'); text.textContent = item.impact;
            impact.append(label, text); container.appendChild(impact);
        }
        if (Object.keys(item.affected || {}).length || Object.values(item.affected_count || {}).some(count => count > 0)) {
            const list = document.createElement('dl'); list.className = 'review-affected';
            for (const [key, label] of Object.entries(REVIEW_AFFECTED_LABELS)) {
                const allValues = item.affected?.[key] || [], count = Math.max(item.affected_count?.[key] || 0, allValues.length);
                const values = key === 'modules' ? allValues : allValues.slice(0, 6);
                if (!count) continue;
                const term = document.createElement('dt'); term.textContent = `${label} (${count})`;
                const text = document.createElement('dd');
                if (key === 'modules') {
                    values.forEach((id, index) => {
                        if (index) { const separator = document.createElement('span'); separator.textContent = ' · '; text.appendChild(separator); }
                        const module = document.createElement('span'); module.setAttribute('title', id);
                        module.textContent = Object.hasOwn(RETRIEVAL_MODULE_NAMES, id) ? RETRIEVAL_MODULE_NAMES[id] : id;
                        text.appendChild(module);
                    });
                } else text.textContent = values.join(' · ');
                if (count > values.length) {
                    const omitted = document.createElement('span'); omitted.className = 'review-affected-omitted';
                    omitted.textContent = `${count - values.length} additional identifier${count - values.length === 1 ? '' : 's'} in the review record.`;
                    text.appendChild(omitted);
                }
                list.append(term, text);
            }
            container.appendChild(list);
        }
        const affectedRefKeys = { variant: 'variant_ids', gene: 'gene_ids', study: 'study_ids', module: 'modules' };
        const distinctRefs = item.refs.filter(ref => !(item.affected?.[affectedRefKeys[ref.type]] || []).includes(ref.id));
        if (distinctRefs.length) {
            const refs = document.createElement('p'); refs.className = 'academic-warning-refs';
            refs.textContent = 'Evidence and retrieval records: ' + distinctRefs.slice(0, 6).map(ref => ref.type === 'reference'
                ? ref.id : `${ref.type}: ${ref.id}`).join(' · ');
            if (distinctRefs.length > 6) refs.textContent += ` · ${distinctRefs.length - 6} additional records in the review record.`;
            container.appendChild(refs);
        }
    }

    function publicFactIssues(value) {
        if (!value || value.schema_version !== '1' || !Array.isArray(value.items) || value.items.length > 64) return null;
        const cleanText = (text, limit) => typeof text === 'string'
            ? text.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, '').slice(0, limit).trim() : '';
        const items = [], seen = new Set();
        for (const source of value.items) {
            if (!source || !['unverified', 'corrected'].includes(source.disposition)
                    || !['final_report', 'candidate_ranking'].includes(source.target)) continue;
            const item = {};
            for (const [key, limit] of Object.entries({ issue_id: 100, location: 180, field: 80, metric: 60,
                metric_label: 80, original_value: 1294, replacement: 240, reason_code: 100,
                reason: 600, excerpt: 240, gene_id: 80, gene_symbol: 80, source_type: 60 })) {
                const text = cleanText(source[key], limit);
                if (text) item[key] = text;
            }
            if (!item.issue_id || !item.reason || !item.metric_label || seen.has(item.issue_id)) continue;
            seen.add(item.issue_id);
            item.disposition = source.disposition; item.target = source.target;
            if (['numeric_error', 'relation_ambiguity', 'not_verifiable', 'representation_normalization', 'evidence_contradiction'].includes(source.finding_category)) {
                item.finding_category = source.finding_category;
            }
            const occurrence = source.occurrence;
            if (occurrence && Number.isSafeInteger(occurrence.start) && Number.isSafeInteger(occurrence.end)
                    && occurrence.start >= 0 && occurrence.end > occurrence.start && occurrence.end <= 2000000) {
                item.occurrence = { start: occurrence.start, end: occurrence.end };
            }
            for (const key of ['variant_ids', 'evidence_ids', 'numeric_fact_ids']) if (Array.isArray(source[key])) {
                item[key] = [...new Set(source[key].slice(0, 12).map(text => cleanText(text, 160)).filter(Boolean))];
            }
            items.push(item);
        }
        const omitted = Number.isInteger(value.omitted_count) && value.omitted_count >= 0 && value.omitted_count <= 10000
            ? value.omitted_count : 0;
        // Duplicate items do not inflate counts. Larger recorded totals are
        // used only when the bounded receipt explicitly declares omitted items.
        const count = disposition => items.filter(item => item.disposition === disposition).length;
        const total = (key, disposition) => omitted && Number.isInteger(value[key]) && value[key] >= count(disposition)
            && value[key] <= 10000 ? value[key] : count(disposition);
        return { schema_version: '1', items, unresolved_count: total('unresolved_count', 'unverified'),
            corrected_count: total('corrected_count', 'corrected'), omitted_count: omitted };
    }

    function factIssueSummary(review) {
        if (!review?.unresolved_count) return '';
        const issues = review.items.filter(item => item.disposition === 'unverified');
        const count = review.unresolved_count;
        const ldOnly = issues.length === count && issues.every(item => item.metric === 'r2');
        return `${count} ${ldOnly ? 'LD value' : 'fact'}${count === 1 ? '' : 's'} unverified`;
    }

    function factIssueTitle(item) {
        const source = item.target === 'candidate_ranking' ? 'Gene prioritization' : 'Report';
        const subject = item.gene_symbol || item.gene_id || '';
        const value = item.original_value ? ` = ${item.original_value}` : '';
        const category = { numeric_error: 'Numeric discrepancy', relation_ambiguity: 'Relationship unclear',
            not_verifiable: 'Insufficient evidence to verify', representation_normalization: 'Precision normalisation',
            evidence_contradiction: 'Evidence mismatch' }[item.finding_category];
        return `${category ? category + ' · ' : ''}${subject ? subject + ' · ' : ''}${item.metric_label}${value} · ${source}`;
    }

    function factIssueHandling(item) {
        if (item.disposition === 'corrected') return item.replacement
            ? `Corrected to: ${item.replacement}` : 'The affected statement was corrected using the saved evidence.';
        return item.replacement ? `Shown as: ${item.replacement}`
            : 'The affected value is labelled as unverified; the LLM interpretation is retained.';
    }

    function appendFactIssueCard(container, item, opened = new Set()) {
        const card = document.createElement('details'); card.className = 'fact-issue';
        card.setAttribute('data-issue-id', item.issue_id); card.setAttribute('data-disposition', item.disposition);
        card.factIssueId = item.issue_id; card.open = opened.has(item.issue_id);
        const heading = document.createElement('summary'); heading.textContent = factIssueTitle(item);
        card.appendChild(heading);
        const rows = [
            ['Where', [item.target === 'candidate_ranking' ? 'Gene prioritization' : 'Final report', item.location].filter(Boolean).join(' · ')],
            ['LLM-stated value', item.original_value],
            [item.disposition === 'corrected' ? 'Why corrected' : 'Why not verified', item.reason],
            ['How handled', factIssueHandling(item)],
            ['Original statement', item.excerpt],
            ['Variants', item.variant_ids?.join(' · ')],
            ['Evidence records', item.evidence_ids?.join(' · ')],
            ['Numeric source facts', item.numeric_fact_ids?.join(' · ')]
        ];
        const details = document.createElement('dl'); details.className = 'fact-issue-facts';
        for (const [label, value] of rows) {
            if (!value) continue;
            const term = document.createElement('dt'); term.textContent = label;
            const text = document.createElement('dd'); text.textContent = value;
            details.append(term, text);
        }
        card.appendChild(details); container.appendChild(card);
    }

    function appendFactIssues(container, review, opened = new Set()) {
        if (!review) return;
        const unresolved = review.items.filter(item => item.disposition === 'unverified');
        const corrected = review.items.filter(item => item.disposition === 'corrected');
        if (unresolved.length || review.unresolved_count) {
            const summary = document.createElement('p'); summary.className = 'fact-issue-summary';
            summary.textContent = factIssueSummary(review);
            container.appendChild(summary);
            for (const item of unresolved) appendFactIssueCard(container, item, opened);
        }
        if (corrected.length) {
            const corrections = document.createElement('details'); corrections.className = 'fact-corrections';
            const title = document.createElement('summary');
            title.textContent = `${review.corrected_count} factual correction${review.corrected_count === 1 ? '' : 's'} applied`;
            corrections.appendChild(title);
            for (const item of corrected) appendFactIssueCard(corrections, item, opened);
            container.appendChild(corrections);
        }
        if (review.omitted_count) {
            const more = document.createElement('p'); more.className = 'fact-issue-omitted';
            more.textContent = `${review.omitted_count} additional fact-review items are retained in the review record.`;
            container.appendChild(more);
        }
    }

    function createReportReview(receipt) {
        const review = publicReviewUpdate(receipt);
        if (!review) return null;
        const academic = review.academic_warnings || [], facts = review.output_review?.fact_issues;
        const ranking = review.ranking_review;
        const rankingNeedsReview = ['needs_review', 'partially_verified', 'PASS_WITH_WARNINGS'].includes(ranking?.status);
        // A successful local correction is shown in the correction ledger;
        // it is not presented as a new user-facing warning.  Keep a generic
        // attention message only for an unresolved fact or a recovery that
        // has no bounded fact receipt.
        const outputWarning = review.output_review?.status === 'PASS_WITH_WARNINGS'
            && (!facts || facts.unresolved_count > 0);
        if (!academic.length && !facts?.items.length && !outputWarning && !rankingNeedsReview && !ranking?.fact_issues?.items.length) return null;
        const block = document.createElement('div'); block.className = 'report-warning report-scientific-review';
        if (!academic.length && !outputWarning) block.className += ' report-fact-corrections';
        const heading = document.createElement('p'); heading.className = 'report-warning-summary';
        const count = academic.length;
        if (count) {
            heading.textContent = `Scientific evidence review · ${count} limitation${count === 1 ? '' : 's'}`;
            block.appendChild(heading);
        }
        for (const item of academic) {
            const card = document.createElement('details'); card.className = 'academic-warning';
            card.setAttribute('data-warning-code', item.code);
            const title = document.createElement('summary'); title.textContent = `${warningCodeLabel(item)} · ${item.label}`;
            card.appendChild(title); appendReviewFacts(card, item); block.appendChild(card);
        }
        appendFactIssues(block, facts);
        if (rankingNeedsReview || ranking?.fact_issues?.items.length) {
            const audit = document.createElement('details'); audit.className = 'ranking-fact-review';
            const title = document.createElement('summary');
            const count = ranking.fact_issues?.unresolved_count || 0;
            title.textContent = count ? `Ranking statement checks · ${count} flagged statements` : 'Ranking statement checks';
            audit.appendChild(title);
            const note = document.createElement('p');
            note.textContent = 'These findings concern the original ranking statements. Unverified does not mean numerically wrong. The LLM order is retained.';
            audit.appendChild(note);
            appendFactIssues(audit, ranking.fact_issues);
            const summary = audit.querySelector('.fact-issue-summary');
            if (summary) summary.textContent = 'Open each flagged statement to inspect its original wording and the reason it was not verified.';
            if (rankingNeedsReview && !ranking.fact_issues?.items.length) {
                const missing = document.createElement('p');
                missing.textContent = 'The saved ranking review did not include statement-level findings.';
                audit.appendChild(missing);
            }
            block.appendChild(audit);
        }
        if (outputWarning && !facts?.unresolved_count) {
            const title = document.createElement('p'); title.className = 'report-warning-summary';
            title.textContent = 'Objective fact verification needs attention'; block.appendChild(title);
            for (const message of review.output_review.warnings.length ? review.output_review.warnings
                : ['The saved review did not include the affected facts. Open the original analysis record for details.']) {
                const text = document.createElement('p'); text.className = 'objective-review-limitation';
                text.textContent = message; block.appendChild(text);
            }
        }
        return block;
    }

    function renderScientificReview(progress, review, event) {
        const stage = progress.stages.get('safety');
        const reviewInterrupted = review.operation_issues.some(item => item.blocking);
        progress.scientificBlocking = reviewInterrupted;
        const openedWarnings = new Set(Array.from(stage.gateReview?.children || []).filter(row => row.open && row.warningCode).map(row => row.warningCode));
        stage.gateReview?.remove(); stage.methodNotes?.remove();
        stage.methodNotes = null;
        const block = document.createElement('div'); block.className = 'gate-review academic-review';
        block.setAttribute('data-answerability', review.answerability.level);
        block.setAttribute('data-assessment-state', reviewInterrupted ? 'not_assessed' : 'assessed');
        const heading = document.createElement('div'); heading.className = 'gate-review-heading';
        heading.textContent = reviewInterrupted ? 'Scientific evidence review not completed'
            : 'Scientific evidence review';
        block.appendChild(heading);
        const warnings = reviewInterrupted ? [] : review.academic_warnings.filter(item => !(progress.requestedTask && progress.requestedTask !== 'prioritize_genes'
            && item.impact_domains.length && item.impact_domains.every(domain => domain === 'gene_ranking')));
        for (const item of warnings) {
            const details = document.createElement('details'); details.className = 'academic-warning';
            details.setAttribute('data-warning-code', item.code);
            details.warningCode = item.code; details.open = openedWarnings.has(item.code);
            const title = document.createElement('summary'); title.textContent = `${warningCodeLabel(item)} · ${item.label}`;
            details.appendChild(title);
            appendReviewFacts(details, item);
            block.appendChild(details);
        }
        if (!warnings.length) {
            const text = document.createElement('p'); text.className = 'academic-review-clear';
            text.textContent = reviewInterrupted ? 'Complete the retrieval step before interpreting evidence.'
                : 'No scientific limitations flagged by the configured checks.'; block.appendChild(text);
        }
        stage.gateReview = block; stage.scientificReview = review;
        stage.details.classList.add('progress-scientific-details');
        stage.details.replaceChildren(block, ...stage.details.children);
        // Evidence qualifications belong in the review, without promoting
        // informational notes to scientific warnings or changing the verdict.
        if (review.method_notes?.length) {
            const notes = document.createElement('details'); notes.className = 'review-method-notes';
            const title = document.createElement('summary'); title.textContent = 'Evidence scope and method notes';
            notes.appendChild(title);
            const list = document.createElement('div'); list.className = 'review-method-note-list';
            for (const item of review.method_notes) {
                const note = document.createElement('details'); note.className = 'review-method-note';
                note.setAttribute('data-assessment-state', item.assessment_state);
                const label = document.createElement('summary');
                // A plain method note needs no qualifier; an unassessed or
                // not-applicable check must say so instead of reading as a
                // completed verification.
                label.textContent = item.assessment_state === 'note' ? item.label
                    : `${item.label} · ${REVIEW_ASSESSMENT_LABELS[item.assessment_state]}`;
                note.appendChild(label); appendReviewFacts(note, item); list.appendChild(note);
            }
            notes.appendChild(list);
            stage.methodNotes = notes; stage.details.appendChild(notes);
        }
        const reviewUnavailable = review.operation_issues.some(item => item.scope === 'review');
        const state = reviewInterrupted ? 'pending' : warnings.length || reviewUnavailable ? 'warning' : 'complete';
        setProgressStageState(stage, state);
        stage.summary.textContent = reviewInterrupted ? 'Not assessed · retrieval stopped'
            : warnings.length ? `${ANSWERABILITY_LABELS[review.answerability.level]} · ${warnings.length} scientific limitation${warnings.length === 1 ? '' : 's'}`
            : reviewUnavailable ? 'Review incomplete · review service unavailable' : 'Review completed · no scientific warnings';
        const retrievalStage = progress.stages.get('retrieve');
        retrievalStage.retrievalBlocked = review.operation_issues.some(item => item.blocking && item.scope !== 'review');
        for (const stageId of ['retrieve', 'safety']) progress.stages.get(stageId)?.operationReview?.remove();
        if (review.operation_issues.length) {
            const items = review.operation_issues;
            const panel = document.createElement('div'); panel.className = 'operation-review';
            for (const item of items) {
                const row = document.createElement('div'); row.className = 'operation-review-item';
                row.setAttribute('data-blocking', String(item.blocking));
                const title = document.createElement('strong'); title.textContent = OPERATION_LABELS[item.code];
                const text = document.createElement('p'); text.textContent = item.message;
                row.append(title, text); panel.appendChild(row);
                if (item.blocking && item.scope !== 'review') setProgressStageState(retrievalStage, 'stopped');
            }
            stage.operationReview = panel; stage.body.appendChild(panel);
        }
        renderRetrievalExecution(retrievalStage);
        // Gate 2 verifies objective output facts. Academic warnings do not fail it.
        const response = progress.stages.get('response');
        const openedFacts = new Set(Array.from(response.gateReview?.children || [])
            .filter(row => row.open && row.factIssueId).map(row => row.factIssueId));
        response.gateReview?.remove();
        const statuses = [event.gate_reviews?.gate2?.status, event.output_review?.status].filter(value => Object.hasOwn(GATE_STATUS_LABELS, value));
        const outputStatus = statuses.includes('STOPPED') ? 'STOPPED' : statuses.includes('PASS_WITH_WARNINGS') ? 'PASS_WITH_WARNINGS'
            : statuses.includes('PASS') ? 'PASS' : 'NOT_RUN';
        const factIssues = event.output_review?.fact_issues;
        rememberObjectiveReview(response, outputStatus, event.output_review?.warnings || [], factIssues);
        const output = document.createElement('div'); output.className = 'gate-review objective-review';
        const title = document.createElement('div'); title.className = 'gate-review-heading';
        title.textContent = `Objective fact verification · ${reviewInterrupted && outputStatus === 'NOT_RUN' ? 'Not assessed' : GATE_STATUS_LABELS[outputStatus]}`;
        const text = document.createElement('p'); text.textContent = 'Identifiers · numerical values · evidence references';
        output.append(title, text);
        appendFactIssues(output, factIssues, openedFacts);
        for (const warning of factIssues?.unresolved_count ? [] : event.output_review?.warnings || []) {
            const limitation = document.createElement('p'); limitation.className = 'objective-review-limitation';
            limitation.textContent = warning; output.appendChild(limitation);
        }
        response.gateReview = output; response.details.appendChild(output);
        if (outputStatus !== 'NOT_RUN') {
            response.phaseStates.set('fact_verification', outputStatus === 'STOPPED' ? 'FAILED' : outputStatus === 'PASS_WITH_WARNINGS' ? 'PARTIAL' : 'SUCCESS_DATA');
            renderPhaseRail(response, interpretationPhaseLabels(response));
        }
    }

    function publicGateReview(event) {
        if (!event.gate_reviews || !Array.isArray(event.gate_warning_items)
                || event.gate_warning_items.length > 32) return null;
        const gate1 = event.gate_reviews.gate1?.status;
        const outputStates = [event.gate_reviews.gate2?.status, event.output_review?.status].filter(Boolean);
        if (outputStates.some(status => !Object.hasOwn(GATE_STATUS_LABELS, status))) return null;
        const gate2 = outputStates.includes('STOPPED') ? 'STOPPED'
            : outputStates.includes('PASS_WITH_WARNINGS') ? 'PASS_WITH_WARNINGS'
            : outputStates.includes('PASS') ? 'PASS' : 'NOT_RUN';
        if (!Object.hasOwn(GATE_STATUS_LABELS, gate1) || !Object.hasOwn(GATE_STATUS_LABELS, gate2)) return null;
        const items = [];
        const legacyOutputWarnings = [];
        for (const item of event.gate_warning_items) {
            // Gate 2 is an objective fact-audit receipt.  Its replacement and
            // unresolved entries are rendered from fact_issues below; W15
            // rows are deliberately not rendered as academic warning cards.
            if (item?.gate === 'gate2') {
                // Preserve a validated explanation from older saved runs.
                // New factual corrections remain in fact_issues, not W cards.
                if (item.check_id === 'model_output_grounding' && Array.isArray(item.messages)
                        && item.messages.length <= 16) {
                    for (const message of item.messages) {
                        if (typeof message === 'string' && publicSafetyWarning(`Safety warning: ${message}`)) {
                            legacyOutputWarnings.push(message);
                        }
                    }
                }
                continue;
            }
            if (!item || !GATE_CHECK_IDS.has(item.check_id) || !['gate1', 'gate2'].includes(item.gate)
                    || !['cause', 'consequence', 'output_validation'].includes(item.role)
                    || !Object.hasOwn(GATE_GROUP_LABELS, item.display_group)
                    || !Array.isArray(item.messages) || item.messages.length > 16
                    || !Array.isArray(item.caused_by) || item.caused_by.some(id => !GATE_CHECK_IDS.has(id))) return null;
            const messages = [];
            for (const message of item.messages) {
                const warning = publicSafetyWarning(`Safety warning: ${message}`);
                const stopped = typeof message === 'string' && new Set([
                    'Safety gate stopped: variant identifier integrity could not be verified.',
                    'Safety gate stopped: genome-build integrity could not be verified.',
                    'Safety gate stopped: allele integrity could not be verified.',
                    'Safety gate stopped: no verifiable retrieved evidence or uploaded observations are available.'
                ]).has(message);
                if (!warning && !stopped) return null;
                messages.push(message);
            }
            items.push({ check_id: item.check_id, gate: item.gate, role: item.role,
                display_group: item.display_group, caused_by: [...item.caused_by], messages });
        }
        const outputWarnings = event.output_review?.warnings || [];
        if (!Array.isArray(outputWarnings) || outputWarnings.length > 8) return null;
        const warnings = outputWarnings.filter(message => typeof message === 'string'
            && (publicSafetyWarning(`Safety warning: Response limitation: ${message}`)
                || publicSafetyWarning(`Safety warning: ${message}`)));
        return { gate1, gate2, items, warnings: [...new Set([...legacyOutputWarnings, ...warnings])].slice(0, 8),
            fact_issues: publicFactIssues(event.output_review?.fact_issues) };
    }

    function renderGateReview(progress, review) {
        for (const [gate, stageId, label] of [
            ['gate1', 'safety', 'Gate 1 · Safety & Sufficiency Review'],
            ['gate2', 'response', 'Gate 2 · Objective fact verification']
        ]) {
            const stage = progress.stages.get(stageId);
            if (!stage) continue;
            if (stage.gateReview) stage.gateReview.remove();
            const block = document.createElement('div');
            block.className = 'gate-review';
            block.setAttribute('data-gate', gate);
            const heading = document.createElement('div');
            heading.className = 'gate-review-heading';
            heading.textContent = `${label}: ${GATE_STATUS_LABELS[review[gate]]}`;
            block.appendChild(heading);
            const groups = new Map();
            const nodes = new Map();
            const selected = review.items.filter(item => item.gate === gate);
            const groupFor = id => {
                if (!groups.has(id)) {
                    const group = document.createElement('div');
                    group.className = 'gate-review-group';
                    const title = document.createElement('div');
                    title.className = 'gate-review-group-label';
                    title.textContent = GATE_GROUP_LABELS[id];
                    group.appendChild(title); block.appendChild(group); groups.set(id, group);
                }
                return groups.get(id);
            };
            for (const item of [...selected.filter(item => item.role !== 'consequence'),
                                ...selected.filter(item => item.role === 'consequence')]) {
                const row = document.createElement('div');
                row.className = `gate-review-item gate-review-${item.role}`;
                row.setAttribute('data-check-id', item.check_id);
                for (const message of item.messages) {
                    const text = document.createElement('div');
                    text.textContent = (item.role === 'consequence' ? 'As a result: ' : '') + message;
                    row.appendChild(text);
                }
                const parent = item.role === 'consequence' ? item.caused_by.map(id => nodes.get(id)).find(Boolean) : null;
                (parent || groupFor(item.display_group)).appendChild(row);
                nodes.set(item.check_id, row);
            }
            stage.gateReview = block;
            stage.details.appendChild(block);
            if (gate === 'gate2') {
                appendFactIssues(block, review.fact_issues);
                const warnings = review.fact_issues?.unresolved_count ? [] : review.warnings || [];
                for (const warning of warnings) {
                    const text = document.createElement('p'); text.className = 'objective-review-limitation';
                    text.textContent = warning; block.appendChild(text);
                }
                rememberObjectiveReview(stage, review.gate2, warnings, review.fact_issues);
                stage.phaseStates.set('fact_verification', ({ PASS: 'SUCCESS_DATA', PASS_WITH_WARNINGS: 'PARTIAL',
                    STOPPED: 'FAILED', NOT_RUN: 'NOT_APPLICABLE' })[review.gate2]);
                renderPhaseRail(stage, interpretationPhaseLabels(stage));
            }
            if (gate === 'gate1') {
                setProgressStageState(stage, review.gate1 === 'PASS' ? 'complete' : review.gate1 === 'STOPPED' ? 'stopped' : 'warning');
                const causeIds = new Set(selected.filter(item => item.role !== 'consequence').map(item => item.check_id));
                const roots = selected.filter(item => item.role !== 'consequence'
                    || !item.caused_by.some(id => causeIds.has(id)));
                stage.summary.textContent = compactWarningSummary(review.gate1 === 'PASS_WITH_WARNINGS' ? 'warning' : review.gate1 === 'STOPPED' ? 'stopped' : 'complete',
                    roots.flatMap(item => item.messages), roots.map(item => GATE_GROUP_LABELS[item.display_group]), 'Gate 1',
                    new Set(roots.map(item => item.check_id)).size);
            }
        }
    }

    function publicReviewUpdate(event) {
        if (!event || event.stage !== "safety" || !["passed", "warning", "stopped"].includes(event.state)
                || !Array.isArray(event.warnings) || event.warnings.length > 24) return null;
        // Audit identifiers are optional and independent of the scientific
        // receipt. Invalid extensions must not hide valid legacy warnings.
        const audit = {};
        if (['needs_review', 'partially_verified', 'verified', 'PASS', 'PASS_WITH_WARNINGS'].includes(event.ranking_review?.status)) {
            audit.ranking_review = { status: event.ranking_review.status,
                fact_issues: publicFactIssues(event.ranking_review.fact_issues) };
        }
        const ldPolicy = publicLdReviewPolicy(event.ld_review_policy || event.review_scope?.ld_review_policy);
        if (ldPolicy) audit.ld_review_policy = ldPolicy;
        if (typeof event.review_policy_version === 'string' && /^[A-Za-z0-9_.-]{1,80}$/.test(event.review_policy_version)) {
            audit.review_policy_version = event.review_policy_version;
        }
        if (typeof event.review_scope_sha256 === 'string' && /^[a-fA-F0-9]{64}$/.test(event.review_scope_sha256)) {
            audit.review_scope_sha256 = event.review_scope_sha256.toLowerCase();
        }
        if (Object.hasOwn(event, 'academic_warnings')) {
            const review = publicScientificReview(event);
            if (!review) return null;
            const gates = {};
            for (const gate of ['gate1', 'gate2']) if (Object.hasOwn(GATE_STATUS_LABELS, event.gate_reviews?.[gate]?.status)) {
                gates[gate] = { status: event.gate_reviews[gate].status };
            }
            return { type: 'review_update', stage: 'safety', state: event.state,
                summary: 'Scientific evidence review', warnings: [], ...review, ...audit,
                ...(SCIENTIFIC_TASKS.has(event.requested_task) ? { requested_task: event.requested_task } : {}),
                gate_reviews: gates, gate_warning_items: [],
                ...(Object.hasOwn(GATE_STATUS_LABELS, event.output_review?.status) ? { output_review: {
                    status: event.output_review.status,
                    warnings: (Array.isArray(event.output_review.warnings) ? event.output_review.warnings.slice(0, 8) : [])
                        .filter(warning => publicSafetyWarning(`Safety warning: Response limitation: ${warning}`)),
                    ...(publicFactIssues(event.output_review.fact_issues) ? { fact_issues: publicFactIssues(event.output_review.fact_issues) } : {})
                } } : {}) };
        }
        const warnings = event.warnings.map(value => publicSafetyWarning(`Safety warning: ${value}`));
        if (warnings.some(value => !value)) return null;
        const gateReview = publicGateReview(event);
        return { type: "review_update", stage: "safety", state: event.state, ...audit,
            summary: publicStructuredStageSummary("safety", event.summary, event.state === "passed" ? "complete" : event.state),
            warnings: [...new Set(warnings.map(value => value.slice("Safety warning: ".length)))],
            gateReview,
            ...(gateReview ? {
                gate_reviews: { gate1: { status: gateReview.gate1 }, gate2: { status: gateReview.gate2 } },
                gate_warning_items: gateReview.items, output_review: { status: gateReview.gate2,
                    warnings: gateReview.warnings, ...(gateReview.fact_issues ? { fact_issues: gateReview.fact_issues } : {}) }
            } : {}) };
    }

    function updatePublicReview(progress, event) {
        const update = publicReviewUpdate(event);
        const stage = progress?.stages?.get("safety");
        if (!update || !stage || progress.complete) return false;
        updateRequestedTask(progress, event);
        if (update.ld_review_policy) applyLdReviewPolicy(progress, update.ld_review_policy);
        for (const row of [...stage.details.children]) {
            if (row.textContent.startsWith("Safety warning: ") || (update.academic_warnings
                    && ['Safety review', 'Scientific evidence review'].includes(row.textContent))) {
                stage.seen.delete(row.textContent); row.remove();
                progress.detailCount = Math.max(0, progress.detailCount - 1);
            }
        }
        stage.safetyWarnings = [];
        stage.summary.classList.remove("progress-safety-warnings");
        if (update.academic_warnings) {
            renderScientificReview(progress, update, update);
            setProgressStageExpanded(stage, stage.userExpanded === true);
            return true;
        }
        const state = update.state === "passed" ? "complete" : update.state;
        setProgressStageState(stage, state);
        for (const warning of update.warnings) {
            const line = `Safety warning: ${warning}`;
            stage.safetyWarnings.push(line);
            if (!update.gateReview) appendChatProgressLine(progress, line, "safety", { state, summary: update.summary });
        }
        stage.summary.textContent = compactWarningSummary(state, stage.safetyWarnings);
        if (update.gateReview) renderGateReview(progress, update.gateReview);
        setProgressStageExpanded(stage, stage.userExpanded === true);
        return true;
    }

    function compactWarningSummary(state, reasons = [], categories = [], prefix = 'Safety review', rootCount = null) {
        const unique = [...new Set(reasons)];
        if (!unique.length) return state === 'stopped' ? `${prefix} · Stopped` : state === 'warning' ? `${prefix} · Warnings` : `${prefix} · Passed`;
        if (!categories.length) categories = unique.map(reason => /\bLD\b|population/i.test(reason) ? 'LD coverage'
            : /candidate|ranking|gene-linked|nearby/i.test(reason) ? 'Ranking'
            : /tissue|cell|context/i.test(reason) ? 'Biological context'
            : /independent|independence/i.test(reason) ? 'Independent support'
            : /significan|GWAS/i.test(reason) ? 'Uploaded GWAS signal' : 'Evidence coverage');
        const groups = [...new Set(categories)];
        const count = rootCount ?? unique.length;
        return `${count} ${rootCount !== null ? (state === 'stopped'
            ? (count === 1 ? 'stopping cause' : 'stopping causes') : (count === 1 ? 'warning cause' : 'warning causes'))
            : state === 'stopped' ? 'issues' : count === 1 ? 'warning' : 'warnings'}`
            + (groups.length ? ` · ${groups[0]}${groups.length > 1 ? ` +${groups.length - 1} ${groups.length === 2 ? 'category' : 'categories'}` : ''}` : '');
    }

    function updateProgressScope(progress, metrics, source = 'metrics') {
        if (!metrics || typeof metrics !== 'object') return;
        const counts = progress.scopeCounts || (progress.scopeCounts = {});
        const explicit = progress.explicitScopeCounts || (progress.explicitScopeCounts = new Set());
        for (const key of Object.keys(SCOPE_LABELS)) {
            if (key === 'verified_proxy_count' && progress.ldUnavailable) continue;
            if (typeof metrics[key] !== 'number' || boundedPublicInteger(metrics[key]) === null) continue;
            // Legacy phase candidate_count can mean the downstream shortlist.
            // An explicit pool/shortlist count remains authoritative, including 0.
            if (source === 'phase' && explicit.has(key)) continue;
            counts[key] = metrics[key];
            if (source === 'metrics' && ['candidate_pool_count', 'shortlist_count'].includes(key)) explicit.add(key);
        }
        for (const stageId of ['retrieve', 'response']) {
            let keys = stageId === 'retrieve' ? Object.keys(SCOPE_LABELS).filter(key => !['ranked_count', 'displayed_ranked_count'].includes(key))
                : ['candidate_pool_count', 'shortlist_count', ...(progress.requestedTask && progress.requestedTask !== 'prioritize_genes' ? [] : ['ranked_count', 'displayed_ranked_count'])];
            const profile = taskPresentation(progress.requestedTask);
            if (profile.scope === 'none') keys = [];
            else if (profile.scope === 'variants') keys = keys.filter(key => !['candidate_pool_count', 'shortlist_count', 'ranked_count', 'displayed_ranked_count'].includes(key));
            const labels = profile.scope === 'genes' ? { ...SCOPE_LABELS, candidate_pool_count: 'Genes considered', shortlist_count: 'Genes assessed' } : SCOPE_LABELS;
            const scope = progress.stages.get(stageId)?.scope;
            if (!scope) continue;
            scope.textContent = keys.filter(key => counts[key] !== undefined
                    && (key !== 'displayed_ranked_count' || counts[key] < counts.ranked_count))
                .map(key => `${labels[key]}: ${counts[key].toLocaleString('en-US')}`).join(' · ');
            scope.hidden = !scope.textContent;
        }
    }

    function reportWarningCategories(text) {
        // Category names are fixed UI labels, inferred only from the actual
        // leading warning text. Unknown sentences contribute no invented cause.
        const categories = [];
        const compact = /^\s*(?:Warning:\s*)?Review limitations: ([^.]+)\.(?:\s+(The explanatory synthesis could not be completed reliably; interpretation remains incomplete\.|Some requested interpretation remains unresolved\.))?\s*$/.exec(String(text || ''));
        const compactLabels = {
            'evidence retrieval': 'Evidence coverage', 'optional evidence coverage': 'Evidence coverage',
            'candidate coverage': 'Evidence coverage', 'variant identity coverage': 'Variant identity',
            'ranking requirements': 'Ranking unavailable', 'ranking unavailable': 'Ranking unavailable',
            'uploaded GWAS signal': 'Uploaded GWAS signal', 'tissue/cell relevance': 'Tissue context',
            'independent support': 'Independent support', 'conflicting evidence': 'Evidence conflicts',
            'complex locus': 'Complex locus', 'literature coverage': 'Literature coverage',
            'prior-report coverage (novelty remains unresolved)': 'Evidence coverage',
            'ranking method': 'Evidence-based ranking', 'additional review': 'Additional review',
            'additional quality review unavailable': 'Additional quality review unavailable',
            'LD evidence': 'LD evidence', 'LD coverage': 'LD evidence'
        };
        if (compact && compact[1].split(', ').every(label => Object.hasOwn(compactLabels, label))) {
            const labels = compact[1].split(', ').map(label => compactLabels[label]);
            if (compact[2]) labels.push('Incomplete interpretation');
            return [...new Set(labels)];
        }
        const add = (pattern, label) => { if (pattern.test(text)) categories.push(label); };
        if (String(text || '').includes('Program evidence checks completed, but the model-assisted review was unavailable. The report uses the program-validated evidence and discloses the remaining limitations.')) {
            categories.push('Additional quality review unavailable');
        }
        add(/coverage[^.]*incomplete|retrieval[^.]*incomplete|retrieved[^.]*missing|required[^.]*not complete/i, 'Evidence coverage');
        add(/could not be uniquely identified|variant identifier integrity|identity[^.]*not verified/i, 'Variant identity');
        add(/\bLD\b|population-specific LD/i, 'LD evidence');
        add(/ranking[^.]*fallback|ordered with a deterministic|ranking[^.]*corrected/i, 'Evidence-based ranking');
        add(/ranking[^.]*unavailable|ranking was not started|no supported gene ranking/i, 'Ranking unavailable');
        add(/interpretation[^.]*(?:incomplete|unresolved)|explanatory synthesis could not be completed|unsupported conclusions were omitted/i, 'Incomplete interpretation');
        add(/independent support[^.]*limited|independence[^.]*(?:not|could not)|cohort independence is not established/i, 'Independent support');
        add(/(?:tissue|cell|context)[^.]*(?:could not|not verified|incomplete)|No[^.]*(?:tissue|cell context|relevant context)/i, 'Tissue context');
        add(/No selected variant reaches|Significance[^.]*could not be verified/i, 'Uploaded GWAS signal');
        if (String(text || '').includes('Open Targets target-literature lists were limited to the configured number of records per gene. Additional publications may be absent from this evidence view.')) {
            categories.push('Literature coverage');
        } else {
            add(/No traceable literature|literature[^.]*coverage/i, 'Literature coverage');
        }
        add(/\bMHC\b|complex region/i, 'Complex locus');
        return categories;
    }

    function publicVariantNormalizations(value) {
        if (!Array.isArray(value) || value.length > 2000) return null;
        const pattern = /^([1-9]|1[0-9]|2[0-2]|X|Y|MT)_([1-9][0-9]{0,11})_([ACGT])_([ACGT])$/;
        const fields = ['original_id', 'canonical_id', 'genome_build', 'reason_code', 'source_p_preserved', 'effect_sign_changed'];
        const seen = new Set(), clean = [];
        for (const item of value) {
            if (!item || typeof item !== 'object' || Object.keys(item).length !== fields.length
                    || fields.some(key => !Object.hasOwn(item, key))) return null;
            const before = typeof item.original_id === 'string' ? pattern.exec(item.original_id) : null;
            const after = typeof item.canonical_id === 'string' ? pattern.exec(item.canonical_id) : null;
            if (!before || !after || item.original_id === item.canonical_id || seen.has(item.original_id)
                    || before[1] !== after[1] || before[2] !== after[2] || before[3] !== after[4] || before[4] !== after[3]
                    || item.genome_build !== 'GRCh38'
                    || item.reason_code !== 'ot_reference_allele_order_conflict'
                    || item.source_p_preserved !== true || item.effect_sign_changed !== false) return null;
            seen.add(item.original_id);
            clean.push(Object.fromEntries(fields.map(key => [key, item[key]])));
        }
        return clean;
    }

    function updateVariantNormalizations(progress, event) {
        const incoming = publicVariantNormalizations(event?.normalizations);
        if (!progress || !incoming || !incoming.length) return false;
        const records = new Map((progress.variantNormalizations || []).map(item => [item.original_id, item]));
        for (const item of incoming) {
            const previous = records.get(item.original_id);
            if (previous && (previous.canonical_id !== item.canonical_id || previous.genome_build !== item.genome_build)) return false;
            records.set(item.original_id, item);
        }
        if (records.size > 2000) return false;
        progress.variantNormalizations = [...records.values()];
        let notice = progress.normalizationNotice;
        if (!notice) {
            notice = document.createElement('details'); notice.className = 'variant-normalization-notice';
            const summary = document.createElement('summary');
            const explanation = document.createElement('p');
            explanation.textContent = 'Variant IDs were normalized to the verified reference-allele order for database queries. Original uploaded IDs, P values, beta estimates and odds ratios are retained. This does not infer the effect allele or reorient effect estimates.';
            const rows = document.createElement('ul'); rows.className = 'variant-normalization-rows';
            notice.append(summary, explanation, rows);
            (progress.stages?.get('safety')?.body || progress.root).appendChild(notice);
            progress.normalizationNotice = notice;
        }
        notice.children[0].textContent = `Reference-allele order normalized · ${records.size} variant${records.size === 1 ? '' : 's'}`;
        const rows = notice.children[2]; rows.replaceChildren();
        for (const item of progress.variantNormalizations) {
            const row = document.createElement('li');
            row.textContent = `${item.original_id} → ${item.canonical_id} (${item.genome_build})`;
            rows.appendChild(row);
        }
        return true;
    }

    function appendPublicChatStageEvent(progress, event) {
        if (event?.type === 'analysis_status' && progress) return updateRetainedRankingProof(progress, event);
        if (event?.type === 'variant_normalization') return updateVariantNormalizations(progress, event);
        if (event?.type === "review_update") return updatePublicReview(progress, event);
        if (!progress || progress.complete || !event || !CHAT_PROGRESS_STAGE_IDS.has(event.stage)) return false;
        updateRequestedTask(progress, event);
        if (event.stage === 'retrieve') applyLdReviewPolicy(progress, event.ld_review_policy || event.review_scope?.ld_review_policy);
        const stateMap = Object.freeze({
            pending: "pending",
            running: "active",
            passed: "complete",
            warning: "warning",
            stopped: "stopped",
            failed: "error"
        });
        let state = stateMap[event.state] || "active";
        updateProgressScope(progress, event.metrics);
        const stage = progress.stages.get(event.stage);
        if (progress.historical && event.operation_timings) {
            stage.recordedOperationTimings = new Map(publicOperationTimings(event.operation_timings)
                .map(item => [item.operation_key, item]));
        }
        if (event.stage === 'retrieve') {
            if (['complete', 'warning', 'error', 'stopped'].includes(state)) stage.executionIncomplete = false;
            if (event.summary === 'External evidence unavailable · source data retained') stage.executionFailureDeclared = true;
            else if (state === 'complete' && event.metrics?.total_records > 0) stage.executionFailureDeclared = false;
            if (state === 'error') stage.retrievalFailed = true;
            if (stage.retrievalFailed) state = 'error';
            else if (stage.retrievalBlocked) state = 'stopped';
        }
        if (event.stage === 'response') updateModelUsage(stage, event.model_usage);
        if (event.stage === 'response' && event.ranking_unavailable === true && !progress.retainedRankingProof) stage.rankingUnavailable = true;
        if (event.stage === 'response' && ['complete', 'warning'].includes(state)
                && !stage.rankingUnavailable && String(event.summary || '') === 'Response ready') {
            if (stage.objectiveReviewStatus === 'PASS') state = 'complete';
            else if (stage.objectiveReviewStatus === 'PASS_WITH_WARNINGS') state = 'warning';
        }
        renderHistoricalProvenance(progress, stage, event.history_provenance, state);
        if (state === "active" && (["complete", "stopped", "error"].includes(stage?.state)
                || (event.stage === "response" && stage?.state === "warning"))) {
            return false;
        }
        const stageSeconds = boundedPublicInteger(event.metrics?.stage_elapsed_seconds, MAX_PUBLIC_ELAPSED_SECONDS);
        if (stageSeconds !== null) {
            stage.hasServerClock = true;
            stage.holdClockForServer = false;
            stage.durationSeconds = stageSeconds;
            stage.clockBaseSeconds = stageSeconds;
            stage.clockAnchor = progressNow();
            renderProgressClock(stage);
        }
        if (event.stage === 'safety' && stage.scientificReview) return true;
        if (event.stage === 'response' && progress.requestedTask !== 'general' && !progress.scientificBlocking) {
            if (boundedPublicInteger(event.metrics?.ranking_elapsed_seconds) !== null || event.metrics?.output_tokens > 0) stage.llmObserved = true;
            if (state === 'active') {
                const explicitSubstage = RESPONSE_SUBSTAGES.has(event.substage) ? event.substage : null;
                const verifying = explicitSubstage ? explicitSubstage === 'fact_verification'
                    : /^(?:Validating response|Verifying (?:the )?(?:response|facts)|Objective fact verification)/i.test(String(event.summary || ''))
                        || event.detail === 'Checking ranking claims against the retrieved evidence...';
                const generationStart = /^(?:Generating the evidence-grounded response|Preparing the final (?:evidence-grounded )?response|Prioritizing candidate genes from (?:the retrieved|reviewed) evidence)\.\.\.$/.test(String(event.detail || ''));
                if (explicitSubstage) stage.currentSubstage = explicitSubstage;
                else if (verifying) stage.currentSubstage = 'fact_verification';
                else if (!stage.currentSubstage || generationStart) stage.currentSubstage = 'inference';
                stage.phaseStates.set(stage.currentSubstage, 'RUNNING');
                stage.section.setAttribute('data-current-substage', stage.currentSubstage);
                if (verifying || explicitSubstage === 'fact_verification') stage.phaseStates.set('inference', 'SUCCESS_DATA');
            } else if (state === 'complete' && (stage.phaseStates.has('inference') || stage.llmObserved)) stage.phaseStates.set('inference', 'SUCCESS_DATA');
            if (stage.rankingUnavailable && ['complete', 'warning'].includes(state)) stage.phaseStates.set('inference', 'PARTIAL');
            renderPhaseRail(stage, interpretationPhaseLabels(stage));
        }
        const retrieval = event.stage === 'retrieve' ? publicRetrievalProgress(event.progress) : null;
        if (retrieval) {
            updateRetrievalStep(progress, retrieval, state);
            // History retains the last substep alongside the terminal totals.
            // Render those totals as the completed-stage summary, not a stale wait.
            if (state === 'active') return true;
        }
        if (event.stage === 'retrieve' && state === 'complete') {
            const rows = [...stage.operationRows.values()];
            const failed = rows.filter(row => ['FAILED', 'BLOCKED'].includes(row.state));
            const partial = rows.some(row => row.state === 'PARTIAL'
                && (row.lastValue?.status === 'incomplete' || row.lastValue?.state === 'PARTIAL'
                    || (row.lastValue?.validated_count !== undefined && row.lastValue?.total_count > row.lastValue.validated_count)));
            if (failed.length || partial) state = failed.some(row => row.lastValue?.tool)
                && event.metrics?.total_records === 0 ? 'error' : 'warning';
            stage.optionalLdOnlyWarning = state === 'warning' && rows.filter(row => ['FAILED', 'BLOCKED', 'PARTIAL'].includes(row.state))
                .every(row => row.lastValue?.phase?.startsWith('ld_'));
        }
        let summary = publicStructuredStageSummary(event.stage, event.summary, state);
        if (event.stage === "response" && stage) {
            stage.evidenceOnlyRecovery = state === "warning" && summary === EVIDENCE_RECOVERY_SUMMARY;
        }
        let appended = false;
        let rankingHeartbeat = false;
        const responseHasMetrics = event.stage === "response"
            && event.metrics
            && typeof event.metrics === "object"
            && ["elapsed_seconds", "total_tokens", "output_tokens", "estimated_output_tokens", "received_chars"]
                .some((key) => boundedPublicInteger(
                    event.metrics[key],
                    key === "elapsed_seconds"
                        ? MAX_PUBLIC_ELAPSED_SECONDS
                        : MAX_PUBLIC_PROGRESS_METRIC
                ) !== null);
        if (event.stage === "retrieve" && summary !== "Results retained · processing limit reached" && summary !== 'Evidence retrieval not required'
                && summary !== 'External evidence unavailable · source data retained'
                && event.metrics && typeof event.metrics === "object") {
            const total = event.metrics.total_records;
            const candidates = event.metrics.candidate_genes;
            const safeTotal = Number.isSafeInteger(total) && total >= 0 && total <= 10_000_000
                && (state !== 'active' || total > 0);
            if (safeTotal) stage.retrievalRecordCount = total;
            const safeCandidates = Number.isSafeInteger(candidates)
                && candidates >= 0 && candidates <= 10_000_000
                && progress.scopeCounts?.candidate_pool_count === undefined
                && progress.scopeCounts?.shortlist_count === undefined;
            if (safeTotal || safeCandidates) {
                summary = (state === 'error' ? 'Evidence query failed · coverage unknown'
                    : state === 'stopped' ? 'Evidence retrieval interrupted'
                    : state === 'warning' ? total === 0 ? 'Evidence retrieval incomplete · coverage unknown' : `Evidence retrieved · ${retrievalCoverageLabel(stage)}`
                    : state === 'complete' && safeTotal && total === 0 ? 'Query complete · no matching records'
                    : state === 'complete' ? 'Evidence ready' : 'Retrieving evidence')
                    + (safeTotal ? ` · ${total.toLocaleString("en-US")} records` : "")
                    + (safeCandidates
                        ? ` · ${candidates.toLocaleString("en-US")} candidates`
                        : "");
            }
            if (state === "complete") {
                activateProgressStage(progress, "retrieve", state);
                appended = true;
                const publications = boundedPublicInteger(event.metrics.publications);
                const abstracts = boundedPublicInteger(event.metrics.publication_abstracts);
                const coverage = [];
                if (publications !== null) {
                    coverage.push(`${publications.toLocaleString("en-US")} unique publications`);
                }
                if (abstracts !== null) {
                    coverage.push(`${abstracts.toLocaleString("en-US")} abstracts`);
                }
                if (coverage.length) {
                    appendChatProgressLine(progress, `Publication coverage: ${coverage.join(" · ")}.`,
                        "retrieve", { state, summary });
                }
            }
        }
        if (event.stage === "response" && state === "active"
                && event.metrics && typeof event.metrics === "object") {
            const rankingElapsed = boundedPublicInteger(
                event.metrics.ranking_elapsed_seconds, MAX_PUBLIC_ELAPSED_SECONDS
            );
            const candidates = boundedPublicInteger(event.metrics.candidate_genes);
            if (rankingElapsed !== null && (!progress.requestedTask || progress.requestedTask === 'prioritize_genes')) {
                summary = "Prioritizing"
                    + (candidates !== null
                        ? ` ${candidates.toLocaleString("en-US")} shortlisted genes`
                        : " shortlisted genes")
                    + (stage.hasServerClock ? "" : ` · ${rankingElapsed.toLocaleString("en-US")}s elapsed`);
                activateProgressStage(progress, "response", state);
                rankingHeartbeat = true;
                appended = true;
            }
        }
        if (event.stage === "retrieve" && summary === "Retrieving evidence"
                && stage?.summary?.textContent?.startsWith("Retrieving evidence ·")) {
            // Preserve counted retrieval progress when a module emits only a detail.
            summary = stage.summary.textContent;
        }
        if (event.stage === "safety" && state === "warning") {
            const warning = publicSafetyWarning(event.detail);
            const reasons = stage.safetyWarnings || (stage.safetyWarnings = []);
            if (warning && !reasons.includes(warning) && reasons.length < 24) reasons.push(warning);
            if (reasons.length) {
                summary = compactWarningSummary(state, reasons);
                if (warning) appended = appendChatProgressLine(progress, warning, "safety", { state, summary });
                // Historical replay ends with the saved summary and no detail;
                // keep the reasons collected from its preceding detail events.
                appended = true;
            } else {
                appended = appendChatProgressLine(
                    progress,
                    "Scientific limitations were recorded. Review the available warning details before interpreting the findings.",
                    "safety",
                    { state, summary }
                ) || appended;
            }
        } else if (event.detail && !rankingHeartbeat) {
            const publicDetail = publicChatStatus(event.detail);
            const usedGenericFallback = publicDetail === "Retrieving evidence..."
                && String(event.detail || "").trim() !== publicDetail;
            const misplacedRetrieval = event.stage !== "retrieve" && /^Retrieved /i.test(publicDetail);
            if (!misplacedRetrieval && (event.stage === "retrieve" || !usedGenericFallback)) {
                appended = appendChatProgressLine(
                    progress,
                    publicDetail,
                    event.stage,
                    { state, summary }
                ) || appended;
            }
        }
        const countLabels = Object.freeze({
            variant_annotations: "variant annotations",
            credible_sets: "credible sets",
            variant_consequences: "variant consequences",
            molecular_qtl: "molecular QTL records",
            colocalization: "colocalization records",
            enhancer_links: "enhancer-to-gene links",
            l2g_predictions: "L2G predictions",
            candidate_genes: "candidate-gene identities",
            gene_expression: "gene-expression records",
            gene_function: "gene-function records",
            pathways: "biological-pathway records",
            publications: "publications",
            publication_abstracts: "publication abstracts",
            total_records: "evidence records"
        });
        if (event.stage === "retrieve" && state === "active"
                && event.metrics && typeof event.metrics === "object") {
            Object.entries(event.metrics).forEach(([key, rawCount]) => {
                const count = Number(rawCount);
                const label = countLabels[key];
                if (!label || !Number.isSafeInteger(count) || count < 0
                        || count > MAX_PUBLIC_PROGRESS_METRIC) return;
                appended = appendChatProgressLine(
                    progress,
                    `Retrieved ${count.toLocaleString("en-US")} ${label}.`,
                    event.stage,
                    { state, summary }
                ) || appended;
            });
        }
        if (responseHasMetrics && rankingHeartbeat) responseProgressMetrics(stage, event.metrics);
        if (responseHasMetrics && !rankingHeartbeat) {
            activateProgressStage(progress, "response", state);
            appended = updateResponseProgress(
                progress,
                event.metrics,
                state
            ) || appended;
            summary = stage?.summary?.textContent || summary;
        }
        if (!appended) {
            appended = appendChatProgressLine(
                progress,
                summary,
                event.stage,
                { state, summary }
            );
        }
        if (stage) {
            if (event.stage === 'response' && ['complete', 'warning'].includes(state)
                    && stage.objectiveReviewStatus === 'PASS_WITH_WARNINGS'
                    && String(summary).startsWith('Response ready')
                    && !stage.rankingUnavailable && !stage.evidenceOnlyRecovery) {
                summary = formatResponseProgress(stage, state);
            }
            if (event.stage === 'retrieve' && state === 'active' && stage.currentRetrieval) summary = retrievalStepSummary(stage.currentRetrieval.value, false, progress.requestedTask);
            stage.summary.textContent = substageSummary(stage, geneticStatusLabel(summary, progress.requestedTask), state);
            if (event.stage === 'retrieve' && stage.retrievalBlocked) stage.summary.textContent = 'Required retrieval did not complete';
            setProgressStageState(stage, state);
            if (state !== "active") setProgressStageExpanded(stage, false);
            if (event.stage === "response" && ["complete", "warning", "stopped", "error"].includes(state)) {
                for (const previous of progress.stages.values()) freezeProgressStage(previous);
                stopProgressClock(progress);
            }
        }
        return appended;
    }

    function publicStructuredStageSummary(stageId, value, state) {
        const text = String(value || "").trim();
        if (stageId === 'retrieve' && ['error', 'stopped', 'warning'].includes(state)
                && /^(?:Retrieving evidence|Evidence ready|Evidence retrieved|Evidence retrieval completed)/i.test(text)) {
            return state === 'error' ? 'Evidence query failed · coverage unknown'
                : state === 'stopped' ? 'Evidence retrieval interrupted' : 'Evidence retrieval incomplete';
        }
        if (stageId === 'response' && /^(?:Model|LLM) ranking incomplete · evidence report available$/.test(text))
            return 'LLM ranking incomplete · evidence report available';
        if (stageId === "retrieve" && state === "warning" && text === "Results retained · processing limit reached") return text;
        if (stageId === 'retrieve' && ['warning', 'error', 'stopped'].includes(state)
                && text === 'External evidence unavailable · source data retained') return text;
        if (stageId === "response" && state === "warning" && text === "Source GWAS summary ready · external review incomplete") return text;
        if (stageId === 'plan' && /^(?:Preparing fixed workflow|Workflow ready · (?:prior-report screening|gene prioritization))$/.test(text)) return text;
        if (stageId === "response" && state === "warning" && text === EVIDENCE_RECOVERY_SUMMARY) return text;
        const bounded = "(?:0|[1-9][0-9]{0,6}|[1-9][0-9]{0,2}(?:,[0-9]{3}){1,2})";
        const patterns = {
            plan: new RegExp(
                "^(?:Understanding request|Request understood)"
                + "(?: · (?:general conversation|variant description|variant-to-gene associations|gene prioritization|target assessment|disease associations|drug repurposing|phenotype context defined|no phenotype context)){0,3}$",
                "i"
            ),
            retrieve: new RegExp(
                `^(?:Evidence retrieval not required|(?:Retrieving evidence|Evidence ready|Evidence retrieved)(?: · ${bounded} records?)?(?: · ${bounded} candidates?)?)$`,
                "i"
            ),
            safety: new RegExp(
                `^Safety review · (?:PASS|PASS_WITH_WARNINGS|STOPPED)(?: · ${bounded} (?:warnings?|limitations?))?$`
            ),
            response: /^(?:Preparing response|Generating response|Validating response|Response ready)(?: · grounded fallback used)?$|^Response stopped · reason reported$/i
        };
        if (text && patterns[stageId]?.test(text)) {
            return stageId === 'retrieve' && state === 'complete'
                ? text.replace(/^Retrieving evidence/i, 'Evidence retrieved') : text;
        }
        const defaults = {
            plan: state === 'complete' ? 'Request understood' : 'Understanding request',
            retrieve: state === 'error' ? 'Evidence query failed · coverage unknown'
                : state === 'stopped' ? 'Evidence retrieval interrupted'
                : state === 'warning' ? 'Evidence retrieval incomplete'
                : state === 'complete' ? 'Evidence retrieval completed' : 'Retrieving evidence',
            safety: state === "stopped"
                ? "Safety review · STOPPED"
                : state === "warning" ? "Safety review · PASS_WITH_WARNINGS" : "Safety review",
            response: state === "complete"
                ? "Response ready"
                : state === "stopped" ? "Response stopped · reason reported" : "Generating response"
        };
        return defaults[stageId];
    }

    function completeChatProgress(progress) {
        if (!progress || progress.complete) return;
        // Saved failed workflows have no live error handler to finalize them.
        // Their unrun stages must not become successful during History replay.
        if ([...progress.stages.values()].some(stage => stage.state === 'error')) {
            failChatProgress(progress);
            return;
        }
        if (progress.historical) {
            // Restoring a saved snapshot is not a stream-completion event.
            // Keep only recorded terminal states; missing/interrupted steps
            // must not acquire successful checkmarks from later activity.
            for (const stage of progress.stages.values()) {
                if (stage.state === 'active') {
                    if (stage.id === 'retrieve') stage.executionIncomplete = true;
                    setProgressStageState(stage, 'warning');
                    stage.summary.textContent = 'Incomplete · completion not recorded';
                } else if (stage.state === 'pending') {
                    stage.summary.textContent = progress.scientificBlocking ? 'Not assessed' : 'Stage not recorded';
                }
                setProgressStageExpanded(stage, false);
            }
            disposeProgress(progress);
            return;
        }
        for (const stage of progress.stages.values()) {
            if (progress.scientificBlocking && stage.state === 'pending') {
                stage.summary.textContent = ['safety', 'response'].includes(stage.id) ? 'Not assessed' : 'Not run';
                setProgressStageExpanded(stage, false); continue;
            }
            if (stage.state === "pending" || stage.state === "active") {
                if (stage.id === 'response' && stage.state === 'active') setProgressStageState(stage, 'complete');
                else if (stage.state === 'active') {
                    if (stage.id === 'retrieve') stage.executionIncomplete = true;
                    setProgressStageState(stage, 'warning');
                    stage.summary.textContent = 'Completion not recorded';
                } else stage.summary.textContent = 'Stage not recorded';
            }
            if (stage.id === "response" && stage.state === "complete") {
                stage.summary.textContent = formatResponseProgress(stage, "complete");
                if (stage.telemetry) stage.telemetry.textContent = stage.summary.textContent;
            }
            setProgressStageExpanded(stage, false);
        }
        progress.complete = true;
        stopProgressClock(progress);
        progress.root.classList.add(progress.scientificBlocking ? 'thinking-failed' : 'thinking-done');
    }

    function failChatProgress(progress) {
        if (!progress || progress.complete) return;
        const stage = [...progress.stages.values()].find(row => row.state === 'error')
            || progress.stages.get(progress.activeStageId || "plan");
        setProgressStageState(stage, "error");
        if (stage && stage.summary.textContent === "Waiting") stage.summary.textContent = "Failed";
        progress.complete = true;
        for (const row of progress.stages.values()) {
            if (row.state === 'pending') row.summary.textContent = 'Not run';
            freezeProgressStage(row);
        }
        stopProgressClock(progress);
        progress.root.classList.add("thinking-failed");
    }

    function publicSafetyWarning(content) {
        const value = String(content || "").trim();
        const prefix = "Safety warning: ";
        if (!value.startsWith(prefix) || value.length > 2000 || /[\r\n]/.test(value)) return null;
        const reason = value.slice(prefix.length);
        // This vocabulary mirrors the server's immutable safety-fact helper;
        // model explanations, provider errors and arbitrary text stay private.
        const fixedReasons = new Set([
            "Open Targets target-literature lists were limited to the configured number of records per gene. Additional publications may be absent from this evidence view.",
            "Europe PMC publication metadata was incomplete for some already identified PMIDs. Their identifiers and available metadata were retained.",
            "Program evidence checks completed, but the model-assisted review was unavailable. The report uses the program-validated evidence and discloses the remaining limitations.",
            "Evidence independence could not be verified.",
            "Conflicting evidence requires interpretation.",
            "No selected variant reaches the configured significance threshold in the loaded GWAS dataset.",
            "Significance in the loaded GWAS dataset could not be verified.",
            "No phenotype-specific tissue or cell context was defined.",
            "No retrieved direct evidence matches the defined phenotype-relevant context.",
            "The retrieved evidence did not establish coverage of the defined phenotype-relevant tissues or cell types.",
            "Phenotype-relevant tissue or cell coverage could not be verified.",
            "The selected locus overlaps the MHC region; correlated signals and study coverage require caution.",
            "The selected locus overlaps a configured complex region; association evidence requires caution.",
            "No traceable literature reference is available for the retrieved evidence.",
            "Some selected variants could not be uniquely identified. Their uploaded values are preserved, but missing matches for them cannot establish novelty.",
            "Candidate ranking was corrected and validated after one repair.",
            "Candidate ranking used a validated evidence-based fallback after a model failure.",
            "Candidate ranking remains unavailable; retrieved evidence is retained.",
            "Candidate ranking was not started because the available evidence did not satisfy its requirements. Retrieved findings remain available; no ranking-model response was generated or rejected.",
            "Candidate discovery is incomplete. Ranking is limited to the directly supported candidates retrieved so far; additional candidates may be missing.",
            "Candidate genes were ordered with a deterministic evidence-based method after the ranking synthesis could not be validated. This method ranks available support; its scores are not causal probabilities.",
            "Prior-report lookup is incomplete; novelty remains unresolved. Gene ranking is exploratory and uses only the available direct evidence.",
            "Prior-report or LD coverage is incomplete; novelty remains unresolved. Available evidence can still be interpreted within its supported scope.",
            "Candidate-gene assessment includes verified LD proxies. Evidence measured at a proxy supports the selected locus indirectly through population-specific LD; it is not a direct measurement of the uploaded SNPs. Uploaded SNP counts and P values are unchanged.",
            "LD proxy coverage is incomplete. Available source-SNP and verified proxy evidence is retained, but additional linked variants and candidate genes may be missing.",
            "LD proxies were identified, but evidence retrieval for some proxies was incomplete. Validated source-SNP and proxy evidence is retained; candidate-gene coverage may be limited.",
            "The LD-expanded analysis includes a proxy in the Open Targets MHC credible-set exclusion interval; correlated signals and study coverage require caution.",
            "Prior-report lookup is incomplete; novelty remains unresolved. Gene evidence was assessed, but no supported gene ranking is available.",
            "No annotated gene was found within the selected locus expanded by 1 Mb on each side. Candidate-gene ranking is unavailable; verified findings are retained.",
            "Annotated-gene coverage for the selected locus expanded by 1 Mb on each side could not be verified; this is a technical coverage gap, not a confirmed gene desert. Candidate-gene ranking is unavailable; verified findings are retained.",
            "No gene-linked molecular QTL, colocalization, enhancer/regulatory, or high-impact coding/splice evidence was found after the configured channels completed. Candidate-gene ranking is unavailable; verified findings are retained.",
            "Response limitation: The explanatory synthesis could not be completed reliably. Verified findings and references are provided, but interpretation remains incomplete.",
            "Response limitation: Some requested interpretation remains unresolved. The report retains verified findings and supported statements; unsupported conclusions were omitted.",
            "Response limitation: Some reported measurements could not be verified and are labelled in place. The model's interpretation and verified evidence are retained.",
            "An additional evidence-review check requires caution; see the report limitations."
        ]);
        if (fixedReasons.has(reason)) return value;
        if (/^Candidate ranking is unavailable because [a-z][a-z' -]{3,100}; retrieved evidence is retained\.$/.test(reason)) return value;
        const mixedRanking = /^([1-9][0-9]?) validated model ranking entries and their original ranking positions were retained; evidence-based rules filled the remaining positions\. This combined ranking orders available support; its scores are not causal probabilities\.$/.exec(reason);
        if (mixedRanking && Number(mixedRanking[1]) <= 20) return value;
        const configuredWindow = /^(?:No annotated gene was found within the selected locus expanded by|Annotated-gene coverage for the selected locus expanded by) ([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?) Mb on each side(?:\. Candidate-gene ranking is unavailable; verified findings are retained\.| could not be verified; this is a technical coverage gap, not a confirmed gene desert\. Candidate-gene ranking is unavailable; verified findings are retained\.)$/.exec(reason);
        if (configuredWindow && Number(configuredWindow[1]) > 0 && Number(configuredWindow[1]) <= 1000) return value;
        const countReason = reason.match(/^(?:Direct|Retrieved) evidence spans only (0|[1-9][0-9]{0,7}) distinct source study\/dataset identifier\(s\); cohort independence is not established\.$/)
            || reason.match(/^Evidence spans only (0|[1-9][0-9]{0,7}) independent dataset\(s\); independent support is limited\.$/)
            || reason.match(/^(0|[1-9][0-9]{0,7}) evidence conflict\(s\) require interpretation\.$/);
        if (countReason && Number(countReason[1]) <= MAX_PUBLIC_PROGRESS_METRIC) return value;
        const missingProvenance = /^([1-9][0-9]{0,7}) original study\/source group\(s\) lack an identifiable PMID or DOI; (0|[1-9][0-9]{0,7}) traceable literature reference\(s\) remain available\.$/.exec(reason);
        if (missingProvenance && missingProvenance.slice(1).every(count => Number(count) <= MAX_PUBLIC_PROGRESS_METRIC)) return value;
        const modules = "[a-z][a-z0-9_.]*(?:, [a-z][a-z0-9_.]*)*";
        if (new RegExp(`^Required retrieval modules were technically incomplete: ${modules}\\.$`).test(reason)
                || new RegExp(`^Direct locus-to-gene evidence coverage could not be established because retrieval was incomplete for: ${modules}\\. This technical gap is not evidence that the locus lacks molecular links\\. Candidate-gene ranking is unavailable; verified findings are retained\\.$`).test(reason)) return value;
        const optional = reason.match(/^Optional evidence coverage was incomplete: (.+)\. Missing records are not evidence of absence\.$/);
        const labels = new Set([
            "Open Targets target literature and publication metadata", "Open Targets target literature", "Europe PMC publication metadata",
            "Europe PMC literature", "Colocalization", "Molecular QTL", "Enhancer-to-gene links",
            "Variant annotations", "Variant consequences", "Gene expression", "Gene function",
            "Regional gene context", "Locus-to-gene evidence", "Other optional evidence",
            "GWAS study lookup", "Credible-set lookup", "Fine-mapping lookup", "Regional gene lookup"
        ]);
        if (optional && labels.has(optional[1])) return value;
        return null;
    }

    function publicChatStatus(content) {
        const value = String(content || "").trim();
        const safetyWarning = publicSafetyWarning(value);
        if (safetyWarning) return safetyWarning;
        const fixedStatuses = new Set([
            "Model ranking did not complete. Retrieved evidence remains available; no rule-based ranking was substituted.",
            "Model ranking was not started because no verified ranking model is configured. Retrieved evidence remains available.",
            "Completing bibliographic details for cited publications...",
            "Preparing the evidence report and references...",
            "Analyzing the question and selected variants...",
            "Identifying the requested analysis workflow...",
            "Planning the evidence retrieval...",
            "Validating the selected variants and data access...",
            "Preparing the requested Open Targets evidence queries...",
            "Building the candidate-gene shortlist from the selected variants' credible-set/LD context and L2G evidence...",
            "Defining the +/-1 Mb candidate-gene window and retrieving gene-centric evidence...",
            "Defining phenotype-relevant tissues and cell types...",
            "Phenotype scope: target phenotype not resolved in EFO.",
            "Phenotype scope: target phenotype only; related phenotypes not assessed.",
            "Preparing the candidate-gene target profiles...",
            "Candidate ranking in progress...",
            "Checking ranking claims against the retrieved evidence...",
            "No external evidence retrieval was required.",
            "Organizing and validating the retrieved evidence...",
            "Organizing evidence relationships and checking retrieval coverage...",
            "Checking evidence sufficiency and reviewing the supported conclusions...",
            "Checking selected and linked variant identities against the saved evidence...",
            "Checking the retrieved evidence before prior-report comparison...",
            "Checking prior phenotype associations and reference LD...",
            "Prioritizing candidate genes from the retrieved evidence...",
            "Prioritizing candidate genes from reviewed evidence...",
            "Preparing the final response...",
            "Preparing the final evidence-grounded response...",
            "Preparing the safe abstention response...",
            "Safety gate passed: sufficient evidence to prepare the response.",
            "Safety gate stopped: insufficient evidence for a substantive ranking.",
            "Safety gate stopped: variant identifier integrity could not be verified.",
            "Safety gate stopped: genome-build integrity could not be verified.",
            "Safety gate stopped: allele integrity could not be verified.",
            "Safety gate stopped: required retrieval modules did not complete.",
            "Safety gate stopped: selected variants did not pass the GWAS-signal requirement.",
            "Safety gate stopped: annotated-gene coverage within 1 Mb was insufficient.",
            "Safety gate stopped: no direct locus-to-gene evidence was found.",
            "Safety gate stopped: the candidate-ranking output failed validation.",
            "Safety gate stopped: a required validation check failed.",
            "Safety gate passed; preparing the evidence-grounded response.",
            "Generating the evidence-grounded response...",
            "Formatting the response and references...",
            "Preparing the fixed Auto Analysis workflow...",
            "Fixed workflow: checking prior phenotype-specific GWAS reports.",
            "Fixed workflow: prioritizing genes for loci with no prior report retrieved."
        ]);
        if (fixedStatuses.has(value)) return value;

        const workflow = value.match(/^Intent identified: ([a-z -]+)\.$/);
        const allowedWorkflows = new Set([
            "general conversation",
            "variant description",
            "variant-to-gene associations",
            "gene prioritization",
            "target assessment",
            "disease associations",
            "drug repurposing"
        ]);
        if (workflow && allowedWorkflows.has(workflow[1])) return value;

        const candidateCount = value.match(
            /^Identified ([1-9][0-9]{0,2}) candidate genes; preparing target profiles\.\.\.$/
        );
        if (candidateCount && Number(candidateCount[1]) <= 200) return value;

        const contextCount = value.match(
            /^Biological context defined: (0|[1-9][0-9]{0,2}) relevant tissues\/cell types\.$/
        );
        if (contextCount && Number(contextCount[1]) <= 200) return value;

        if (/^Phenotype scope: target plus [0-8] related phenotypes from the saved EFO scope\.$/.test(value)) return value;

        const allowedEvidenceLabels = [
            "target expression, function, safety, and tractability data",
            "selected variant and gene identifiers",
            "drug, indication, and clinical evidence",
            "pharmacogenomic evidence",
            "model-organism phenotypes",
            "Gene Ontology annotations",
            "colocalization evidence",
            "molecular QTL evidence",
            "GWAS study metadata",
            "enhancer-to-gene links",
            "target-disease evidence",
            "candidate-gene identities",
            "gene-set enrichment data",
            "variant consequences",
            "fine-mapping data",
            "molecular interactions",
            "supporting literature",
            "clinical candidates",
            "mechanisms of action",
            "variant annotations",
            "biological pathways",
            "genetic constraint",
            "target tractability",
            "disease profile data",
            "credible sets",
            "regulatory links",
            "gene expression",
            "gene function",
            "target profiles",
            "target safety",
            "disease profiles",
            "drug indications",
            "L2G predictions",
            "known drugs",
            "pathways",
            "the requested evidence"
        ];
        const allowedRetrievedLabels = new Set([
            "variant annotation", "variant annotations",
            "GWAS study record", "GWAS study records",
            "credible set", "credible sets",
            "fine-mapping record", "fine-mapping records",
            "variant consequence", "variant consequences",
            "molecular QTL record", "molecular QTL records",
            "colocalization record", "colocalization records",
            "enhancer-to-gene link", "enhancer-to-gene links",
            "L2G prediction", "L2G predictions",
            "candidate-gene identity", "candidate-gene identities",
            "gene-function record", "gene-function records",
            "gene-expression record", "gene-expression records",
            "Gene Ontology annotation", "Gene Ontology annotations",
            "biological-pathway record", "biological-pathway records",
            "genetic-constraint record", "genetic-constraint records",
            "target-tractability record", "target-tractability records",
            "target-safety record", "target-safety records",
            "molecular-interaction record", "molecular-interaction records",
            "model-organism phenotype record", "model-organism phenotype records",
            "publication record", "publication records",
            "publication abstract", "publication abstracts",
            "supporting publication", "supporting publications",
            "drug record", "drug records",
            "mechanism-of-action record", "mechanism-of-action records",
            "drug-indication record", "drug-indication records",
            "drug-warning record", "drug-warning records",
            "adverse-event record", "adverse-event records",
            "pharmacogenomic record", "pharmacogenomic records",
            "clinical-target record", "clinical-target records",
            "other evidence record", "other evidence records",
            "evidence record", "evidence records"
        ]);
        // Accept the previous count wording during rolling upgrades, but only for
        // evidence labels already admitted by the strict public vocabulary above.
        allowedEvidenceLabels.forEach((label) => {
            allowedRetrievedLabels.add(`${label} record`);
            allowedRetrievedLabels.add(`${label} records`);
        });

        function isPublicEvidenceDescription(description) {
            if (!description || description.length > 300 || /[\r\n]/.test(description)) {
                return false;
            }
            const qtlLabel = "(?:eQTL|pQTL|sQTL|tuQTL|sc-eQTL|sc-pQTL|sc-sQTL|sc-tuQTL)";
            const qtlGroup = new RegExp(
                `\\(${qtlLabel}(?:, ${qtlLabel})*(?:,? and ${qtlLabel})?\\)`,
                "g"
            );
            let remainder = description.replace(qtlGroup, "");
            allowedEvidenceLabels.forEach((label) => {
                remainder = remainder.split(label).join("");
            });
            remainder = remainder.replace(/,/g, "").replace(/\band\b/g, "").replace(/\s/g, "");
            return remainder === "";
        }

        const planned = value.match(/^Planned evidence: (.+)\.$/);
        if (planned && isPublicEvidenceDescription(planned[1])) return value;
        const fetching = value.match(/^Fetching (.+)\.\.\.$/);
        if (fetching && isPublicEvidenceDescription(fetching[1])) return value;
        const boundedCount = "(0|[1-9][0-9]{0,7}|[1-9][0-9]{0,2}(?:,[0-9]{3}){1,2})";
        if (/^Retrying candidate ranking from saved evidence \(attempt [1-3]\/3\)\.$/.test(value)) return value;
        const retrieved = value.match(new RegExp(`^Retrieved ${boundedCount} (.+)\\.$`));
        if (retrieved
                && Number(retrieved[1].replace(/,/g, "")) <= 10_000_000
                && allowedRetrievedLabels.has(retrieved[2])) {
            return value;
        }
        const rankingStarted = value.match(new RegExp(
            `^Evidence ready: ${boundedCount} audited records\\. Prioritizing `
            + `${boundedCount} shortlisted (?:gene|genes) from a compact deduplicated set\\.\\.\\.$`
        ));
        if (rankingStarted
                && Number(rankingStarted[1].replace(/,/g, "")) <= 10_000_000
                && Number(rankingStarted[2].replace(/,/g, "")) <= 10_000_000) {
            return value;
        }
        const rankingCompleted = value.match(new RegExp(
            `^Candidate ranking completed(?: for ${boundedCount} (?:gene|genes))?`
            + `(?: · ${boundedCount} model output tokens)?\\.$`
        ));
        if (rankingCompleted
                && [...rankingCompleted].slice(1).filter(Boolean).every(
                    (item) => Number(item.replace(/,/g, "")) <= 10_000_000
                )) {
            return value;
        }
        const resolved = value.match(
            new RegExp(`^Resolved ${boundedCount} selected identifiers?\\.$`)
        );
        if (resolved && Number(resolved[1].replace(/,/g, "")) <= 10_000_000) return value;

        const gatePassed = value.match(
            new RegExp(
                `^Safety gate passed: ${boundedCount} decision-eligible evidence records? `
                + `across ${boundedCount} (?:traceable source identifiers?|(?:distinct|independent) datasets?)\\.$`
            )
        );
        if (gatePassed
                && Number(gatePassed[1].replace(/,/g, "")) <= 10_000_000
                && Number(gatePassed[2].replace(/,/g, "")) <= 10_000_000) {
            return value;
        }
        const stoppedReasons = new Set([
            "the selected variants did not pass the GWAS-signal requirement",
            "the permitted evidence was insufficient",
            "the retrieved evidence contained unresolved conflicts",
            "the analysis input did not pass validation",
            "the configured data-safety policy blocked this analysis",
            "a required evidence service was unavailable",
            "the analysis did not complete its required validation",
            "the evidence requirements were not satisfied"
        ]);
        const gateStopped = value.match(/^Safety gate stopped: ([a-z -]+)\.$/);
        if (gateStopped && stoppedReasons.has(gateStopped[1])) return value;

        // Keep older services usable while never rendering arbitrary worker
        // telemetry, provider responses, paths, or model reasoning.
        if (value === "Preparing an evidence-grounded answer...") {
            return "Preparing the final evidence-grounded response...";
        }
        return "Retrieving evidence...";
    }

    function createAssistantMessage() {
        const messages = document.getElementById("chat-messages");
        const message = document.createElement("div");
        message.className = "message assistant";
        const content = document.createElement("div");
        content.className = "message-content assistant-message";
        message.appendChild(content);
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
        return { message, content };
    }

    async function querySingleMessage(question, request) {
        addMessage("user", question);
        const priorHistory = getChatHistory();
        addToChatHistory("user", question);
        const region = ManhattanPlot.getSelectedRegion();
        const selection = region?.snps
            ? selectVariantsForRegion(region)
            : (() => {
                const values = ManhattanPlot.getVariantIds();
                return {
                    variantIds: values.slice(0, MAX_CHAT_VARIANTS),
                    totalCount: values.length,
                    selectedCount: Math.min(values.length, MAX_CHAT_VARIANTS),
                    truncated: values.length > MAX_CHAT_VARIANTS
                };
            })();
        const variants = selection.variantIds;
        const pending = createTransientChatStatus(
            "Analyzing the question and selected variants..."
        );
        request.progress = pending;
        let assistant = null;
        let content = "";
        let reviewUpdate = null;
        if (selection.truncated) addMessage("system", selectionDisclosure(selection));
        try {
            await requestAnswer({ question, region, variants, history: priorHistory }, (event) => {
                if (event.type === "status") {
                    appendPublicChatProgress(pending, event.content);
                } else if (event.type === "stage" || event.type === "review_update" || event.type === "variant_normalization" || event.type === 'analysis_status') {
                    if (event.type === 'review_update') reviewUpdate = publicReviewUpdate(event) || reviewUpdate;
                    appendPublicChatStageEvent(pending, event);
                } else if (event.type === 'ranking_evidence') {
                    if (!assistant) assistant = createAssistantMessage();
                    renderRankingEvidence(assistant.content, [...(assistant.content.rankingEvidence || []), event]);
                } else if (event.type === 'gene_evidence_cards') {
                    if (!assistant) assistant = createAssistantMessage();
                    assistant.content.reportVariantNormalizations = pending.variantNormalizations;
                    renderGeneEvidenceCards(assistant.content, event.cards, event.variant_labels);
                } else if (event.type === "answer_chunk") {
                    const chunk = String(event.content || "");
                    appendChatProgressLine(
                        pending,
                        "Generating the evidence-grounded response...",
                        "response",
                        { summary: "Generating response" }
                    );
                    if (!assistant) assistant = createAssistantMessage();
                    content += chunk;
                    recordResponseOutput(pending, chunk);
                    renderMarkdown(assistant.content, content, true, reviewUpdate);
                } else if (event.type === "complete") {
                    if (!content.trim()) throw new Error("No evidence summary was returned");
                    if (!assistant) assistant = createAssistantMessage();
                    renderMarkdown(assistant.content, content, false, reviewUpdate);
                    completeChatProgress(pending);
                } else if (event.type === "error") {
                    throw new Error(event.content || "Analysis failed");
                }
            }, request);
            if (!isCurrentChatRequest(request)) throw abortError();
            if (!content.trim()) throw new Error("No evidence summary was returned");
            if (!assistant) assistant = createAssistantMessage();
            renderMarkdown(assistant.content, content, false, reviewUpdate);
            completeChatProgress(pending);
            addToChatHistory("assistant", content);
        } catch (error) {
            if (isAbortError(error) || !isCurrentChatRequest(request)) {
                pending.root.remove();
                if (assistant?.message?.isConnected) assistant.message.remove();
                return;
            }
            failChatProgress(pending);
            if (!assistant) assistant = createAssistantMessage();
            assistant.content.textContent = `❌ ${error.message}`;
            renderRankingEvidence(assistant.content);
            addToChatHistory('assistant', `❌ ${error.message}`);
        }
    }

    async function sendMessage() {
        const input = document.getElementById("chat-input");
        const question = input.value.trim();
        if (!question) return;
        const request = beginChatRequest();
        try {
            const bridgeReady = await checkMcpData();
            if (!isCurrentChatRequest(request)) return;
            if (!bridgeReady) {
                showMcpMissingPopup();
                return;
            }
            if (input.value.trim() === question) input.value = "";
            const regions = ManhattanPlot.getMultiSelectRegions();
            if (regions.length > 1) await queryMultipleRegions(question, request);
            else await querySingleMessage(question, request);
        } catch (error) {
            if (!isAbortError(error) && isCurrentChatRequest(request)) {
                addMessage("system", `Request failed: ${error.message}`);
            }
        } finally {
            finishChatRequest(request);
        }
    }

    function sendQuickPrompt() {
        const regions = ManhattanPlot.getMultiSelectRegions();
        const total = regions.reduce((sum, region) => sum + region.snps.length, 0);
        document.getElementById("chat-input").value =
            window.GWASMIND_UI_PROMPTS.describe_selected.replace("$count", String(total));
        sendMessage();
    }

    function addMessage(role, content, reviewUpdate = null, rankingEvidence = null, geneEvidenceCards = null,
        variantNormalizations = null, variantLabels = null) {
        const messages = document.getElementById("chat-messages");
        const message = document.createElement("div");
        message.className = `message ${role}`;
        const body = document.createElement("div");
        body.className = "message-content";
        if (role === "user") {
            body.className += " user-message";
            message.style.cssText = "display:flex;flex-direction:column;align-items:flex-end;margin-bottom:16px;";
            body.textContent = content;
        } else if (role === "system") {
            body.style.cssText = "background:#fff3cd;padding:10px 14px;border-radius:8px;font-size:14px;color:#856404;border-left:3px solid #ffc107;";
            body.textContent = content;
        } else {
            body.className += " assistant-message";
            body.reportVariantNormalizations = publicVariantNormalizations(variantNormalizations);
            body.variantLabels = window.GWASMindVariants?.cleanLabels(variantLabels) || [];
            if (rankingEvidence) renderRankingEvidence(body, rankingEvidence);
            if (geneEvidenceCards) renderGeneEvidenceCards(body, geneEvidenceCards);
            renderMarkdown(body, content, false, reviewUpdate);
        }
        message.appendChild(body);
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    }

    function appendWelcome(messages) {
        const template = document.getElementById("chat-welcome-template");
        if (messages && template?.content) {
            messages.appendChild(template.content.cloneNode(true));
        }
    }

    if (typeof document.getElementById === "function") {
        appendWelcome(document.getElementById("chat-messages"));
    }

    function clearChat() {
        abortActiveChatRequest();
        const messages = document.getElementById("chat-messages");
        messages.replaceChildren();
        appendWelcome(messages);
        chatHistory = [];
    }

    function changeFontSize(delta) {
        chatFontSize = Math.max(10, Math.min(24, chatFontSize + Number(delta || 0)));
        document.getElementById("chat-messages").style.fontSize = `${chatFontSize}px`;
    }

    function exportChatHistory() {
        const messages = document.getElementById("chat-messages");
        const items = messages.querySelectorAll(
            ".message, .region-response-card, .welcome-message"
        );
        if (!items.length) {
            addMessage("system", "No chat history is available to export.");
            return;
        }
        const now = new Date();
        const lines = [
            "# GWASMind Chat History",
            "",
            `**Date**: ${now.toLocaleDateString()}`,
            `**Time**: ${now.toLocaleTimeString()}`,
            "",
            "---",
            ""
        ];
        items.forEach((item) => {
            if (item.classList.contains("region-response-card")) {
                const badge = item.querySelector(".region-badge")?.textContent || "Region";
                const range = item.querySelector(".region-range")?.textContent || "";
                lines.push(`## ${badge}`, range ? `**Range**: ${range}` : "");
            } else if (item.classList.contains("user")) {
                lines.push("## User");
            } else if (item.classList.contains("assistant")) {
                lines.push("## Assistant");
            } else if (item.classList.contains("system")) {
                lines.push("## System");
            } else {
                lines.push("## Welcome");
            }
            const content = item.querySelector(".card-content, .message-content");
            const raw = (content?.dataset?.reportMarkdown && (window.reportExportSource?.(content) || content.dataset.reportMarkdown))
                || (content || item).textContent.trim().replace(/\s+/g, " ");
            const text = window.GWASMindVariants?.markdown(raw, content?.variantLabels) || raw;
            if (text) lines.push("", text);
            lines.push("", "---", "");
        });
        const blob = new Blob([lines.join("\n")], {
            type: "text/markdown;charset=utf-8"
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `GWASMind_Chat_${now.toISOString().replace(/[:.]/g, "-")}.md`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    window.sendMessage = sendMessage;
    window.sendQuickPrompt = sendQuickPrompt;
    window.addMessage = addMessage;
    window.clearChat = clearChat;
    window.changeFontSize = changeFontSize;
    window.exportChatHistory = exportChatHistory;
    window.highlightCardFromLeft = highlightCardFromLeft;
    window.highlightFromCard = highlightFromCard;
    window.cancelActiveChatRequest = abortActiveChatRequest;
    window.GWASMindProgress = { create: createTransientChatStatus, update: appendPublicChatStageEvent,
        complete: completeChatProgress, fail: failChatProgress, dispose: disposeProgress,
        publicRetrievalProgress, publicReviewUpdate, publicVariantNormalizations, retrievalStepKey, reportWarningCategories, stageTitle, createReportReview, publicModelUsage,
        publicFactIssues, factIssueSummary, factIssueTitle, factIssueHandling,
        publicOperationTimings, operationTimings,
        publicRankingEvidence, renderRankingEvidence, rankingDisplaySummary,
        publicGeneEvidenceCards, renderGeneEvidenceCards, openGeneEvidenceCardsModal, publicQtlRows, createQtlDisplayRow,
        renderVariantNormalizations: (destination, value) => {
            const normalizations = publicVariantNormalizations(value);
            if (!destination || !normalizations?.length) return false;
            destination.reportVariantNormalizations = normalizations;
            window.decorateQtlEvidence?.(destination);
            destination.querySelector?.(':scope > .variant-normalization-notice')?.remove();
            return updateVariantNormalizations({ root: destination }, { normalizations });
        },
        academicWarningLabel: code => ACADEMIC_WARNING_LABELS[code] || null };
    window.restoreWorkspaceChat = function (turns) {
        clearChat();
        for (const turn of turns) {
            if (!['user', 'assistant', 'system'].includes(turn.role)) continue;
            if (turn.role === 'assistant' && ((Array.isArray(turn.progress) && turn.progress.length)
                    || publicVariantNormalizations(turn.variant_normalizations)?.length)) {
                const savedProgress = createTransientChatStatus('Saved analysis workflow.', null, { historical: true });
                updateRetainedRankingProof(savedProgress, turn);
                for (const stage of (Array.isArray(turn.progress) ? turn.progress : []).slice(0, 4)) {
                    if (!CHAT_PROGRESS_STAGE_IDS.has(stage.stage)) continue;
                    const details = Array.isArray(stage.details) ? stage.details.slice(0, 24) : [];
                    for (const detail of details) appendPublicChatStageEvent(savedProgress, { ...stage, progress: undefined, detail });
                    for (const step of (Array.isArray(stage.substeps) ? stage.substeps : [])) {
                        appendPublicChatStageEvent(savedProgress, { ...stage, detail: '', progress: step });
                    }
                    appendPublicChatStageEvent(savedProgress, { ...stage, detail: '' });
                    if (stage.review_update) appendPublicChatStageEvent(savedProgress, { ...stage.review_update, type: 'review_update' });
                }
                appendPublicChatStageEvent(savedProgress, { type: 'variant_normalization', normalizations: turn.variant_normalizations });
                for (const stage of savedProgress.stages.values()) compactHistoricalDetails(stage);
                completeChatProgress(savedProgress);
            }
            const reviewUpdate = turn.role === 'assistant' && Array.isArray(turn.progress)
                ? turn.progress.find(stage => stage.stage === 'safety')?.review_update : null;
            addMessage(turn.role, turn.content, reviewUpdate, turn.role === 'assistant' ? turn.ranking_evidence : null,
                turn.role === 'assistant' ? turn.gene_evidence_cards : null,
                turn.role === 'assistant' ? turn.variant_normalizations : null,
                turn.role === 'assistant' ? turn.variant_labels : null);
        }
        // Persistence retains every turn; only the model context is bounded.
        chatHistory = MAX_CHAT_HISTORY === 0 ? [] : turns.filter(turn => ['user', 'assistant'].includes(turn.role))
            .slice(-MAX_CHAT_HISTORY).map(turn => ({ role: turn.role, content: turn.content.slice(0, MAX_CHAT_HISTORY_ITEM_CHARS) }));
    };

    function compactHistoricalDetails(stage) {
        // Headers carry status; expanded History shows actual operations and
        // review facts. Do not repeat a summary as if it were a saved log.
        for (const row of [...stage.details.children]) {
            if (row.className !== 'progress-stage-detail') continue;
            const text = row.textContent;
            if (text === stage.summary.textContent || /^(?:Request understood(?: ·|$)|Retrieving evidence(?:\.{3}| ·|$)|Evidence ready(?: ·|$)|Evidence retrieved · coverage incomplete|Response ready$|Evidence review ·|Scientific limitations were recorded\.)/.test(text)) {
                row.remove();
            }
        }
        if (!stage.details.children.length && stage.state !== 'pending') {
            const note = document.createElement('p'); note.className = 'progress-history-missing';
            note.textContent = 'Detailed stage history was not saved.';
            stage.details.appendChild(note);
        }
    }
}());

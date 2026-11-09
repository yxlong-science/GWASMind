(function () {
    'use strict';

    const MAX_LOCUS_ALERT_LINES = 20;
    const states = new Set(['queued', 'running', 'completed', 'failed', 'cancelled']);
    const terminal = new Set(['completed', 'failed', 'cancelled']);
    const workflows = { screening: 'Prior-report screening', prioritization: 'Gene prioritization',
        exploratory_prioritization: 'Gene evidence assessment · novelty unresolved' };
    const stages = { plan: 'Define Analysis Scope', retrieve: 'Retrieve Genetic Evidence', safety: 'Safety & Sufficiency Review', response: 'Interpret & Verify Findings' };
    const stateLabels = { queued: 'Waiting', running: 'Running', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled' };
    const number = value => Number.isSafeInteger(value) && value >= 0 && value <= 10_000_000 ? value : null;
    const isScreeningLog = value => /^Screening locus .+ for prior .+ GWAS reports\.$/.test(String(value || ''));
    const measuredSeconds = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
    const now = () => typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now();
    function elapsedTime(clock) {
        if (clock.seconds === null) return null;
        return clock.seconds + (clock.running && clock.anchor !== null ? Math.max(0, now() - clock.anchor) / 1000 : 0);
    }
    function formatElapsed(value) {
        if (measuredSeconds(value) === null) return '\u2014';
        const seconds = Math.floor(value), minutes = Math.floor(seconds / 60), hours = Math.floor(minutes / 60);
        return hours ? `${hours}h ${String(minutes % 60).padStart(2, '0')}m ${String(seconds % 60).padStart(2, '0')}s`
            : minutes ? `${minutes}m ${String(seconds % 60).padStart(2, '0')}s` : `${seconds}s`;
    }
    function newClock() { return { seconds: null, anchor: null, running: false, observedAt: null, finishedAt: null }; }
    const node = (tag, className, text) => {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (text !== undefined) element.textContent = text;
        return element;
    };

    function screeningResult(locus) {
        if (locus?.analysis_mode !== 'prior_report_screening') return null;
        const status = locus.screening_status;
        const classification = status === 'PREVIOUSLY_REPORTED' ? 'known'
            : status === 'NOVEL_IN_SCOPE' ? 'novel' : 'unresolved';
        const labels = { known: 'Known', novel: 'Novel (in scope)', unresolved: 'Unresolved' };
        const fallback = {
            known: 'A prior association with the queried phenotype was found.',
            novel: 'No prior association was found within the completed screening scope.',
            unresolved: 'The available evidence does not establish whether this locus is known or novel.'
        };
        const reason = String(locus.screening_reason || locus.novelty_notice || fallback[classification]);
        const timeoutIssue = 'The configured total LD screening time budget was reached.';
        const timedOut = classification === 'unresolved' && (
            (Array.isArray(locus.screening?.issues) && locus.screening.issues.includes(timeoutIssue))
            || reason.includes(timeoutIssue));
        const summary = timedOut ? 'Screening timed out; Known / Novel status remains unresolved.'
            : reason.length > 220 ? fallback[classification] : reason;
        return { classification, label: labels[classification], reason, summary };
    }

    function screeningCounts(results = []) {
        const counts = { known: 0, novel: 0, unresolved: 0 };
        for (const locus of results) {
            const result = screeningResult(locus);
            if (result) counts[result.classification] += 1;
        }
        return counts;
    }

    function screeningReportMarkdown(locus) {
        const original = String(locus?.report_markdown || '');
        const result = screeningResult(locus);
        if (!result) return original;
        const reason = String(locus.screening_reason || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/[\\`*_{}\[\]]/g, '\\$&').replace(/[\r\n]+/g, ' ');
        return `## Prior-association screening: ${result.label}`
            + (reason ? `\n\n${reason}` : '')
            + (original ? `\n\n${original}` : '');
    }

    function renderScreeningConclusion(container, locus) {
        if (!container) return;
        const result = screeningResult(locus);
        let conclusion = container.querySelector('.auto-screening-conclusion');
        if (!result) { conclusion?.remove(); return; }
        if (!conclusion) {
            conclusion = node('section', 'auto-screening-conclusion');
            conclusion.append(node('h4', 'screening-classification'), node('p', 'screening-reason'));
            container.prepend(conclusion);
        }
        conclusion.querySelector('.screening-classification').textContent = `Known / Novel screening: ${result.label}`;
        conclusion.querySelector('.screening-reason').textContent = result.summary;
        let scope = conclusion.querySelector('.screening-full-scope');
        if (result.summary !== result.reason) {
            if (!scope) {
                scope = node('details', 'screening-full-scope');
                scope.append(node('summary', '', 'Full screening scope and reason'), node('p', 'screening-scope-text'));
                conclusion.append(scope);
            }
            scope.querySelector('.screening-scope-text').textContent = result.reason;
        } else scope?.remove();
    }

    function preserveView(container, update) {
        if (!container?.getBoundingClientRect) return update();
        const scroller = container.closest('.analysis-upper') || document.scrollingElement;
        if (!scroller) return update();
        const viewport = scroller === document.scrollingElement
            ? { top: 0, bottom: window.innerHeight } : scroller.getBoundingClientRect();
        const visible = element => {
            const bounds = element.getBoundingClientRect();
            return bounds.bottom > viewport.top && bounds.top < viewport.bottom;
        };
        const selected = container.querySelector('.result-card.selected');
        const anchor = selected && visible(selected) ? selected
            : [...container.querySelectorAll('.auto-locus-card')].find(visible);
        const top = anchor?.getBoundingClientRect().top;
        const result = update();
        if (anchor?.isConnected) scroller.scrollTop += anchor.getBoundingClientRect().top - top;
        return result;
    }

    function create(container, summary, options = {}) {
        const rows = new Map();
        const scopeHost = options.scopeContainer;
        const totalTime = options.totalTime;
        const batchClock = newClock();
        let historical = false;
        let disposed = false;
        let clockTimer = null;
        let scopeContext = {};
        let sharedScope = null;
        function updateClock(clock, timing, running) {
            if (timing && typeof timing === 'object') {
                const observedAt = typeof timing.observed_at === 'string' ? Date.parse(timing.observed_at) : NaN;
                if (clock.finishedAt && !timing.finished_at) return;
                const seconds = measuredSeconds(timing.elapsed_seconds);
                if (seconds !== null) {
                    clock.seconds = running && clock.running ? Math.max(elapsedTime(clock), seconds) : seconds;
                    if (Number.isFinite(observedAt)) clock.observedAt = observedAt;
                    clock.finishedAt = timing.finished_at || clock.finishedAt;
                    clock.running = running && !historical && !disposed && !timing.finished_at;
                    clock.anchor = clock.running ? now() : null;
                    return;
                }
            }
            if (!running || historical || disposed) {
                clock.seconds = elapsedTime(clock); clock.running = false; clock.anchor = null;
            }
        }
        function renderClocks() {
            if (totalTime) {
                const value = elapsedTime(batchClock);
                totalTime.textContent = `Total ${formatElapsed(value)}`;
                totalTime.title = value === null ? 'Total elapsed time was not recorded.' : 'Total elapsed time';
                totalTime.dataset.elapsedSeconds = value === null ? '' : String(Math.floor(value));
            }
            for (const row of rows.values()) {
                const value = elapsedTime(row.clock);
                row.duration.textContent = formatElapsed(value);
                row.duration.title = value === null ? (row.state === 'queued' ? 'Waiting to start' : 'Elapsed time was not recorded.') : 'Locus elapsed time';
                row.duration.dataset.elapsedSeconds = value === null ? '' : String(Math.floor(value));
            }
        }
        function scheduleClock() {
            renderClocks();
            if (historical || disposed || (!batchClock.running && ![...rows.values()].some(row => row.clock.running))) {
                if (clockTimer !== null) clearTimeout(clockTimer);
                clockTimer = null;
                return;
            }
            if (clockTimer !== null) return;
            clockTimer = setTimeout(() => { clockTimer = null; scheduleClock(); }, 1000);
        }
        function updateBatchTiming(event) {
            const active = event.state ? ['running', 'cancelling'].includes(event.state)
                : ['complete', 'error'].includes(event.type) ? false
                : event.timing ? Boolean(event.timing.started_at && !event.timing.finished_at) : batchClock.running;
            updateClock(batchClock, event.timing, active);
            scheduleClock();
        }
        function updateRowTiming(row, timing) {
            if (row.state === 'queued') Object.assign(row.clock, newClock());
            else updateClock(row.clock, timing, row.state === 'running');
        }
        function rememberScope(value = {}) {
            for (const key of ['analysis_mode', 'source_phenotype', 'target_phenotype', 'phenotype']) {
                if (typeof value[key] === 'string' && value[key].trim()) scopeContext[key] = value[key].trim();
            }
            for (const [source, target] of [['snps', 'snp_count'], ['snp_count', 'snp_count'], ['locus_count', 'locus_count']]) {
                if (number(value[source]) !== null) scopeContext[target] = value[source];
            }
            if (number(value.loci) !== null) scopeContext.locus_count = value.loci;
            if (Array.isArray(value.queries)) scopeContext.queries = value.queries;
        }
        const common = values => values.length > 0 && values.every(value => value !== undefined && value !== null && value !== '' && value === values[0]) ? values[0] : null;
        function renderSharedScope() {
            if (!scopeHost) return;
            scopeHost.hidden = !rows.size;
            if (!rows.size) { scopeHost.replaceChildren(); sharedScope = null; return; }
            if (!sharedScope) {
                const root = node('section', 'auto-shared-scope');
                const heading = node('strong', 'auto-shared-scope-heading', 'Analysis scope');
                const content = node('div', 'auto-shared-scope-content');
                root.append(heading, content); scopeHost.append(root);
                sharedScope = { root, content };
            }
            const allRows = [...rows.values()];
            const queries = scopeContext.queries || [];
            const results = allRows.map(row => row.resultData);
            const screens = results.map(result => result?.screening || {});
            const recordedSources = results.map(result => result?.source_phenotype);
            const hasRecordedSource = recordedSources.some(source => typeof source === 'string' && source.trim());
            const sourcePhenotype = hasRecordedSource ? common(recordedSources) : scopeContext.source_phenotype;
            const queryPhenotype = scopeContext.target_phenotype || common(queries.map(query => query.trait))
                || common(results.map(result => result?.query_phenotype)) || scopeContext.phenotype || sourcePhenotype;
            const build = common(queries.map(query => query.genome_build)) || common(screens.map(screen => screen.genome_build));
            const queryCount = queries.length && queries.every(query => Array.isArray(query.variants))
                ? queries.reduce((sum, query) => sum + query.variants.length, 0) : null;
            const snps = scopeContext.snp_count ?? queryCount;
            const loci = scopeContext.locus_count || queries.length || rows.size;
            const ranking = allRows.some(row => [...row.workflows.keys()].some(name => name !== 'screening'))
                || results.some(result => result && result.analysis_mode !== 'prior_report_screening' && result.status === 'Prioritized');
            const screening = scopeContext.analysis_mode === 'prior_report_screening'
                || allRows.some(row => row.scopePlan?.requested_task === 'describe_variants')
                || results.some(result => result?.analysis_mode === 'prior_report_screening');
            const lines = [['Task', ranking ? 'Prior-report screening and gene prioritization' : screening ? 'Known / Novel screening' : 'Locus analysis']];
            if (queryPhenotype) lines.push(['Phenotype', queryPhenotype]);
            if (sourcePhenotype && queryPhenotype && sourcePhenotype !== queryPhenotype) {
                lines.push(['Source GWAS', sourcePhenotype]);
            }
            lines.push(['Selection', (snps === null || snps === undefined ? '' : `${snps} ${snps === 1 ? 'SNP' : 'SNPs'} across `)
                + `${loci} ${loci === 1 ? 'locus' : 'loci'}` + (build ? ` · ${build}` : '')]);
            const population = common(screens.map(screen => screen.population));
            const threshold = common(screens.map(screen => screen.r2_threshold));
            const windowBp = common(screens.map(screen => screen.window_bp));
            if (typeof population === 'string' && measuredSeconds(threshold) !== null && threshold <= 1
                    && number(windowBp) !== null && windowBp > 0) {
                lines.push(['LD scope', `${population} · r² ≥ ${threshold} · ±${windowBp.toLocaleString('en-US')} bp`]);
            }
            sharedScope.content.replaceChildren(...lines.map(([label, text]) => {
                const line = node('p', 'auto-shared-scope-line');
                line.append(node('span', 'auto-shared-scope-label', label), node('span', 'auto-shared-scope-value', text));
                return line;
            }));
        }
        let statsSeen = false;
        let counts = { running: 0, waiting: 0, completed: 0, failed: 0, cancelled: 0 };
        function renderCounts() {
            if (!summary) return;
            summary.textContent = `${counts.running} running · ${counts.waiting} waiting · ${counts.completed} completed`
                + (counts.failed ? ` (${counts.failed} failed)` : '')
                + (counts.cancelled ? ` · ${counts.cancelled} cancelled` : '');
        }
        function countRows() {
            counts = { running: 0, waiting: 0, completed: 0, failed: 0, cancelled: 0 };
            for (const row of rows.values()) {
                if (row.state === 'queued') counts.waiting += 1;
                else if (row.state === 'failed') { counts.failed += 1; counts.completed += 1; }
                else counts[row.state] += 1;
            }
            renderCounts();
        }
        function ensureRow(item) {
            const index = number(item.locus_index);
            if (index === null) return null;
            let row = rows.get(index);
            if (row) return row;
            const locusNo = number(item.locus_no) || index + 1;
            const root = node('details', 'auto-locus-card');
            root.open = false;
            root.dataset.locusIndex = String(index); root.dataset.locusNo = String(locusNo);
            const header = node('summary', 'auto-locus-heading');
            const title = node('strong', 'auto-locus-number', `Locus ${locusNo}`);
            const badge = node('span', 'auto-locus-state', 'Waiting');
            const identity = node('span', 'auto-locus-id', String(item.lead_rsid || item.locus_id || ''));
            identity.title = identity.textContent;
            const current = node('span', 'auto-locus-current', 'Waiting');
            const duration = node('span', 'auto-locus-time', '\u2014');
            duration.title = 'Waiting to start';
            header.append(title, identity, current, badge, duration);
            header.addEventListener('click', () => options.onSelect?.(locusNo));
            const expanded = node('div', 'auto-locus-expanded');
            const details = node('details', 'auto-locus-progress');
            details.open = true;
            details.hidden = true;
            details.append(node('summary', '', 'Workflow progress'));
            const progressArea = node('div', 'auto-locus-workflows');
            details.append(progressArea);
            const result = node('div', 'auto-locus-result');
            expanded.append(result, details); root.append(header, expanded);
            const next = [...rows.values()].filter(value => value.index > index).sort((a, b) => a.index - b.index)[0];
            if (next) container.insertBefore(root, next.root); else container.append(root);
            row = { index, locusNo, root, header, identity, badge, current, duration, clock: newClock(), details, expanded, progressArea, result,
                alertNodes: [], workflows: new Map(), state: 'queued', workflow: '', resultData: null, scopePlan: null };
            rows.set(index, row);
            return row;
        }
        function renderRowLabel(row) {
            const screening = screeningResult(row.resultData);
            row.badge.textContent = screening?.label || row.resultData?.status || stateLabels[row.state];
            row.badge.dataset.classification = screening?.classification || '';
            row.badge.hidden = !row.resultData && !terminal.has(row.state);
            row.current.hidden = Boolean(row.resultData) || terminal.has(row.state);
        }
        function setState(row, state) {
            if (!states.has(state) || (terminal.has(row.state) && !terminal.has(state))) return;
            row.state = state;
            row.root.dataset.state = state;
            renderRowLabel(row);
            if (state === 'running' && !row.workflows.size) row.current.textContent = 'Preparing';
            if (terminal.has(state)) {
                updateClock(row.clock, null, false);
                for (const progress of row.workflows.values()) {
                    if (state === 'failed' && [...progress.stages.values()].some(stage => stage.state === 'active')) {
                        window.GWASMindProgress?.fail(progress);
                    } else window.GWASMindProgress?.dispose(progress);
                }
                row.current.textContent = stateLabels[state];
            }
        }
        function ensureWorkflow(row, workflow, historical = false, restoring = false) {
            let progress = row.workflows.get(workflow);
            if (!progress && window.GWASMindProgress) {
                row.details.hidden = false;
                const section = node('div', 'auto-workflow-section');
                if (workflow !== 'screening') section.append(node('h4', '', workflows[workflow]));
                row.progressArea.append(section);
                progress = window.GWASMindProgress.create(restoring ? '' : 'Preparing the analysis...', section, {
                    fixedWorkflow: true, historical, restoring,
                    requestedTask: workflow === 'screening' ? 'describe_variants' : 'prioritize_genes' });
                row.workflows.set(workflow, progress);
                if (scopeHost && progress.stages.get('plan')) progress.stages.get('plan').section.hidden = true;
            }
            return progress;
        }
        const board = {
            get hasLoci() { return rows.size > 0; },
            get rows() { return rows; },
            reset(scope = {}) {
                board.dispose(); rows.clear(); container.replaceChildren();
                historical = false; disposed = false; Object.assign(batchClock, newClock());
                scopeContext = {}; rememberScope(scope); renderClocks();
                renderSharedScope();
                statsSeen = false; counts = { running: 0, waiting: 0, completed: 0, failed: 0, cancelled: 0 }; renderCounts();
            },
            dispose() {
                disposed = true;
                if (clockTimer !== null) clearTimeout(clockTimer);
                clockTimer = null;
                updateClock(batchClock, null, false);
                for (const row of rows.values()) {
                    updateClock(row.clock, null, false);
                    for (const progress of row.workflows.values()) window.GWASMindProgress?.dispose(progress);
                }
                renderClocks();
            },
            restore(snapshot = {}, options = {}) {
                const liveSnapshot = options.historical === false || (options.historical !== true
                    && ['running', 'cancelling'].includes(snapshot.state));
                board.reset(snapshot);
                historical = !liveSnapshot;
                board.handle({ type: 'loci', state: snapshot.state || snapshot.status,
                    timing: snapshot.timing, data: Array.isArray(snapshot.loci) ? snapshot.loci : [] });
                // A saved History view is explicitly historical even if it was
                // captured during a run. Current server snapshots can resume
                // live updates; terminal loci still retain their frozen stages.
                for (const saved of (Array.isArray(snapshot.progress) ? snapshot.progress : [])) {
                    if (!saved || !Object.hasOwn(workflows, saved.workflow)) continue;
                    const stagesSaved = (Array.isArray(saved.stages) ? saved.stages : [])
                        .filter(stage => stage && Object.hasOwn(stages, stage.stage)).slice(0, 4);
                    const normalizations = window.GWASMindProgress?.publicVariantNormalizations?.(saved.variant_normalizations);
                    if (!stagesSaved.length && !normalizations?.length) continue;
                    const row = ensureRow(saved);
                    if (!row) continue;
                    const historical = !liveSnapshot || terminal.has(row.state);
                    const progress = ensureWorkflow(row, saved.workflow, historical, true);
                    if (!progress) continue;
                    for (const stage of stagesSaved) {
                        if (stage.stage === 'plan' && (saved.workflow === 'screening' || !row.scopePlan)) row.scopePlan = stage;
                        const event = { ...stage, type: 'stage' };
                        for (const detail of (Array.isArray(stage.details) ? stage.details : []).slice(-24)) {
                            window.GWASMindProgress.update(progress, { ...event, progress: undefined, detail });
                        }
                        for (const step of (Array.isArray(stage.substeps) ? stage.substeps : [])) {
                            window.GWASMindProgress.update(progress, { ...event, detail: '', progress: step });
                        }
                        window.GWASMindProgress.update(progress, { ...event, detail: '' });
                        if (stage.review_update) window.GWASMindProgress.update(progress,
                            { ...stage.review_update, type: 'review_update' });
                    }
                    if (normalizations?.length) window.GWASMindProgress.update(progress,
                        { type: 'variant_normalization', normalizations });
                    if (historical) window.GWASMindProgress.complete(progress);
                    else {
                        progress.restoring = false;
                        row.workflow = saved.workflow;
                        const active = stagesSaved.findLast(stage => stage.state === 'running');
                        if (active) {
                            const task = progress.requestedTask || (saved.workflow === 'screening' ? 'describe_variants' : 'prioritize_genes');
                            const title = window.GWASMindProgress?.stageTitle?.(active.stage, task) || stages[active.stage];
                            row.current.textContent = active.stage === 'plan' ? 'Scoping' : title;
                        }
                    }
                }
                countRows();
                renderSharedScope();
                scheduleClock();
            },
            resultContainer(locusNo) { return [...rows.values()].find(row => row.locusNo === Number(locusNo))?.result || null; },
            setResult(locus) {
                const row = [...rows.values()].find(item => item.locusNo === Number(locus?.locus_no));
                if (!row) return;
                row.resultData = locus;
                row.identity.textContent = String(locus.lead_rsid || row.identity.textContent || '');
                row.identity.title = row.identity.textContent;
                updateRowTiming(row, locus.timing);
                renderRowLabel(row);
                renderSharedScope(); scheduleClock();
            },
            select(locusNo) {
                for (const row of rows.values()) row.root.classList.toggle('selected', row.locusNo === Number(locusNo));
            },
            handle(event) {
                if (!event || typeof event !== 'object') return false;
                if (event.type === 'log' && isScreeningLog(event.message)) return true;
                if (event.type === 'loci' && Array.isArray(event.data)) {
                    rememberScope(event);
                    for (const item of event.data) {
                        const row = ensureRow(item);
                        if (row) { setState(row, item.state || 'queued'); updateRowTiming(row, item.timing); }
                    }
                    updateBatchTiming(event);
                    renderSharedScope();
                    if (!statsSeen) countRows(); return true;
                }
                if (event.type === 'stats') {
                    rememberScope(event); updateBatchTiming(event); renderSharedScope();
                    for (const key of Object.keys(counts)) if (number(event[key]) !== null) { counts[key] = event[key]; statsSeen = true; }
                    renderCounts(); return false;
                }
                if (event.type === 'locus_state') {
                    const row = ensureRow(event); if (!row) return false;
                    if (workflows[event.workflow]) row.workflow = event.workflow;
                    setState(row, event.state); updateRowTiming(row, event.timing);
                    scheduleClock(); renderSharedScope(); if (!statsSeen) countRows(); return true;
                }
                if (event.type === 'timing') {
                    for (const item of Array.isArray(event.loci) ? event.loci : []) {
                        const row = ensureRow(item);
                        if (!row) continue;
                        setState(row, item.state || row.state);
                        updateRowTiming(row, item.timing);
                    }
                    updateBatchTiming(event);
                    if (Array.isArray(event.loci)) countRows();
                    renderSharedScope();
                    return true;
                }
                if (event.type === 'complete' || event.type === 'error') {
                    updateBatchTiming(event);
                    for (const row of rows.values()) updateClock(row.clock, null, false);
                    scheduleClock(); return false;
                }
                if (['workflow_review_update', 'workflow_variant_normalization'].includes(event.type) && workflows[event.workflow]) {
                    const row = rows.get(number(event.locus_index));
                    const progress = row?.workflows.get(event.workflow);
                    if (progress) window.GWASMindProgress?.update(progress, { ...event,
                        type: event.type === 'workflow_variant_normalization' ? 'variant_normalization' : 'review_update' });
                    return true;
                }
                if (event.type === 'workflow_stage' && workflows[event.workflow] && stages[event.stage]) {
                    const row = ensureRow(event); if (!row || terminal.has(row.state)) return true;
                    setState(row, 'running'); row.workflow = event.workflow;
                    const progress = ensureWorkflow(row, event.workflow);
                    if (progress) window.GWASMindProgress.update(progress, { ...event, type: 'stage' });
                    if (event.stage === 'plan' && (event.workflow === 'screening' || !row.scopePlan)) {
                        row.scopePlan = event;
                        renderSharedScope();
                    }
                    const task = progress?.requestedTask || (event.workflow === 'screening' ? 'describe_variants' : 'prioritize_genes');
                    const title = window.GWASMindProgress?.stageTitle?.(event.stage, task) || stages[event.stage];
                    row.current.textContent = event.stage === 'plan' ? 'Scoping' : title;
                    if (!statsSeen) countRows(); return true;
                }
                if ((event.type === 'task' || event.type === 'log') && number(event.locus_index) !== null) {
                    const row = ensureRow(event); if (!row) return false;
                    if (event.type === 'log' && ['warning', 'error'].includes(event.level)) {
                        const entry = node('p', `auto-locus-alert ${event.level}`, String(event.message || ''));
                        entry.setAttribute('role', event.level === 'error' ? 'alert' : 'status');
                        row.expanded.append(entry);
                        row.alertNodes.push(entry);
                        if (row.alertNodes.length > MAX_LOCUS_ALERT_LINES) row.alertNodes.shift().remove();
                    }
                    return true;
                }
                return false;
            }
        };
        renderCounts();
        renderClocks();
        window.addEventListener('pagehide', () => board.dispose());
        return board;
    }
    window.GWASMindAutoProgress = { create, preserveView, screeningResult, screeningCounts,
        screeningReportMarkdown, renderScreeningConclusion, formatElapsed };
}());

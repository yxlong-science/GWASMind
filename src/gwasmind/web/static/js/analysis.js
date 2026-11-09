/**
 * Auto Analysis Module
 * Screen uploaded loci for prior reports of the source phenotype.
 */

let selectedFile = null;
let taskId = null;
let latestNovel = 0;
let latestKnown = 0;
let latestUnresolved = 0;
let analysisUploadController = null;
let analysisUploadGeneration = 0;
let analysisRunActive = false;
let analysisStreamConnected = false;
let activeProgressEventSource = null;
let analysisStreamGeneration = 0;
let analysisReconnectTimer = null;
let analysisSourcePhenotype = '';
let analysisLocusBoard = null;
let completedLocusResults = [];
const completedLocusEvidence = new Map();

function getAnalysisLocusBoard() {
    const area = document.getElementById('analysis-locus-progress');
    if (!analysisLocusBoard && area && window.GWASMindAutoProgress) {
        analysisLocusBoard = window.GWASMindAutoProgress.create(area, document.getElementById('analysis-batch-summary'), {
            scopeContainer: document.getElementById('analysis-shared-scope'), totalTime: document.getElementById('analysis-total-time')
        });
    }
    return analysisLocusBoard;
}

const MAX_LOCUS_FILE_BYTES = globalThis.GWASMindConfig.get('web.max_json_bytes');
const MAX_ANALYSIS_LOG_LINES = 200;
const TASK_ID_PATTERN = /^[a-f0-9]{32}$/;
const MAX_STREAM_RECONNECTS = globalThis.GWASMindConfig.get('web.stream_reconnect_attempts');
const STREAM_RECONNECT_DELAYS_MS = globalThis.GWASMindConfig.get('web.stream_reconnect_delays_ms');

function cancelPendingUpload() {
    analysisUploadGeneration += 1;
    if (analysisUploadController) analysisUploadController.abort();
    analysisUploadController = null;
}

function closeProgressStream() {
    if (analysisReconnectTimer !== null) {
        clearTimeout(analysisReconnectTimer);
        analysisReconnectTimer = null;
    }
    if (activeProgressEventSource) {
        activeProgressEventSource.close();
        activeProgressEventSource = null;
    }
    analysisStreamConnected = false;
}

function requestRunCancellation(id) {
    if (!TASK_ID_PATTERN.test(id || '')) return;
    fetch(`/api/analysis/quick-cancel/${id}`, {
        method: 'POST',
        keepalive: true
    }).catch(() => {});
}

function abandonAnalysisRun(message) {
    const runId = taskId;
    const shouldCancel = analysisRunActive && runId;
    analysisStreamGeneration += 1;
    closeProgressStream();
    analysisRunActive = false;
    analysisLocusBoard?.dispose();
    if (shouldCancel) requestRunCancellation(runId);
    if (message) {
        addLog('error', message);
        showResult(false, message);
    }
}

function resetAnalysisLifecycle() {
    cancelPendingUpload();
    abandonAnalysisRun();
    taskId = null;
    completedLocusResults = [];
    completedLocusEvidence.clear();
    analysisLocusBoard?.reset();
}

function selectAnalysisFile(file) {
    resetAnalysisLifecycle();
    selectedFile = file;
    showFileInfo(file);
}

// File-selection handler.
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectAnalysisFile(file);
    }
}

// Display the selected file.
function showFileInfo(file) {
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatFileSize(file.size);
    document.getElementById('file-info').classList.add('show');
    document.getElementById('start-btn').style.display = 'inline-block';
}

// Format a file size for display.
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Drag-and-drop support.
document.addEventListener('DOMContentLoaded', function() {
    const uploadSection = document.getElementById('upload-section');
    
    if (uploadSection) {
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) {
                selectAnalysisFile(file);
            }
        });
    }
});

// Start the analysis.
async function startAnalysis() {
    if (!selectedFile) {
        alert('Please select a file first.');
        return;
    }
    if (analysisUploadController || analysisRunActive) return;
    const sourcePhenotype = String(document.getElementById('source-phenotype')?.value || '').trim();
    if (!sourcePhenotype || sourcePhenotype.length > 500) {
        alert('Enter the source GWAS phenotype (up to 500 characters) before starting analysis.');
        document.getElementById('source-phenotype')?.focus();
        return;
    }

    const file = selectedFile;
    analysisSourcePhenotype = sourcePhenotype;
    completedLocusResults = [];
    completedLocusEvidence.clear();
    getAnalysisLocusBoard()?.reset({ analysis_mode: 'prior_report_screening',
        source_phenotype: sourcePhenotype, target_phenotype: sourcePhenotype });
    const controller = new AbortController();
    const generation = ++analysisUploadGeneration;
    analysisUploadController = controller;
    
    // Replace the upload area with progress details.
    document.getElementById('upload-section').style.display = 'none';
    document.getElementById('start-btn').style.display = 'none';
    document.getElementById('progress-section').classList.add('active');
    
    try {
        addLog('info', 'Uploading file...');

        if (file.size > MAX_LOCUS_FILE_BYTES) {
            throw new Error(`Locus file exceeds the ${formatFileSize(MAX_LOCUS_FILE_BYTES)} safety limit.`);
        }

        const text = await file.text();
        if (generation !== analysisUploadGeneration || controller.signal.aborted) return;
        const requestBody = JSON.stringify({
            filename: file.name,
            text: text
        });
        if (new TextEncoder().encode(requestBody).byteLength > MAX_LOCUS_FILE_BYTES) {
            throw new Error(`Encoded locus upload exceeds the ${formatFileSize(MAX_LOCUS_FILE_BYTES)} request limit.`);
        }
        
        const uploadResponse = await fetch('/api/analysis/quick-upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: requestBody,
            signal: controller.signal
        });

        const uploadResult = await uploadResponse.json();
        if (generation !== analysisUploadGeneration || controller.signal.aborted) return;
        if (!uploadResponse.ok) {
            throw new Error(uploadResult.error || 'Upload failed');
        }

        if (!TASK_ID_PATTERN.test(uploadResult.task_id || '')) {
            throw new Error('The server returned an invalid analysis task ID.');
        }

        taskId = uploadResult.task_id;
        latestNovel = 0;
        latestKnown = 0;
        latestUnresolved = 0;
        for (const key of ['known', 'novel', 'unresolved']) updateStats(key, 0);
        
        addLog('success', `File uploaded: ${uploadResult.snp_count} SNPs in ${uploadResult.locus_count} loci`);
        if (Array.isArray(uploadResult.warnings)) {
            uploadResult.warnings.forEach((warning) => addLog('warning', warning));
        }
        updateStats('snps', uploadResult.snp_count);
        updateStats('loci', uploadResult.locus_count);
        
        // Start the server-sent event progress stream.
        startProgressStream(taskId);
        
    } catch (error) {
        if (error.name === 'AbortError' || generation !== analysisUploadGeneration) return;
        addLog('error', `Error: ${error.message}`);
        showResult(false, error.message);
    } finally {
        if (analysisUploadController === controller) analysisUploadController = null;
    }
}

// Server-sent event progress stream.
function startProgressStream(runId) {
    if (!TASK_ID_PATTERN.test(runId)) {
        showResult(false, 'Invalid analysis task ID.');
        return;
    }

    analysisRunActive = true;
    const streamState = {
        generation: ++analysisStreamGeneration,
        taskId: runId,
        sourcePhenotype: analysisSourcePhenotype,
        lastSequence: 0,
        reconnectAttempts: 0,
        sawTerminal: false,
        sawDone: false
    };
    connectProgressStream(streamState);
}

function isCurrentProgressStream(state) {
    return analysisRunActive
        && state.generation === analysisStreamGeneration
        && state.taskId === taskId;
}

function connectProgressStream(state) {
    if (!isCurrentProgressStream(state)) return;
    closeProgressStream();

    let eventSource;
    try {
        eventSource = new EventSource(`/api/analysis/quick-stream/${state.taskId}?phenotype=${encodeURIComponent(state.sourcePhenotype)}`);
    } catch (error) {
        scheduleProgressReconnect(state);
        return;
    }
    activeProgressEventSource = eventSource;

    eventSource.onopen = function() {
        if (isCurrentProgressStream(state) && activeProgressEventSource === eventSource) {
            analysisStreamConnected = true;
        }
    };

    eventSource.onmessage = function(event) {
        if (!isCurrentProgressStream(state) || activeProgressEventSource !== eventSource) return;
        if (event.data === '[DONE]') {
            state.sawDone = true;
            closeProgressStream();
            analysisRunActive = false;
            if (!state.sawTerminal) {
                requestRunCancellation(state.taskId);
                const message = 'The analysis stream ended before a result was produced.';
                addLog('error', message);
                showResult(false, message);
            }
            return;
        }
        
        try {
            const data = JSON.parse(event.data);
            if (!data || typeof data !== 'object' || Array.isArray(data) || typeof data.type !== 'string') {
                throw new Error('Invalid analysis event.');
            }
            if (data.type !== 'timing' && Number.isSafeInteger(data.seq)) {
                if (data.seq <= state.lastSequence) return;
                state.lastSequence = data.seq;
            }
            if (data.type === 'complete' || data.type === 'error') state.sawTerminal = true;
            handleProgressEvent(data);
        } catch (e) {
            abandonAnalysisRun('The analysis stream returned an invalid event. The server run was cancelled.');
        }
    };
    
    eventSource.onerror = function(error) {
        if (!isCurrentProgressStream(state) || activeProgressEventSource !== eventSource) return;
        console.error('SSE Error:', error);
        eventSource.close();
        activeProgressEventSource = null;
        analysisStreamConnected = false;
        if (state.sawTerminal || state.sawDone) {
            analysisRunActive = false;
            return;
        }
        scheduleProgressReconnect(state);
    };
}

function scheduleProgressReconnect(state) {
    if (!isCurrentProgressStream(state)) return;
    if (state.reconnectAttempts >= MAX_STREAM_RECONNECTS) {
        abandonAnalysisRun('Connection lost before analysis completed. The server run was cancelled.');
        return;
    }
    const delay = STREAM_RECONNECT_DELAYS_MS[state.reconnectAttempts]
        ?? STREAM_RECONNECT_DELAYS_MS[STREAM_RECONNECT_DELAYS_MS.length - 1];
    state.reconnectAttempts += 1;
    analysisReconnectTimer = setTimeout(() => {
        analysisReconnectTimer = null;
        connectProgressStream(state);
    }, delay);
}

window.addEventListener('beforeunload', () => {
    cancelPendingUpload();
    if (analysisRunActive && taskId) requestRunCancellation(taskId);
    closeProgressStream();
});

window.resetAnalysisLifecycle = resetAnalysisLifecycle;

// Handle a progress event.
function handleProgressEvent(data) {
    const preserveView = window.GWASMindAutoProgress?.preserveView;
    return preserveView ? preserveView(document.getElementById('analysis-locus-progress'), () => applyProgressEvent(data))
        : applyProgressEvent(data);
}

function applyProgressEvent(data) {
    const board = getAnalysisLocusBoard();
    const handled = board?.handle(data);
    if (board?.hasLoci) document.getElementById('current-task').style.display = 'none';
    switch (data.type) {
        case 'workflow_ranking_evidence': {
            if (!Number.isSafeInteger(data.locus_index) || data.locus_index < 0
                    || !['prioritization', 'exploratory_prioritization'].includes(data.workflow)
                    || !board?.rows?.has(data.locus_index)) break;
            const descriptor = window.GWASMindProgress?.publicRankingEvidence?.(data);
            if (!descriptor) break;
            const number = data.locus_index + 1;
            const entries = new Map((completedLocusEvidence.get(number) || [])
                .map(row => [`${row.run_id}:${row.query_index}`, row]));
            entries.set(`${descriptor.run_id}:${descriptor.query_index}`, descriptor);
            completedLocusEvidence.set(number, [...entries.values()].slice(0, 256));
            window.GWASMindProgress?.renderRankingEvidence?.(board.resultContainer(number), completedLocusEvidence.get(number));
            break;
        }
        case 'loci':
            renderCompletedLoci();
            break;
        case 'locus_state':
        case 'workflow_stage':
            break;
        case 'results':
            completedLocusResults = Array.isArray(data.data) ? data.data : [];
            renderCompletedLoci();
            updateScreeningStats(window.GWASMindAutoProgress?.screeningCounts?.(completedLocusResults));
            break;
        case 'progress':
            updateProgress(data.percent);
            break;
            
        case 'task':
            if (!board?.hasLoci) updateCurrentTask(data.text);
            break;
            
        case 'log':
            if (!handled) addLog(data.level || 'info', data.message);
            break;
            
        case 'stats':
            if (data.snps !== undefined) updateStats('snps', data.snps);
            if (data.loci !== undefined) updateStats('loci', data.loci);
            updateScreeningStats(data);
            break;
            
        case 'complete':
            board?.dispose();
            showResult(true, {
                novel_count: latestNovel,
                known_count: latestKnown,
                unresolved_count: latestUnresolved
            });
            break;
            
        case 'error':
            addLog('error', String(data.message || 'Analysis failed.'));
            showResult(false, String(data.message || 'Analysis failed.'));
            break;
    }
}

function updateScreeningStats(data = {}) {
    if (data.known !== undefined) latestKnown = toNonNegativeInteger(data.known);
    if (data.novel !== undefined) latestNovel = toNonNegativeInteger(data.novel);
    if (data.unresolved !== undefined) latestUnresolved = toNonNegativeInteger(data.unresolved);
    updateStats('known', latestKnown);
    updateStats('novel', latestNovel);
    updateStats('unresolved', latestUnresolved);
}

function renderCompletedLoci() {
    for (const locus of completedLocusResults) {
        analysisLocusBoard?.setResult?.(locus);
        const host = analysisLocusBoard?.resultContainer(locus.locus_no);
        if (!host) continue;
        let summary = host.querySelector('.auto-locus-result-summary');
        if (!summary) { summary = document.createElement('p'); summary.className = 'auto-locus-result-summary'; host.append(summary); }
        const screening = window.GWASMindAutoProgress?.screeningResult?.(locus);
        summary.textContent = `${locus.lead_rsid || 'Locus ' + locus.locus_no} · ${screening?.label || locus.status || 'Result available'}`
            + (screening ? ` · ${screening.summary}` :
            (locus.top_gene ? ` · ${locus.top_gene}` : '')
            + (window.GWASMindProgress?.rankingDisplaySummary?.(locus)
                ? ` · ${window.GWASMindProgress.rankingDisplaySummary(locus)}` : ''));
        if (screening && locus.report_markdown) {
            let details = host.querySelector('.auto-screening-report');
            if (!details) {
                details = document.createElement('details');
                details.className = 'auto-screening-report';
                const title = document.createElement('summary');
                title.textContent = 'Screening report';
                const content = document.createElement('div');
                content.className = 'auto-screening-report-content markdown-content';
                details.append(title, content);
                host.append(details);
            }
            const content = details.querySelector('.auto-screening-report-content');
            const exported = window.GWASMindAutoProgress?.screeningReportMarkdown?.(locus) ?? locus.report_markdown;
            if (content.reportMarkdown !== locus.report_markdown || content.exportMarkdown !== exported) {
                content.reportMarkdown = locus.report_markdown;
                content.exportMarkdown = exported;
                if (window.GWASMindSafe?.renderMarkdown) content.innerHTML = window.GWASMindSafe.renderMarkdown(locus.report_markdown);
                else content.textContent = locus.report_markdown;
                window.attachReportActions?.(content, exported);
                window.GWASMindProgress?.renderVariantNormalizations?.(content, locus.variant_normalizations);
            }
            window.GWASMindAutoProgress?.renderScreeningConclusion?.(content, locus);
        }
        if (Array.isArray(locus.ranking_evidence)) {
            completedLocusEvidence.set(locus.locus_no, locus.ranking_evidence.slice(0, 256)
                .map(value => window.GWASMindProgress?.publicRankingEvidence?.(value)).filter(Boolean));
        }
        window.GWASMindProgress?.renderRankingEvidence?.(host, completedLocusEvidence.get(locus.locus_no) || []);
    }
}

// Update the progress bar.
function updateProgress(percent) {
    const numericPercent = Math.max(0, Math.min(100, Number(percent) || 0));
    document.getElementById('progress-bar').style.width = numericPercent + '%';
    document.getElementById('progress-percent').textContent = numericPercent + '%';
}

// Update the current task.
function updateCurrentTask(text) {
    document.getElementById('task-text').textContent = text;
}

// Append a log entry.
function addLog(level, message) {
    const logSection = document.getElementById('log-section');
    const icons = {
        'success': '[OK]',
        'error': '[X]',
        'warning': '[!]',
        'info': '[-]',
        'pending': '[ ]'
    };
    
    const safeLevel = Object.prototype.hasOwnProperty.call(icons, level) ? level : 'info';
    const entry = document.createElement('div');
    entry.className = `log-entry ${safeLevel}`;

    const icon = document.createElement('span');
    icon.className = 'icon';
    icon.textContent = icons[safeLevel];

    const text = document.createElement('span');
    text.textContent = String(message ?? '');

    entry.append(icon, text);
    logSection.appendChild(entry);
    while (logSection.childElementCount > MAX_ANALYSIS_LOG_LINES) logSection.firstElementChild.remove();
    // New log entries must not move a reader away from the line they are inspecting.
}

function toNonNegativeInteger(value) {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? Math.floor(number) : 0;
}

// Update a statistic.
function updateStats(type, value) {
    const el = document.getElementById(`stat-${type}`);
    if (el) el.textContent = value;
}

// Display the result.
function showResult(success, data) {
    analysisLocusBoard?.dispose();
    if (!analysisLocusBoard?.hasLoci) document.getElementById('progress-section').classList.remove('active');
    document.getElementById('result-section').classList.add('active');
    
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    
    if (success) {
        resultIcon.textContent = 'OK';
        resultIcon.className = 'result-icon success';
        resultTitle.textContent = 'Analysis Complete!';
        
        document.getElementById('novel-count').textContent = data.novel_count || 0;
        document.getElementById('known-count').textContent = data.known_count || 0;
        document.getElementById('unresolved-count').textContent = data.unresolved_count || 0;
        
        // Keep the legacy button appearance while using the bounded quick-download route.
        if (TASK_ID_PATTERN.test(taskId || '')) {
            const downloadButton = document.getElementById('download-btn');
            downloadButton.href = `/api/analysis/quick-download/${taskId}`;
            downloadButton.style.display = 'inline-block';
        }
    } else {
        resultIcon.textContent = 'X';
        resultIcon.className = 'result-icon error';
        resultTitle.textContent = 'Analysis Failed';
        document.getElementById('result-summary').textContent = data;
        document.getElementById('download-btn').style.display = 'none';
    }
}

// Clear the log.
function clearLogs() {
    document.getElementById('log-section').replaceChildren();
}

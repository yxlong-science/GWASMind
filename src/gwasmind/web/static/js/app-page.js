// Async JSON parser to avoid blocking main thread
async function parseJSONAsync(text, onProgress) {
    return new Promise((resolve, reject) => {
        // Use setTimeout to yield to main thread before parsing
        setTimeout(() => {
            try {
                if (onProgress) onProgress('Parsing JSON...');
                const result = JSON.parse(text);
                resolve(result);
            } catch (e) {
                reject(e);
            }
        }, 10);
    });
}

function reportAppError(context, error) {
    const detail = error instanceof Error ? error.message : String(error || 'Unknown error');
    console.error(`[GWASMind] ${context}:`, error);
    alert(`${context}: ${detail}`);
}

let loadedSourcePhenotype = '';
let loadedDatasetId = null;

function setLoadedPlotMetadata(metadata = {}) {
    const identifier = metadata.history_dataset_id;
    loadedDatasetId = typeof identifier === 'string' && /^[a-f0-9]{32}$/.test(identifier) ? identifier : null;
    loadedSourcePhenotype = String(metadata.source_phenotype || '').trim();
    const container = document.querySelector('.plot-container');
    if (!container) return;
    let notice = container.querySelector('.history-plot-notice');
    const text = typeof metadata.history_plot_notice === 'string' ? metadata.history_plot_notice.trim() : '';
    if (!text) { notice?.remove(); return; }
    if (!notice) {
        notice = document.createElement('p');
        notice.className = 'history-plot-notice';
        notice.setAttribute('role', 'status');
        const plot = container.querySelector('.plot-wrapper');
        if (plot) container.insertBefore(notice, plot);
        else container.append(notice);
    }
    notice.textContent = text;
}

// Initialize Manhattan plot when page loads
document.addEventListener('DOMContentLoaded', async () => {
    const loadingEl = document.getElementById('plot-loading');
    const noDataEl = document.getElementById('no-data-message');
    const loadingText = loadingEl ? loadingEl.querySelector('p') : null;
    
    const updateLoadingText = (text) => {
        if (loadingText) loadingText.textContent = text;
    };
    
    try {
        // Use XMLHttpRequest for accurate download progress
        updateLoadingText('Connecting...');
        
        const result = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('GET', '/api/gwas-data', true);
            
            xhr.onprogress = function(event) {
                if (event.lengthComputable) {
                    const percent = Math.min(99, Math.round((event.loaded / event.total) * 100));
                    updateLoadingText(`Downloading... ${percent}%`);
                } else {
                    // Fallback: show MB downloaded
                    const mb = (event.loaded / 1024 / 1024).toFixed(1);
                    updateLoadingText(`Downloading... ${mb}MB`);
                }
            };
            
            xhr.onload = function() {
                if (xhr.status === 200) {
                    updateLoadingText('Parsing data...');
                    try {
                        const responseData = JSON.parse(xhr.responseText);
                        
                        // Expand compact rows into objects for the existing plot contract.
                        if (responseData.format === 'array' && Array.isArray(responseData.data)) {
                            // Format: [chromosome, position, neg_log_p, snp_id, pvalue, ref, alt]
                            responseData.data = responseData.data.map(item => ({
                                chromosome: item[0],
                                position: item[1],
                                neg_log_p: item[2],
                                snp_id: item[3],
                                pvalue: item[4],
                                ref: item[5],
                                alt: item[6]
                            }));
                        }
                        setLoadedPlotMetadata(responseData.metadata || {});
                        
                        resolve(responseData);
                    } catch (e) {
                        console.error('Data processing error:', e);
                        reject(new Error('Failed to process data'));
                    }
                } else {
                    reject(new Error(`Server error: ${xhr.status}`));
                }
            };
            
            xhr.onerror = function() {
                reject(new Error('Network error'));
            };
            
            xhr.send();
        });
        
        if (result.data && result.data.length > 0) {
            // Step 4: Render plot
            updateLoadingText(`Rendering ${result.data.length.toLocaleString()} SNPs...`);
            // A background tab may throttle requestAnimationFrame indefinitely.
            // Keep the paint opportunity without making plot startup depend on it.
            await Promise.race([
                new Promise(resolve => requestAnimationFrame(resolve)),
                new Promise(resolve => setTimeout(resolve, 100))
            ]);
            await new Promise(r => setTimeout(r, 50));
            
            const rendered = await ManhattanPlot.init('manhattan-plot', result.data);
            if (!rendered) throw new Error('The Manhattan plot could not be rendered');
        } else {
            // No data: Show no data message
            if (loadingEl) loadingEl.style.display = 'none';
            if (noDataEl) noDataEl.style.display = 'flex';
        }
        window.dispatchEvent(new CustomEvent('gwasmind-dataset-ready', { detail: {
            metadata: result.metadata || {}, plotDataAvailable: Boolean(result.data?.length)
        } }));
    } catch (error) {
        console.error('Failed to load GWAS data:', error);
        if (loadingEl) {
            const icon = document.createElement('div');
            icon.style.cssText = 'font-size: 48px; margin-bottom: 10px; color: #ef4444;';
            icon.textContent = '⚠️';
            const title = document.createElement('p');
            title.textContent = 'Error loading data';
            const detail = document.createElement('p');
            detail.style.cssText = 'font-size: 12px; color: #999;';
            detail.textContent = error.message;
            loadingEl.replaceChildren(icon, title, detail);
        }
    }
});

// Header file upload (client-side processing)
let headerGwasUploadGeneration = 0;
let headerGwasUploadController = null;

function cancelHeaderGwasUpload() {
    headerGwasUploadGeneration += 1;
    if (headerGwasUploadController) headerGwasUploadController.abort();
    headerGwasUploadController = null;
    const input = document.getElementById('file-input-header');
    const loading = document.getElementById('plot-loading');
    const plot = document.getElementById('manhattan-plot');
    if (input) input.value = '';
    if (loading && typeof ManhattanPlot !== 'undefined' && ManhattanPlot._plotInitialized) {
        loading.style.display = 'none';
    }
    if (plot) plot.style.opacity = '1';
}

document.getElementById('file-input-header').addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const enteredPhenotype = window.prompt(
        loadedSourcePhenotype
            ? `Source GWAS phenotype (current: ${loadedSourcePhenotype}):`
            : 'Source GWAS phenotype:',
        ''
    );
    const sourcePhenotype = String(enteredPhenotype || '').trim().replace(/\s+/g, ' ');
    if (!sourcePhenotype) {
        e.target.value = '';
        if (enteredPhenotype !== null) {
            alert('A source GWAS phenotype is required.');
        }
        return;
    }
    if (sourcePhenotype.length > 500) {
        e.target.value = '';
        alert('The source GWAS phenotype must be 500 characters or fewer.');
        return;
    }

    cancelHeaderGwasUpload();
    const controller = new AbortController();
    const generation = ++headerGwasUploadGeneration;
    headerGwasUploadController = controller;
    
    // Show processing indicator
    const loadingEl = document.getElementById('plot-loading');
    const plotEl = document.getElementById('manhattan-plot');
    if (loadingEl) {
        loadingEl.style.display = 'flex';
        loadingEl.innerHTML = `
            <div class="loading-spinner"></div>
            <p>Processing file locally...</p>
        `;
    }
    if (plotEl) plotEl.style.opacity = '0.3';
    
    try {
        // Process file client-side
        const processedResult = await GWASProcessor.processFile(file, (percent, message) => {
            if (generation !== headerGwasUploadGeneration || controller.signal.aborted) return;
            if (loadingEl) {
                loadingEl.querySelector('p').textContent = `${percent}% - ${message}`;
            }
        }, { signal: controller.signal });
        if (generation !== headerGwasUploadGeneration || controller.signal.aborted) return;
        processedResult.metadata.source_phenotype = sourcePhenotype;
        window.cancelActiveChatRequest?.();
        cancelAnalysisChatRequest();
        await window.preserveWorkspaceHistory?.();
        if (generation !== headerGwasUploadGeneration || controller.signal.aborted) return;
        
        // Upload pre-processed data
        if (loadingEl) loadingEl.querySelector('p').textContent = 'Uploading sampled data...';
        
        const response = await fetch('/upload-processed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(processedResult),
            signal: controller.signal
        });
        
        const result = await response.json();
        if (generation !== headerGwasUploadGeneration || controller.signal.aborted) return;
        
        if (response.ok && result.success) {
            // A history URL belongs to the old dataset, not the new upload.
            window.location.href = '/app';
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        if (error.name === 'AbortError' || generation !== headerGwasUploadGeneration) return;
        reportAppError('Processing error', error);
        if (loadingEl) loadingEl.style.display = 'none';
        if (plotEl) plotEl.style.opacity = '1';
        e.target.value = '';
    } finally {
        if (headerGwasUploadController === controller) headerGwasUploadController = null;
    }
});

// Clear data
let clearDataActive = false;

async function clearData() {
    if (clearDataActive) {
        alert('Data clearing is already in progress.');
        return;
    }
    if (!confirm('Clear all loaded data?')) return;

    clearDataActive = true;
    cancelHeaderGwasUpload();
    cancelAnalysisDemo();
    cancelAnalysisUpload();
    abandonAnalysisRun();
    cancelAnalysisChatRequest();
    if (typeof window.cancelActiveChatRequest === 'function') window.cancelActiveChatRequest();
    try {
        await window.preserveWorkspaceHistory?.();
        const response = await fetch('/clear-data', { method: 'POST' });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.error || `Clear failed (HTTP ${response.status})`);
        }
        if (!result.redirect) throw new Error('The server did not provide a redirect after clearing data');
        window.location.href = result.redirect;
    } catch (error) {
        reportAppError('Unable to clear data', error);
    } finally {
        clearDataActive = false;
    }
}

// Toggle panel visibility
let cancelPendingPanelPlotResize = null;

function cssTimeToMilliseconds(value) {
    const match = String(value || '').trim().match(/^(-?\d*\.?\d+)(ms|s)$/);
    if (!match) return 0;
    const amount = Math.max(0, Number(match[1]) || 0);
    return match[2] === 's' ? amount * 1000 : amount;
}

function transitionTimeFor(element, propertyName) {
    if (typeof getComputedStyle !== 'function') return 0;
    const style = getComputedStyle(element);
    const properties = String(style.transitionProperty || '').split(',').map(value => value.trim());
    const durations = String(style.transitionDuration || '').split(',').map(cssTimeToMilliseconds);
    const delays = String(style.transitionDelay || '').split(',').map(cssTimeToMilliseconds);
    const itemCount = Math.max(properties.length, durations.length, delays.length);
    let longest = 0;
    for (let index = 0; index < itemCount; index += 1) {
        const property = properties[index % properties.length];
        if (property !== 'all' && property !== propertyName) continue;
        longest = Math.max(
            longest,
            durations[index % durations.length] + delays[index % delays.length]
        );
    }
    return longest;
}

function resizePlotAfterPanelChange() {
    const plot = document.getElementById('manhattan-plot');
    if (!plot || !plot._fullLayout) return;
    if (typeof Plotly === 'undefined' || !Plotly.Plots || typeof Plotly.Plots.resize !== 'function') return;
    try {
        const resize = Plotly.Plots.resize(plot);
        if (resize && typeof resize.catch === 'function') {
            resize.catch(error => console.warn('[GWASMind] Plot resize failed:', error));
        }
    } catch (error) {
        console.warn('[GWASMind] Plot resize failed:', error);
    }
}

function schedulePlotResize(mainContainer) {
    if (cancelPendingPanelPlotResize) cancelPendingPanelPlotResize();

    let cancelled = false;
    let queued = false;
    let firstFrame = null;
    let secondFrame = null;
    let fallbackTimer = null;
    let transitionHandler = null;

    const release = () => {
        if (firstFrame !== null) cancelAnimationFrame(firstFrame);
        if (secondFrame !== null) cancelAnimationFrame(secondFrame);
        if (fallbackTimer !== null) clearTimeout(fallbackTimer);
        if (transitionHandler) mainContainer.removeEventListener('transitionend', transitionHandler);
        firstFrame = null;
        secondFrame = null;
        fallbackTimer = null;
        transitionHandler = null;
        if (cancelPendingPanelPlotResize === cancel) cancelPendingPanelPlotResize = null;
    };
    const cancel = () => {
        cancelled = true;
        release();
    };
    const finish = () => {
        secondFrame = null;
        if (cancelled) return;
        release();
        resizePlotAfterPanelChange();
    };
    const queueAfterLayout = () => {
        if (cancelled || queued) return;
        queued = true;
        firstFrame = requestAnimationFrame(() => {
            firstFrame = null;
            secondFrame = requestAnimationFrame(finish);
        });
    };

    cancelPendingPanelPlotResize = cancel;
    const transitionMs = transitionTimeFor(mainContainer, 'margin-right');
    if (transitionMs > 0) {
        transitionHandler = event => {
            if (event.target === mainContainer && event.propertyName === 'margin-right') {
                queueAfterLayout();
            }
        };
        mainContainer.addEventListener('transitionend', transitionHandler);
        fallbackTimer = setTimeout(queueAfterLayout, transitionMs + 80);
    } else {
        queueAfterLayout();
    }
}

function toggleChat() {
    const panel = document.getElementById('right-panel');
    const mainContainer = document.getElementById('main-app');
    const btn = document.getElementById('toggle-chat-btn');
    
    if (panel.classList.contains('collapsed')) {
        panel.classList.remove('collapsed');
        mainContainer.classList.remove('panel-collapsed');
        btn.textContent = 'Hide Panel';
        btn.setAttribute('aria-expanded', 'true');
    } else {
        panel.classList.add('collapsed');
        mainContainer.classList.add('panel-collapsed');
        btn.textContent = 'Show Panel';
        btn.setAttribute('aria-expanded', 'false');
    }
    schedulePlotResize(mainContainer);
}

// Switch between tabs
let panelSwitchGeneration = 0;
async function switchTab(tabName) {
    if (!['chat', 'analysis'].includes(tabName)) return false;
    const generation = ++panelSwitchGeneration;
    const plotArea = document.querySelector('#main-app > .plot-container');
    plotArea?.classList.add('switching-plot-panel');
    try {
        if (typeof ManhattanPlot !== 'undefined' && ManhattanPlot.switchPanel) {
            const restored = await ManhattanPlot.switchPanel(tabName);
            if (!restored || generation !== panelSwitchGeneration) return false;
        }
        // Update tab buttons and content only after the matching plot is ready.
        document.querySelectorAll('.panel-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`.panel-tab[data-tab="${tabName}"]`).classList.add('active');
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`tab-${tabName}`).classList.add('active');
        return true;
    } finally {
        if (generation === panelSwitchGeneration) plotArea?.classList.remove('switching-plot-panel');
    }
}

// Toggle export menu
function toggleExportMenu() {
    const menu = document.getElementById('export-menu');
    const button = document.querySelector('[data-action="toggle-export"]');
    const expanded = menu.style.display === 'none';
    menu.style.display = expanded ? 'block' : 'none';
    if (button) button.setAttribute('aria-expanded', String(expanded));
}

// Close export menu when clicking outside
document.addEventListener('click', (e) => {
    const dropdown = document.querySelector('.export-dropdown');
    if (dropdown && !dropdown.contains(e.target)) {
        const menu = document.getElementById('export-menu');
        if (menu) menu.style.display = 'none';
        const button = document.querySelector('[data-action="toggle-export"]');
        if (button) button.setAttribute('aria-expanded', 'false');
    }
});

// ============================================================
// Auto Analysis Panel Functions (with Plot Linking)
// ============================================================
let currentAnalysisTaskId = null;
let analysisResults = [];
const autoProgressCards = new Map();
let autoLocusBoard = null;
const analysisResultCards = new Map();
const analysisRankingEvidence = new Map();
let uploadedSnpData = [];  // Store uploaded SNP positions for highlighting
let selectedLocusNo = null;
let analysisDiscussionClosed = false;
let currentAnalysisHistoryId = null;
let analysisConversations = new Map();
let activeAnalysisEventSource = null;
let analysisRunActive = false;
let analysisStreamConnected = false;
let analysisStreamGeneration = 0;
let analysisReconnectTimer = null;
let analysisUploadController = null;
let analysisUploadGeneration = 0;
let analysisDemoController = null;
let analysisDemoGeneration = 0;

const MAX_LOCUS_UPLOAD_BYTES = globalThis.GWASMindConfig.get('web.max_json_bytes');
const MAX_ANALYSIS_STREAM_RECONNECTS = globalThis.GWASMindConfig.get('web.stream_reconnect_attempts');
const ANALYSIS_STREAM_RECONNECT_DELAYS_MS = globalThis.GWASMindConfig.get('web.stream_reconnect_delays_ms');
const LOCUS_UPLOAD_LIMIT_LABEL = MAX_LOCUS_UPLOAD_BYTES >= 1024 * 1024
    ? `${Number((MAX_LOCUS_UPLOAD_BYTES / (1024 * 1024)).toFixed(1))} MB`
    : MAX_LOCUS_UPLOAD_BYTES >= 1024 ? `${Number((MAX_LOCUS_UPLOAD_BYTES / 1024).toFixed(1))} KB` : `${MAX_LOCUS_UPLOAD_BYTES} B`;
const ANALYSIS_TASK_ID_PATTERN = /^[a-f0-9]{32}$/;

function getAutoLocusBoard() {
    const area = document.getElementById('panel-locus-progress');
    if (!autoLocusBoard && area && window.GWASMindAutoProgress) {
        autoLocusBoard = window.GWASMindAutoProgress.create(area, document.getElementById('panel-batch-summary'), {
            scopeContainer: document.getElementById('panel-shared-scope'),
            totalTime: document.getElementById('panel-total-time'), onSelect: selectAnalysisLocus
        });
    }
    return autoLocusBoard;
}

function clearAnalysisResults() {
    analysisResults = []; analysisResultCards.clear(); selectedLocusNo = null;
    analysisRankingEvidence.clear();
    currentLocusContext = null; analysisChatHistory = [];
    analysisDiscussionClosed = false;
    document.getElementById('results-list').replaceChildren();
    document.getElementById('quick-suggestions').style.display = 'none';
    document.getElementById('analysis-chat-area').style.display = 'none';
}

function acceptAnalysisHistoryId(historyId) {
    if (!ANALYSIS_TASK_ID_PATTERN.test(historyId || '') || historyId === currentAnalysisHistoryId) return;
    currentAnalysisHistoryId = historyId;
    for (const [locusNo, messages] of analysisConversations) {
        if (!messages.length) continue;
        window.saveWorkspaceAnalysisConversation?.(historyId, Number(locusNo), messages)
            .catch(error => console.warn('Unable to save the analysis discussion:', error.message));
    }
}

function cancelAnalysisUpload() {
    analysisUploadGeneration += 1;
    if (analysisUploadController) analysisUploadController.abort();
    analysisUploadController = null;
}

function cancelAnalysisDemo() {
    analysisDemoGeneration += 1;
    if (analysisDemoController) analysisDemoController.abort();
    analysisDemoController = null;
    const button = document.querySelector('[data-action="load-analysis-demo"]');
    if (button) button.disabled = false;
}

function setAnalysisStartEnabled(enabled) {
    const button = document.querySelector('[data-action="start-quick-analysis"]');
    if (button) button.disabled = !enabled;
}

function closeAnalysisStream() {
    if (analysisReconnectTimer !== null) {
        clearTimeout(analysisReconnectTimer);
        analysisReconnectTimer = null;
    }
    if (activeAnalysisEventSource) {
        activeAnalysisEventSource.close();
        activeAnalysisEventSource = null;
    }
    analysisStreamConnected = false;
}

function requestAnalysisCancellation(taskId) {
    if (!ANALYSIS_TASK_ID_PATTERN.test(taskId || '')) return;
    fetch(`/api/analysis/quick-cancel/${taskId}`, {
        method: 'POST',
        keepalive: true
    }).catch(() => {});
}

function abandonAnalysisRun(message) {
    const taskToCancel = currentAnalysisTaskId;
    const shouldCancel = analysisRunActive && taskToCancel;
    analysisStreamGeneration += 1;
    closeAnalysisStream();
    analysisRunActive = false;
    for (const progress of autoProgressCards.values()) window.GWASMindProgress?.dispose(progress);
    autoProgressCards.clear();
    autoLocusBoard?.dispose();
    if (shouldCancel) requestAnalysisCancellation(taskToCancel);
    if (message) showAnalysisFailure(message);
}

// Open Auto Analysis - switch to analysis tab
function openAutoAnalysis() {
    switchTab('analysis');
    // Ensure panel is visible
    const panel = document.getElementById('right-panel');
    if (panel.classList.contains('collapsed')) {
        toggleChat();
    }
}

function resetAnalysisPanel() {
    cancelAnalysisDemo();
    cancelAnalysisUpload();
    abandonAnalysisRun();
    cancelAnalysisChatRequest();
    currentAnalysisHistoryId = null;
    analysisConversations.clear();
    document.getElementById('analysis-upload-section').style.display = 'block';
    document.getElementById('analysis-progress-section').style.display = 'none';
    document.getElementById('analysis-results-section').style.display = 'none';
    document.getElementById('analysis-file-info').style.display = 'none';
    document.getElementById('highlight-notice').style.display = 'none';
    document.getElementById('panel-log-area').replaceChildren();
    currentAnalysisTaskId = null;
    clearAnalysisResults();
    autoLocusBoard?.reset();
    uploadedSnpData = [];
    selectedLocusNo = null;
    setAnalysisStartEnabled(false);
    
    // Clear highlights on plot
    if (typeof ManhattanPlot !== 'undefined' && ManhattanPlot.clearPanelHighlights) {
        ManhattanPlot.clearPanelHighlights('analysis');
    }
}

window.addEventListener('beforeunload', () => {
    cancelHeaderGwasUpload();
    cancelAnalysisDemo();
    cancelAnalysisUpload();
    cancelAnalysisChatRequest();
    if (analysisRunActive && currentAnalysisTaskId) {
        requestAnalysisCancellation(currentAnalysisTaskId);
    }
    closeAnalysisStream();
});

// Re-analyze button: reset and go back to upload interface
function resetAnalysis() {
    // Clear file input
    document.getElementById('analysis-file-input').value = '';
    document.getElementById('target-phenotype').value = '';
    
    // Hide chat area
    document.getElementById('analysis-chat-area').style.display = 'none';
    
    // Reset panel
    resetAnalysisPanel();
}

// Load demo data
async function loadDemoData() {
    cancelAnalysisDemo();
    cancelAnalysisUpload();
    if (analysisRunActive) abandonAnalysisRun();
    currentAnalysisTaskId = null;
    uploadedSnpData = [];
    if (typeof ManhattanPlot !== 'undefined') ManhattanPlot.clearPanelHighlights?.('analysis');
    setAnalysisStartEnabled(false);
    document.getElementById('analysis-file-info').style.display = 'none';
    document.getElementById('highlight-notice').style.display = 'none';
    const controller = new AbortController();
    const generation = ++analysisDemoGeneration;
    analysisDemoController = controller;
    const demoButton = document.querySelector('[data-action="load-analysis-demo"]');
    if (demoButton) demoButton.disabled = true;
    try {
        // Fetch demo file from server
        const response = await fetch('/demo_data/chr1_all_loci.csv', { signal: controller.signal });
        if (!response.ok) throw new Error('Demo file not found');
        
        const text = await response.text();
        if (generation !== analysisDemoGeneration || controller.signal.aborted) return;
        const blob = new Blob([text], { type: 'text/csv' });
        const file = new File([blob], 'chr1_all_loci.csv', { type: 'text/csv' });
        
        // Create a fake event and call handler
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        document.getElementById('analysis-file-input').files = dataTransfer.files;
        
        // Trigger file handling
        const uploaded = await uploadAnalysisFile(file, { externalSignal: controller.signal });
        if (!uploaded || generation !== analysisDemoGeneration || controller.signal.aborted) return;
        
        // Set default phenotype if empty
        const phenotypeInput = document.getElementById('target-phenotype');
        if (!phenotypeInput.value.trim()) {
            phenotypeInput.value = 'Asthma';
        }
    } catch (error) {
        if (error.name === 'AbortError' || generation !== analysisDemoGeneration) return;
        console.error('Failed to load demo data:', error);
        reportAppError('Failed to load demo data', error);
    } finally {
        if (analysisDemoController === controller) {
            analysisDemoController = null;
            if (demoButton) demoButton.disabled = false;
        }
    }
}

// File upload handling
document.getElementById('analysis-file-input').addEventListener('change', handleAnalysisFile);

const dropZone = document.getElementById('analysis-drop-zone');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
        cancelAnalysisDemo();
        uploadAnalysisFile(file);
    }
});

function handleAnalysisFile(e) {
    const file = e.target.files[0];
    if (!file) return Promise.resolve(false);
    cancelAnalysisDemo();
    return uploadAnalysisFile(file);
}

async function uploadAnalysisFile(file, options = {}) {
    const datasetIdForUpload = loadedDatasetId;
    if (analysisUploadController) analysisUploadController.abort();
    if (analysisRunActive) abandonAnalysisRun();
    const controller = new AbortController();
    const generation = ++analysisUploadGeneration;
    analysisUploadController = controller;
    const abortFromOutside = () => controller.abort();
    options.externalSignal?.addEventListener('abort', abortFromOutside, { once: true });
    if (options.externalSignal?.aborted) controller.abort();
    currentAnalysisTaskId = null;
    uploadedSnpData = [];
    setAnalysisStartEnabled(false);
    document.getElementById('analysis-file-info').style.display = 'none';
    document.getElementById('highlight-notice').style.display = 'none';

    try {
        if (!file || !Number.isFinite(file.size) || file.size <= 0) {
            throw new Error('The locus file is empty or unreadable');
        }
        if (file.size > MAX_LOCUS_UPLOAD_BYTES) {
            throw new Error(`Locus file exceeds the ${LOCUS_UPLOAD_LIMIT_LABEL} safety limit`);
        }
        const text = await file.text();
        if (generation !== analysisUploadGeneration || controller.signal.aborted) return;

        const requestBody = JSON.stringify({ filename: file.name, text, dataset_id: datasetIdForUpload });
        if (new Blob([requestBody]).size > MAX_LOCUS_UPLOAD_BYTES) {
            throw new Error(`Encoded locus upload exceeds the ${LOCUS_UPLOAD_LIMIT_LABEL} request limit`);
        }
        const response = await fetch('/api/analysis/quick-upload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: requestBody,
            signal: controller.signal
        });
        const result = await response.json();
        if (generation !== analysisUploadGeneration || controller.signal.aborted) return;
        if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
        if (!ANALYSIS_TASK_ID_PATTERN.test(result.task_id || '')) {
            throw new Error('The server returned an invalid analysis task ID');
        }

        currentAnalysisTaskId = result.task_id;
        document.getElementById('analysis-file-name').textContent = file.name;
        const statsElement = document.getElementById('analysis-file-stats');
        statsElement.textContent = result.original_snp_count > result.snp_count
            ? `${result.snp_count} analyzed / ${result.original_snp_count} input SNPs | ${result.locus_count} Loci`
            : `${result.snp_count} SNPs | ${result.locus_count} Loci`;
        if (Array.isArray(result.warnings) && result.warnings.length) {
            statsElement.title = result.warnings.join('\n');
        } else {
            statsElement.removeAttribute('title');
        }
        document.getElementById('analysis-file-info').style.display = 'flex';
        
        // ★ Highlight uploaded SNPs on Manhattan plot (GREEN)
        if (result.snp_positions && result.snp_positions.length > 0) {
            uploadedSnpData = result.snp_positions;
            highlightUploadedSnps(result.snp_positions);
            
            // Show notice
            document.getElementById('highlight-count').textContent = result.snp_positions.length;
            document.getElementById('highlight-notice').style.display = 'flex';
        }
        setAnalysisStartEnabled(true);
        return true;
    } catch (error) {
        if (error.name === 'AbortError' || generation !== analysisUploadGeneration) return false;
        currentAnalysisTaskId = null;
        setAnalysisStartEnabled(false);
        document.getElementById('analysis-file-input').value = '';
        reportAppError('Upload error', error);
        return false;
    } finally {
        options.externalSignal?.removeEventListener('abort', abortFromOutside);
        if (analysisUploadController === controller) analysisUploadController = null;
    }
}

// ★ Highlight SNPs on Manhattan plot
function highlightUploadedSnps(snpPositions) {
    if (typeof ManhattanPlot !== 'undefined' && ManhattanPlot.setPanelHighlights) {
        return ManhattanPlot.setPanelHighlights('analysis', snpPositions, 'uploaded');
    }
}

// ★ Highlight a specific locus (Yellow)
function highlightLocus(locusNo) {
    if (!analysisResults.length) return;
    
    const locus = analysisResults.find(l => l.locus_no === locusNo);
    if (!locus) return;
    
    if ((locus.snp_positions || locus.selected_variants?.length) && typeof ManhattanPlot !== 'undefined' && ManhattanPlot.setPanelHighlights) {
        // First restore all to green
        ManhattanPlot.setPanelHighlights('analysis', uploadedSnpData, 'uploaded');
        // Then highlight selected locus in yellow
        ManhattanPlot.setPanelHighlights('analysis', locus.selected_variants?.length ? locus.selected_variants : locus.snp_positions, 'selected');
        // Keep the user's current viewport instead of zooming automatically.
        // ManhattanPlot.zoomToRegion(locus.chr, locus.start_pos, locus.end_pos);
    }
    
}

async function startQuickAnalysis() {
    if (!currentAnalysisTaskId) {
        alert('Please upload a valid loci file before starting analysis.');
        return;
    }
    if (analysisRunActive) {
        alert('An analysis is already running.');
        return;
    }
    
    // Get target phenotype
    const targetPhenotype = document.getElementById('target-phenotype').value.trim();
    if (!targetPhenotype) {
        alert('Please enter a target phenotype (e.g., Asthma, Type 2 Diabetes).');
        return;
    }
    cancelAnalysisChatRequest();
    clearAnalysisResults();
    currentAnalysisHistoryId = null;
    analysisConversations.clear();
    for (const progress of autoProgressCards.values()) window.GWASMindProgress?.dispose(progress);
    autoProgressCards.clear();
    document.getElementById('panel-log-area').replaceChildren();
    getAutoLocusBoard()?.reset({ analysis_mode: 'prior_report_screening',
        source_phenotype: targetPhenotype, target_phenotype: targetPhenotype });
    
    // Show progress section
    document.getElementById('analysis-upload-section').style.display = 'none';
    document.getElementById('analysis-progress-section').style.display = 'block';
    document.getElementById('analysis-results-section').style.display = 'none';
    document.getElementById('panel-progress-bar').style.width = '0%';
    document.getElementById('panel-progress-text').textContent = '0%';
    updateAnalysisScreeningStats({ known: 0, novel: 0, unresolved: 0 });
    analysisRunActive = true;
    setAnalysisStartEnabled(false);
    const streamState = {
        generation: ++analysisStreamGeneration,
        taskId: currentAnalysisTaskId,
        phenotype: targetPhenotype,
        lastSequence: 0,
        reconnectAttempts: 0,
        sawTerminal: false,
        sawDone: false
    };
    connectAnalysisStream(streamState);
}

function isCurrentAnalysisStream(state) {
    return analysisRunActive
        && state.generation === analysisStreamGeneration
        && state.taskId === currentAnalysisTaskId;
}

function connectAnalysisStream(state) {
    if (!isCurrentAnalysisStream(state)) return;
    closeAnalysisStream();

    let eventSource;
    try {
        eventSource = new EventSource(`/api/analysis/quick-stream/${state.taskId}?phenotype=${encodeURIComponent(state.phenotype)}`);
    } catch (error) {
        scheduleAnalysisStreamReconnect(state);
        return;
    }
    activeAnalysisEventSource = eventSource;

    eventSource.onopen = () => {
        if (isCurrentAnalysisStream(state) && activeAnalysisEventSource === eventSource) {
            analysisStreamConnected = true;
        }
    };

    eventSource.onmessage = (event) => {
        if (!isCurrentAnalysisStream(state) || activeAnalysisEventSource !== eventSource) return;
        if (event.data === '[DONE]') {
            state.sawDone = true;
            closeAnalysisStream();
            analysisRunActive = false;
            if (!state.sawTerminal) {
                requestAnalysisCancellation(state.taskId);
                showAnalysisFailure('The analysis stream ended before a result was produced.');
            }
            return;
        }

        try {
            const data = JSON.parse(event.data);
            if (data.type !== 'timing' && Number.isSafeInteger(data.seq)) {
                if (data.seq <= state.lastSequence) return;
                state.lastSequence = data.seq;
            }
            if (data.type === 'complete' || data.type === 'error') state.sawTerminal = true;
            handleAnalysisEvent(data);
        } catch (error) {
            console.error('Parse error:', error);
        }
    };

    eventSource.onerror = () => {
        if (!isCurrentAnalysisStream(state) || activeAnalysisEventSource !== eventSource) return;
        eventSource.close();
        activeAnalysisEventSource = null;
        analysisStreamConnected = false;
        if (state.sawTerminal || state.sawDone) {
            analysisRunActive = false;
            return;
        }
        scheduleAnalysisStreamReconnect(state);
    };
}

function scheduleAnalysisStreamReconnect(state) {
    if (!isCurrentAnalysisStream(state)) return;
    if (state.reconnectAttempts >= MAX_ANALYSIS_STREAM_RECONNECTS) {
        abandonAnalysisRun('Connection lost before analysis completed. The server run was cancelled.');
        return;
    }
    const delay = ANALYSIS_STREAM_RECONNECT_DELAYS_MS[state.reconnectAttempts]
        ?? ANALYSIS_STREAM_RECONNECT_DELAYS_MS[ANALYSIS_STREAM_RECONNECT_DELAYS_MS.length - 1];
    state.reconnectAttempts += 1;
    analysisReconnectTimer = setTimeout(() => {
        analysisReconnectTimer = null;
        connectAnalysisStream(state);
    }, delay);
}

function handleAnalysisEvent(data) {
    const preserveView = window.GWASMindAutoProgress?.preserveView;
    return preserveView ? preserveView(document.getElementById('panel-locus-progress'), () => applyAnalysisEvent(data))
        : applyAnalysisEvent(data);
}

function applyAnalysisEvent(data) {
    const board = getAutoLocusBoard();
    const handled = board?.handle(data);
    if (board?.hasLoci) document.getElementById('panel-current-task').style.display = 'none';
    switch (data.type) {
        case 'workflow_ranking_evidence': {
            if (!Number.isSafeInteger(data.locus_index) || data.locus_index < 0
                    || !['prioritization', 'exploratory_prioritization'].includes(data.workflow)
                    || !board?.rows?.has(data.locus_index)) break;
            const descriptor = window.GWASMindProgress?.publicRankingEvidence?.(data);
            if (!descriptor) break;
            const number = data.locus_index + 1;
            const previous = analysisRankingEvidence.get(number) || [];
            const entries = new Map(previous.map(row => [`${row.run_id}:${row.query_index}`, row]));
            entries.set(`${descriptor.run_id}:${descriptor.query_index}`, descriptor);
            analysisRankingEvidence.set(number, [...entries.values()].slice(0, 256));
            window.GWASMindProgress?.renderRankingEvidence?.(board.resultContainer(number), analysisRankingEvidence.get(number));
            break;
        }
        case 'loci':
            renderAnalysisResults();
            break;
        case 'locus_state':
            break;
        case 'workflow_review_update':
        case 'workflow_variant_normalization':
        case 'workflow_stage': {
            if (handled) break;
            const label = data.workflow === 'screening' ? ''
                : data.workflow === 'exploratory_prioritization' ? 'Gene evidence assessment · novelty unresolved' : 'Gene prioritization';
            const area = document.getElementById('panel-log-area');
            const key = `${currentAnalysisTaskId}-${data.locus_index}-${data.workflow}`;
            let progress = autoProgressCards.get(key);
            if (!progress && window.GWASMindProgress) {
                const title = document.createElement('p');
                title.textContent = `Locus ${Number(data.locus_index) + 1}` + (label ? ` · ${label}` : '');
                area.append(title);
                progress = window.GWASMindProgress.create('Preparing the analysis...', area, { fixedWorkflow: true });
                autoProgressCards.set(key, progress);
            }
            if (progress) window.GWASMindProgress.update(progress, { ...data,
                type: data.type === 'workflow_review_update' ? 'review_update'
                    : data.type === 'workflow_variant_normalization' ? 'variant_normalization' : 'stage' });
            break;
        }
        case 'progress':
            document.getElementById('panel-progress-bar').style.width = data.percent + '%';
            document.getElementById('panel-progress-text').textContent = data.percent + '%';
            break;
        case 'task':
            if (!board?.hasLoci) document.getElementById('panel-current-task').querySelector('span').textContent = data.text;
            break;
        case 'stats':
            if (data.snps !== undefined) document.getElementById('panel-stat-snps').textContent = data.snps;
            if (data.loci !== undefined) document.getElementById('panel-stat-loci').textContent = data.loci;
            updateAnalysisScreeningStats(data);
            break;
        case 'log':
            if (!handled) addPanelLog(data.message, data.level);
            break;
        case 'results':
            acceptAnalysisHistoryId(data.history_id);
            analysisResults = Array.isArray(data.data) ? data.data : [];
            renderAnalysisResults();
            if (currentLocusContext) {
                const discussed = analysisResults.find(locus => locus.locus_no === currentLocusContext.locus_no);
                if (discussed) currentLocusContext = discussed;
            }
            break;
        case 'complete':
            for (const progress of autoProgressCards.values()) window.GWASMindProgress?.complete(progress);
            board?.dispose();
            acceptAnalysisHistoryId(data.history_id);
            showAnalysisResults(data.download_url);
            break;
        case 'error':
            showAnalysisFailure(data.message || 'Analysis failed.');
            break;
    }
}

function updateAnalysisScreeningStats(data = {}) {
    const legacy = analysisResults.length > 0
        && analysisResults.every(locus => locus.analysis_mode !== 'prior_report_screening');
    document.getElementById('panel-label-known').textContent = legacy ? 'Prioritized' : 'Known';
    document.getElementById('panel-label-novel').textContent = legacy ? 'Needs review' : 'Novel (in scope)';
    document.getElementById('panel-unresolved-stat').hidden = legacy;
    const values = legacy ? {
        known: data.prioritized ?? analysisResults.filter(locus => locus.decision_status === 'PRIORITIZED' || locus.status === 'Prioritized').length,
        novel: data.needs_review ?? analysisResults.filter(locus => locus.status === 'Needs review').length,
        unresolved: 0
    } : window.GWASMindAutoProgress?.screeningCounts?.(analysisResults) || { known: 0, novel: 0, unresolved: 0 };
    for (const key of ['known', 'novel', 'unresolved']) {
        const value = !legacy && data[key] !== undefined ? data[key] : values[key];
        document.getElementById(`panel-stat-${key}`).textContent = String(Math.max(0, Number(value) || 0));
    }
}

function addPanelLog(message, level = 'info') {
    const logArea = document.getElementById('panel-log-area');
    const entry = document.createElement('div');
    entry.className = 'log-entry ' + level;
    entry.textContent = message;
    logArea.appendChild(entry);
    // Preserve the user's current log position as new entries arrive.
}

function showAnalysisResults(downloadUrl) {
    document.getElementById('analysis-progress-section').style.display = autoLocusBoard?.hasLoci ? 'block' : 'none';
    document.getElementById('analysis-results-section').style.display = 'block';
    document.getElementById('download-report-btn').href = downloadUrl;
    document.querySelector('.results-title').textContent = 'Complete';
    document.getElementById('download-report-btn').style.display = '';
    
    renderAnalysisResults();
}

function renderAnalysisScopeWarning(container, locus) {
    if (!container) return;
    let notice = container.querySelector('.screening-scope-warning');
    const warnings = locus.analysis_mode === 'prior_report_screening' && Array.isArray(locus.warnings)
        ? locus.warnings.filter(value => typeof value === 'string' && value.trim()) : [];
    if (!warnings.length) { notice?.remove(); return; }
    if (!notice) {
        notice = document.createElement('p');
        notice.className = 'screening-scope-warning auto-locus-alert';
        container.append(notice);
    }
    const scope = locus.auto_scope_review || {};
    const counts = [
        ['excluded_source_variant_count', 'source SNP', 'source SNPs'],
        ['excluded_proxy_query_count', 'proxy query', 'proxy queries'],
        ['excluded_unidentified_row_count', 'unidentified LD row', 'unidentified LD rows']
    ].filter(([key]) => Number.isSafeInteger(scope[key]) && scope[key] > 0)
        .map(([key, singular, plural]) => `${scope[key]} ${scope[key] === 1 ? singular : plural}`);
    notice.textContent = counts.length ? `Warning: excluded ${counts.join(' · ')}.`
        : 'Warning: screening scope has limitations. See Details.';
    notice.title = warnings.join('\n');
}

function renderAnalysisResults() {
    updateAnalysisScreeningStats();
    
    // Populate results as cards
    const resultsList = document.getElementById('results-list');
    resultsList.style.display = autoLocusBoard?.hasLoci && analysisResults.every(locus => autoLocusBoard.resultContainer(locus.locus_no)) ? 'none' : '';
    
    analysisResults.forEach(locus => {
        autoLocusBoard?.setResult?.(locus);
        const screening = window.GWASMindAutoProgress?.screeningResult?.(locus);
        const snapshots = rankingEvidenceForLocus(locus);
        const existing = analysisResultCards.get(locus.locus_no);
        if (existing) {
            const badge = existing.querySelector('.status-badge-small');
            badge.textContent = screening?.label || locus.status;
            badge.className = `status-badge-small ${screening?.classification || (locus.status === 'Prioritized' ? 'prioritized' : 'needs-review')}`;
            existing.querySelector('.lead-snp').textContent = locus.lead_rsid;
            const topGene = existing.querySelector('.top-gene');
            if (topGene) topGene.textContent = locus.top_gene || '-';
            const reason = existing.querySelector('.screening-reason');
            if (reason) reason.textContent = screening?.summary || '';
            renderAnalysisScopeWarning(existing.querySelector('.screening-info'), locus);
            const rankingScope = existing.querySelector('.ranking-display-summary');
            if (rankingScope) {
                rankingScope.textContent = window.GWASMindProgress?.rankingDisplaySummary?.(locus) || '';
                rankingScope.hidden = !rankingScope.textContent;
            }
            const notice = existing.querySelector('.locus-novelty-notice');
            if (notice) { notice.textContent = screening ? '' : locus.novelty_notice || ''; notice.hidden = !notice.textContent; }
            const preview = existing.querySelector('.traits-preview');
            preview.title = locus.gwas_traits || ''; preview.textContent = (locus.gwas_traits || (screening ? '' : '-')).slice(0, 40);
            preview.hidden = !preview.textContent;
            const host = autoLocusBoard?.resultContainer(locus.locus_no) || resultsList;
            if (existing.parentElement !== host) host.append(existing);
            window.GWASMindProgress?.renderRankingEvidence?.(host === resultsList ? existing : host, snapshots);
            return;
        }
        const card = document.createElement('div');
        card.className = 'result-card';
        card.dataset.locusNo = locus.locus_no;
        const statusClass = screening?.classification || (locus.status === 'Prioritized' ? 'prioritized' : 'needs-review');
        const header = document.createElement('div');
        header.className = 'result-card-header';
        const locusInfo = document.createElement('div');
        locusInfo.className = 'locus-info';
        const locusNumber = document.createElement('span');
        locusNumber.className = 'locus-num';
        locusNumber.textContent = `Locus ${locus.locus_no}`;
        const badge = document.createElement('span');
        badge.className = `status-badge-small ${statusClass}`;
        badge.textContent = screening?.label || locus.status;
        locusInfo.append(locusNumber, badge);
        const lead = document.createElement('span');
        lead.className = 'lead-snp';
        lead.textContent = locus.lead_rsid;
        header.append(locusInfo, lead);

        const body = document.createElement('div');
        body.className = 'result-card-body';
        const geneInfo = document.createElement('div');
        geneInfo.className = screening ? 'screening-info' : 'gene-info';
        if (screening) {
            const reason = document.createElement('p');
            reason.className = 'screening-reason';
            reason.textContent = screening.summary;
            geneInfo.append(reason);
            renderAnalysisScopeWarning(geneInfo, locus);
        } else {
            const topGene = document.createElement('div');
            topGene.className = 'top-gene';
            topGene.textContent = locus.top_gene || '-';
            const rankingScope = document.createElement('div');
            rankingScope.className = 'ranking-display-summary';
            rankingScope.textContent = window.GWASMindProgress?.rankingDisplaySummary?.(locus) || '';
            rankingScope.hidden = !rankingScope.textContent;
            geneInfo.append(topGene, rankingScope);
        }
        const traits = document.createElement('div');
        traits.className = 'traits-preview';
        traits.title = locus.gwas_traits || '';
        const preview = locus.gwas_traits || (screening ? '' : '-');
        traits.textContent = preview.substring(0, 40) + (preview.length > 40 ? '...' : '');
        traits.hidden = !preview;
        geneInfo.append(traits);
        const actions = document.createElement('div');
        actions.className = 'result-card-actions';
        const chat = document.createElement('button');
        chat.type = 'button';
        chat.className = 'view-btn analysis-chat-btn';
        chat.textContent = 'Chat';
        chat.setAttribute('aria-label', `Chat about Locus ${locus.locus_no}`);
        chat.addEventListener('click', (event) => {
            event.stopPropagation();
            openAnalysisDiscussion(locus.locus_no);
        });
        const details = document.createElement('button');
        details.type = 'button';
        details.className = 'view-btn';
        details.textContent = 'Details';
        details.addEventListener('click', (event) => {
            event.stopPropagation();
            showLocusDetail(analysisResults.findIndex(item => item.locus_no === locus.locus_no));
        });
        actions.append(chat, details);
        body.append(geneInfo, actions);
        const novelty = document.createElement('p');
        novelty.className = 'locus-novelty-notice';
        novelty.textContent = screening ? '' : locus.novelty_notice || '';
        novelty.hidden = !novelty.textContent;
        card.append(header, body, novelty);
        
        // ★ Click card to highlight locus
        card.addEventListener('click', () => selectResultCard(card, locus.locus_no));
        
        analysisResultCards.set(locus.locus_no, card);
        const host = autoLocusBoard?.resultContainer(locus.locus_no) || resultsList;
        host.appendChild(card);
        window.GWASMindProgress?.renderRankingEvidence?.(host === resultsList ? card : host, snapshots);
    });
}

function rankingEvidenceForLocus(locus) {
    if (!Number.isSafeInteger(locus?.locus_no) || locus.locus_no < 1) return [];
    if (Array.isArray(locus.ranking_evidence)) {
        analysisRankingEvidence.set(locus.locus_no, locus.ranking_evidence.slice(0, 256)
            .map(value => window.GWASMindProgress?.publicRankingEvidence?.(value)).filter(Boolean));
    }
    return analysisRankingEvidence.get(locus.locus_no) || [];
}

function showAnalysisFailure(message) {
    for (const progress of autoProgressCards.values()) window.GWASMindProgress?.fail(progress);
    autoLocusBoard?.dispose();
    document.getElementById('analysis-progress-section').style.display = autoLocusBoard?.hasLoci ? 'block' : 'none';
    document.getElementById('analysis-results-section').style.display = 'block';
    document.getElementById('analysis-chat-area').style.display = 'none';
    document.querySelector('.results-title').textContent = 'Failed';
    document.getElementById('download-report-btn').style.display = 'none';
    const resultsList = document.getElementById('results-list');
    resultsList.style.display = '';
    resultsList.replaceChildren();
    const errorCard = document.createElement('div');
    errorCard.className = 'result-card';
    errorCard.textContent = `Error: ${message}`;
    resultsList.appendChild(errorCard);
}

// ★ Select a result card and highlight on plot
function selectResultCard(card, locusNo) {
    selectAnalysisLocus(locusNo);
    // Update selected state
    document.querySelectorAll('.result-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    
}

function selectAnalysisLocus(locusNo) {
    selectedLocusNo = Number(locusNo);
    autoLocusBoard?.select?.(selectedLocusNo);
    for (const [number, card] of analysisResultCards) card.classList.toggle('selected', number === selectedLocusNo);
    highlightLocus(selectedLocusNo);
}

function openAnalysisDiscussion(locusNo) {
    const locus = analysisResults.find(item => item.locus_no === Number(locusNo));
    if (!locus) return;
    selectAnalysisLocus(locus.locus_no);
    if (currentLocusContext?.locus_no !== locus.locus_no) updateAnalysisChatContext(locus);
    else currentLocusContext = locus;
    analysisDiscussionClosed = false;
    document.getElementById('analysis-chat-area').style.display = 'flex';
}

function closeAnalysisDiscussion() {
    analysisDiscussionClosed = true;
    document.getElementById('analysis-chat-area').style.display = 'none';
}

document.getElementById('close-analysis-discussion')?.addEventListener('click', closeAnalysisDiscussion);

function showLocusDetail(idx) {
    const locus = analysisResults[idx];
    if (!locus) return;
    const escapeText = window.GWASMindSafe?.escapeText || ((value) => String(value ?? ''));
    document.getElementById('detail-modal-title').textContent = 
        `Locus ${locus.locus_no}: ${locus.lead_rsid}`;
    const screening = window.GWASMindAutoProgress?.screeningResult?.(locus);
    document.getElementById('detail-modal-body').classList.toggle('screening-detail', Boolean(screening));
    if (locus.report_markdown) {
        const body = document.getElementById('detail-modal-body');
        body.innerHTML = formatMarkdown(locus.report_markdown);
        window.attachReportActions?.(body,
            window.GWASMindAutoProgress?.screeningReportMarkdown?.(locus) ?? locus.report_markdown);
        window.GWASMindProgress?.renderVariantNormalizations(body, locus.variant_normalizations);
        window.GWASMindProgress?.renderRankingEvidence?.(body, rankingEvidenceForLocus(locus));
        window.GWASMindProgress?.renderGeneEvidenceCards?.(body, locus.gene_evidence_cards || []);
        window.GWASMindAutoProgress?.renderScreeningConclusion?.(body, locus);
        document.getElementById('detail-modal').style.display = 'flex';
        return;
    }
    
    // Format AI interpretation with Markdown
    let interpretation = locus.llm_interpretation || '';
    
    // Collapse repeated blank lines while preserving Markdown line breaks.
    interpretation = interpretation.replace(/\n{3,}/g, '\n\n');
    interpretation = interpretation.replace(/\n\s+\n/g, '\n\n');
    
    // Normalize line breaks around headings and lists.
    interpretation = interpretation.replace(/(\*\*References\*\*|\*\*Gene Evidence\*\*)/g, '\n\n$1\n');
    
    let interpretationHtml = interpretation ? formatMarkdown(interpretation) : '';
    
    // Remove empty paragraphs.
    interpretationHtml = interpretationHtml.replace(/<p>\s*<\/p>/g, '');
    interpretationHtml = interpretationHtml.replace(/<p>(\s|&nbsp;)*<\/p>/gi, '');
    
    // Candidate genes use the server-provided fields directly.
    const candidateGenesStr = locus.candidate_genes || locus.top_gene || 'None';
    
    // References from the evidence ledger
    let referencesSection = '';
    if (locus.references) {
        referencesSection = `
<h4>References</h4>
<p>${escapeText(locus.references)}</p>`;
    } else if (locus.key_references && locus.key_references.length > 0) {
        referencesSection = `
<h4>References</h4>
<p>${escapeText(locus.key_references.slice(0, 3).join('; '))}</p>`;
    }
    
    // Evidence summary; no hidden chain-of-thought is displayed.
    let evidenceSummarySection = '';
    if (interpretationHtml) {
        evidenceSummarySection = `
<h4>Evidence Summary</h4>
<div class="markdown-content">${interpretationHtml}</div>`;
    }
    
    // Parse simple GWAS-trait Markdown with bold text and line breaks.
    let gwasTraitsHtml = escapeText(locus.gwas_traits || 'No associations found')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
    
    let content = `<h4>Basic Info</h4>
<p><strong>Chr:</strong> ${escapeText(locus.chr)} | <strong>Status:</strong> ${escapeText(screening?.label || locus.status)} | <strong>SNPs:</strong> ${escapeText(locus.snp_count)}<br>
<strong>Lead SNP:</strong> Position: ${escapeText(locus.pos || locus.lead_pos)} | -log10(p): ${escapeText(locus.neg_log_p !== undefined ? locus.neg_log_p : (locus.p_value ? (-Math.log10(parseFloat(locus.p_value))).toFixed(2) : 'N/A'))}</p>

<h4>GWAS Associations</h4>
<div class="gwas-traits-content">${gwasTraitsHtml}</div>

${screening ? '' : `<h4>Candidate Genes</h4>
<p>${escapeText(candidateGenesStr)}</p>
${window.GWASMindProgress?.rankingDisplaySummary?.(locus) ? `<p>${escapeText(window.GWASMindProgress.rankingDisplaySummary(locus))}</p>` : ''}`}
${referencesSection}
${evidenceSummarySection}`;
    
    document.getElementById('detail-modal-body').innerHTML = window.GWASMindSafe
        ? window.GWASMindSafe.sanitizeHtml(content)
        : content;
    window.GWASMindProgress?.renderRankingEvidence?.(document.getElementById('detail-modal-body'), rankingEvidenceForLocus(locus));
    window.GWASMindProgress?.renderGeneEvidenceCards?.(document.getElementById('detail-modal-body'), locus.gene_evidence_cards || []);
    window.GWASMindAutoProgress?.renderScreeningConclusion?.(document.getElementById('detail-modal-body'), locus);
    document.getElementById('detail-modal').style.display = 'flex';
}

function closeDetailModal() {
    document.getElementById('detail-modal').style.display = 'none';
}

// Close detail modal on overlay click
document.getElementById('detail-modal').addEventListener('click', (e) => {
    if (e.target.id === 'detail-modal') closeDetailModal();
});

// ============================================================
// Tutorial Modal Functions
// ============================================================
let tutorialLoaded = false;

function openTutorialModal() {
    const modal = document.getElementById('tutorial-modal');
    modal.style.display = 'flex';
    
    // Load tutorial content if not already loaded
    if (!tutorialLoaded) {
        loadTutorialContent();
    }
}

function closeTutorialModal() {
    document.getElementById('tutorial-modal').style.display = 'none';
}

async function loadTutorialContent() {
    const bodyEl = document.getElementById('tutorial-modal-body');
    
    try {
        // Fetch the tutorial markdown file
        const response = await fetch('/static/docs/tutorial.md');
        if (!response.ok) throw new Error('Failed to load tutorial');
        
        const markdown = await response.text();
        
        // Parse with the bundled Marked API, then allow only safe DOM content.
        // A missing renderer is an actionable load error, not tutorial prose.
        const safeDom = window.GWASMindSafe;
        if (!safeDom || typeof safeDom.renderMarkdown !== 'function') {
            throw new Error('Tutorial renderer is unavailable');
        }
        bodyEl.innerHTML = safeDom.renderMarkdown(markdown);
        
        // Style images for better display in modal
        const images = bodyEl.querySelectorAll('img');
        images.forEach(img => {
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            img.style.borderRadius = '8px';
            img.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
            img.style.margin = '16px 0';
            img.style.display = 'block';
        });
        
        // Add IDs to headings for anchor navigation (must be done BEFORE binding click events)
        const headings = bodyEl.querySelectorAll('h1, h2, h3');
        headings.forEach(heading => {
            // Generate ID from heading text (similar to GitHub style)
            const id = heading.textContent
                .toLowerCase()
                .replace(/[^\w\s-]/g, '')  // Remove special chars except spaces and hyphens
                .replace(/\s+/g, '-')       // Replace spaces with hyphens
                .replace(/^-+|-+$/g, '')    // Trim leading/trailing hyphens
                .replace(/-+/g, '-');       // Collapse multiple hyphens
            heading.id = id;
        });
        
        // Fix anchor links to scroll within modal instead of page navigation
        const anchorLinks = bodyEl.querySelectorAll('a[href^="#"]');
        anchorLinks.forEach(link => {
            link.style.cursor = 'pointer';  // Ensure pointer cursor
            link.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                const targetId = this.getAttribute('href').slice(1);
                // Use getElementById (works with numeric IDs)
                let targetElement = document.getElementById(targetId);
                // Fallback: search within modal body using attribute selector
                if (!targetElement) {
                    targetElement = bodyEl.querySelector('[id="' + targetId + '"]');
                }
                if (targetElement) {
                    // Scroll the element into view within the modal
                    targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
                return false;  // Extra prevention
            };
        });
        
        // Intercept /contact links to open Contact modal instead of navigating
        const contactLinks = bodyEl.querySelectorAll('a[href="/contact"], a[href*="contact"]');
        contactLinks.forEach(link => {
            link.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                closeTutorialModal();  // Close tutorial modal first
                setTimeout(() => openContactModal(), 100);  // Then open contact modal
                return false;
            };
        });
        
        tutorialLoaded = true;
    } catch (error) {
        const errorPanel = document.createElement('div');
        errorPanel.style.cssText = 'text-align: center; padding: 40px; color: #888;';
        const title = document.createElement('p');
        title.textContent = 'Failed to load tutorial content';
        const detail = document.createElement('p');
        detail.style.fontSize = '12px';
        detail.textContent = error.message;
        const link = document.createElement('a');
        link.href = '/tutorial';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.style.color = '#5a7a9a';
        link.textContent = 'Open in new tab →';
        errorPanel.append(title, detail, link);
        bodyEl.replaceChildren(errorPanel);
    }
}

// Close tutorial modal on overlay click
document.getElementById('tutorial-modal').addEventListener('click', (e) => {
    if (e.target.id === 'tutorial-modal') closeTutorialModal();
});

// ============================================================
// Contact Modal Functions
// ============================================================
function openContactModal() {
    document.getElementById('contact-modal').style.display = 'flex';
}

function closeContactModal() {
    document.getElementById('contact-modal').style.display = 'none';
}

// Close contact modal on overlay click
document.getElementById('contact-modal').addEventListener('click', (e) => {
    if (e.target.id === 'contact-modal') closeContactModal();
});

// Close modals with Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeTutorialModal();
        closeContactModal();
        closeDetailModal();
    }
});

// ============================================================
// Analysis context-aware chat.
// ============================================================
let currentLocusContext = null;  // Currently selected locus context.
let analysisChatHistory = [];    // Analysis chat history.
let analysisChatController = null;
let analysisChatGeneration = 0;

const MAX_ANALYSIS_CHAT_HISTORY = globalThis.GWASMindConfig.get('web.chat_history_limit');
const MAX_ANALYSIS_CHAT_ITEM_CHARS = globalThis.GWASMindConfig.get('web.chat_history_item_max_chars');
const SCREENING_SUGGESTIONS = {
    analysis_summary: { label: 'Summary', prompt: 'Explain the Known / Novel screening result for this locus and phenotype.' },
    analysis_regulatory: { label: 'Evidence', prompt: 'Which prior associations and LD evidence support the screening result for this locus?' },
    analysis_function: { label: 'Next steps', prompt: 'What additional evidence would resolve uncertainty in the Known / Novel status of this locus?' }
};

function cancelAnalysisChatRequest() {
    analysisChatGeneration += 1;
    if (analysisChatController) analysisChatController.abort();
    analysisChatController = null;
    const sendButton = document.querySelector('.analysis-send-btn');
    if (sendButton) sendButton.disabled = false;
}

function beginAnalysisChatRequest() {
    cancelAnalysisChatRequest();
    const request = {
        controller: new AbortController(),
        generation: ++analysisChatGeneration
    };
    analysisChatController = request.controller;
    return request;
}

function isCurrentAnalysisChatRequest(request) {
    return request.generation === analysisChatGeneration
        && request.controller === analysisChatController
        && !request.controller.signal.aborted;
}

// Update the analysis chat context.
function updateAnalysisChatContext(locus) {
    cancelAnalysisChatRequest();
    if (typeof window.cancelActiveChatRequest === 'function') {
        window.cancelActiveChatRequest();
    }
    currentLocusContext = locus;
    document.getElementById('analysis-chat-input').disabled = false;
    
    // Update the context indicator.
    const indicator = document.getElementById('context-indicator');
    const pulse = indicator.querySelector('.context-pulse');
    const text = indicator.querySelector('.context-text');
    
    pulse.classList.add('active');
    text.classList.add('active');
    text.textContent = `Discussing Locus ${locus.locus_no} (${locus.lead_rsid})...`;
    
    // Show quick suggestions.
    document.getElementById('quick-suggestions').style.display = 'flex';
    for (const button of document.getElementById('quick-suggestions').querySelectorAll('button')) {
        const suggestion = SCREENING_SUGGESTIONS[button.dataset.promptKey];
        if (!suggestion) continue;
        button.textContent = locus.analysis_mode === 'prior_report_screening' ? suggestion.label
            : button.dataset.promptKey === 'analysis_function' ? 'Gene Function' : suggestion.label;
    }
    
    // Each locus owns its full discussion; only the model input is truncated.
    analysisChatHistory = analysisConversations.get(locus.locus_no) || [];
    analysisConversations.set(locus.locus_no, analysisChatHistory);
    const messagesArea = document.getElementById('analysis-chat-messages');
    const welcome = document.createElement('div');
    welcome.className = 'analysis-welcome';
    const selected = document.createElement('p');
    selected.append('Selected ');
    const selectedLocus = document.createElement('strong');
    selectedLocus.textContent = `Locus ${locus.locus_no}`;
    selected.appendChild(selectedLocus);
    const summary = document.createElement('p');
    summary.style.marginTop = '5px';
    const screening = window.GWASMindAutoProgress?.screeningResult?.(locus);
    summary.textContent = `Lead SNP: ${locus.lead_rsid} | Status: ${screening?.label || locus.status}`
        + (screening ? '' : ` | Gene: ${locus.top_gene || 'Unknown'}`);
    const prompt = document.createElement('p');
    prompt.style.cssText = 'margin-top: 10px; color: #5a7a9a;';
    prompt.textContent = 'Ask questions about this locus';
    welcome.append(selected, summary, prompt);
    messagesArea.replaceChildren(welcome);
    for (const turn of analysisChatHistory) {
        const savedMessage = addAnalysisChatMessage(turn.content, turn.role);
        if (turn.role === 'assistant') window.attachReportActions?.(savedMessage, turn.content);
        if (turn.role === 'assistant') window.GWASMindProgress?.renderRankingEvidence?.(savedMessage, turn.ranking_evidence || []);
    }
}

// ★ Send quick suggestion
function sendAnalysisSuggestion(question) {
    if (currentLocusContext?.analysis_mode === 'prior_report_screening') {
        const key = Object.keys(SCREENING_SUGGESTIONS)
            .find(name => window.GWASMIND_UI_PROMPTS?.[name] === question);
        if (key) question = SCREENING_SUGGESTIONS[key].prompt;
    }
    document.getElementById('analysis-chat-input').value = question;
    sendAnalysisMessage();
}

// ★ Send analysis message
async function sendAnalysisMessage() {
    const input = document.getElementById('analysis-chat-input');
    const message = input.value.trim();
    
    if (!message) return;
    if (!currentLocusContext) {
        alert('Please select a locus first');
        return;
    }
    
    // Clear the input field.
    input.value = '';
    
    // Add the user message to the interface.
    addAnalysisChatMessage(message, 'user');

    const locusContext = currentLocusContext;
    const request = beginAnalysisChatRequest();
    const priorHistory = (MAX_ANALYSIS_CHAT_HISTORY === 0 ? [] : analysisChatHistory.slice(-MAX_ANALYSIS_CHAT_HISTORY))
        .map(turn => ({ role: turn.role, content: turn.content.slice(0, MAX_ANALYSIS_CHAT_ITEM_CHARS) }));
    analysisChatHistory.push({ role: 'user', content: message, timestamp: new Date().toISOString() });
    if (currentAnalysisHistoryId) window.saveWorkspaceAnalysisConversation?.(
        currentAnalysisHistoryId, Number(locusContext.locus_no), analysisChatHistory
    ).catch(error => console.warn('Unable to save the analysis question:', error.message));
    
    // Serialize locus facts as data. All model instructions live in the
    // server-side runtime prompt catalog.
    const contextData = buildLocusContextData(locusContext);
    const selection = Array.isArray(locusContext.selected_variants)
        ? locusContext.selected_variants.map(item => typeof item === 'string' ? item : item.variant_id).filter(Boolean)
        : (locusContext.lead_rsid ? [locusContext.lead_rsid] : []);
    const analysisReference = currentAnalysisHistoryId
        ? { history_id: currentAnalysisHistoryId, locus_no: Number(locusContext.locus_no) }
        : { task_id: currentAnalysisTaskId, locus_no: Number(locusContext.locus_no) };
    
    // Create the assistant-message placeholder.
    const aiMsgDiv = addAnalysisChatMessage('', 'assistant', true);
    let discussionRunId = null;
    const discussionEvidence = new Map();
    
    try {
        // Disable send button
        const sendBtn = document.querySelector('.analysis-send-btn');
        sendBtn.disabled = true;
        
        // Send request through the local single-agent service. Credentials,
        // when configured, never leave the local Python process.
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: request.controller.signal,
            body: JSON.stringify({
                message: message,
                region: locusContext.selected_region || {
                    chromosome: locusContext.chr,
                    start: locusContext.lead_pos || locusContext.pos,
                    end: locusContext.lead_pos || locusContext.pos
                },
                variants: selection,
                history: priorHistory,
                analysis_context: contextData,
                auto_analysis_ref: analysisReference
            })
        });
        
        if (!response.ok || !response.body) {
            let detail = `HTTP ${response.status}`;
            try {
                const errorPayload = await response.json();
                detail = errorPayload.error || detail;
            } catch (ignored) {}
            throw new Error(detail);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let buffer = '';
        let sawDone = false;
        let sawComplete = false;
        
        while (true) {
            if (!isCurrentAnalysisChatRequest(request)) {
                const cancelled = new Error('Request cancelled');
                cancelled.name = 'AbortError';
                throw cancelled;
            }
            const { done, value } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
            buffer = buffer.replace(/\r\n/g, '\n');
            const blocks = buffer.split('\n\n');
            buffer = blocks.pop() || '';
            for (const block of blocks) {
                const data = block.split('\n')
                    .filter((line) => line.startsWith('data:'))
                    .map((line) => line.slice(5).trimStart())
                    .join('\n');
                if (!data) continue;
                if (data === '[DONE]') {
                    sawDone = true;
                    continue;
                }
                const parsed = JSON.parse(data);
                if (parsed.type === 'run' && typeof parsed.content === 'string' && /^[a-f0-9]{32}$/.test(parsed.content)) {
                    discussionRunId = parsed.content;
                }
                if (parsed.type === 'ranking_evidence') {
                    const descriptor = window.GWASMindProgress?.publicRankingEvidence?.(parsed);
                    if (descriptor && descriptor.run_id === discussionRunId) {
                        discussionEvidence.set(`${descriptor.run_id}:${descriptor.query_index}`, descriptor);
                        window.GWASMindProgress?.renderRankingEvidence?.(aiMsgDiv, [...discussionEvidence.values()].slice(0, 256));
                    }
                }
                if (parsed.type === 'error') {
                    throw new Error(parsed.content || 'Analysis failed');
                }
                if (parsed.type === 'complete') sawComplete = true;
                if ((parsed.type === 'status' || parsed.type === 'stage') && !fullResponse) {
                    // The workflow reports progress as stage events; show their public text.
                    const progressText = parsed.type === 'stage' ? (parsed.detail || parsed.summary) : parsed.content;
                    aiMsgDiv.textContent = progressText || 'Retrieving evidence...';
                    window.GWASMindProgress?.renderRankingEvidence?.(aiMsgDiv, [...discussionEvidence.values()].slice(0, 256));
                } else if (parsed.type === 'answer_chunk' && parsed.content) {
                    fullResponse += parsed.content;
                    aiMsgDiv.innerHTML = formatMarkdown(fullResponse);
                    window.GWASMindProgress?.renderRankingEvidence?.(aiMsgDiv, [...discussionEvidence.values()].slice(0, 256));
                }
            }
            if (done) break;
        }
        if (buffer.trim() || !sawDone || !sawComplete) {
            throw new Error('The response stream ended unexpectedly');
        }
        if (!isCurrentAnalysisChatRequest(request)) {
            const cancelled = new Error('Request cancelled');
            cancelled.name = 'AbortError';
            throw cancelled;
        }
        
        // Remove the streaming class after completion.
        aiMsgDiv.classList.remove('streaming');
        window.attachReportActions?.(aiMsgDiv, fullResponse);
        window.GWASMindProgress?.renderRankingEvidence?.(aiMsgDiv, [...discussionEvidence.values()].slice(0, 256));
        
        // Record the completed exchange.
        const timestamp = new Date().toISOString();
        analysisChatHistory.push({ role: 'assistant', content: fullResponse, timestamp,
            ...(discussionEvidence.size ? { ranking_evidence: [...discussionEvidence.values()].slice(0, 256) } : {}) });
        try {
            if (currentAnalysisHistoryId) await window.saveWorkspaceAnalysisConversation?.(
                currentAnalysisHistoryId, Number(locusContext.locus_no), analysisChatHistory
            );
        } catch (error) {
            const notice = document.createElement('p');
            notice.className = 'dialog-error';
            notice.textContent = 'Discussion is visible but could not be saved to history. '
                + 'Use Copy or Export to keep it, or retry after local storage is available.';
            aiMsgDiv.append(notice);
        }
        
    } catch (error) {
        if (error.name === 'AbortError' || !isCurrentAnalysisChatRequest(request)) {
            if (aiMsgDiv.isConnected) {
                aiMsgDiv.textContent = 'Request cancelled.';
                aiMsgDiv.classList.remove('streaming');
            }
            return;
        }
        request.controller.abort();
        aiMsgDiv.textContent = 'Sorry, an error occurred: ' + error.message;
        window.GWASMindProgress?.renderRankingEvidence?.(aiMsgDiv, [...discussionEvidence.values()].slice(0, 256));
        aiMsgDiv.classList.remove('streaming');
    } finally {
        if (analysisChatController === request.controller) {
            analysisChatController = null;
            document.querySelector('.analysis-send-btn').disabled = false;
        }
    }
}

// ★ Build locus context data (not an LLM prompt)
function buildLocusContextData(locus) {
    const targetPhenotype = locus.query_phenotype || document.getElementById('target-phenotype').value.trim() || 'Not specified';
    const screening = window.GWASMindAutoProgress?.screeningResult?.(locus);

    return JSON.stringify({
        target_phenotype: targetPhenotype,
        source_phenotype: locus.source_phenotype || targetPhenotype,
        interpretation_rules: {
            status_is_evidence_sufficiency_not_historical_novelty: !screening,
            database_absence_is_not_proof_of_novelty: true,
            ...(screening ? { novel_status_is_limited_to_completed_screening_scope: true } : {})
        },
        ...(screening ? { analysis_mode: locus.analysis_mode, prior_report_screening: {
            status: locus.screening_status || 'UNRESOLVED',
            label: screening.label, reason: screening.reason,
            novelty_status: locus.novelty_status || null
        } } : {}),
        locus: {
            locus_number: locus.locus_no,
            lead_snp: locus.lead_rsid,
            chromosome: locus.chr,
            lead_position: locus.lead_pos,
            negative_log10_p: locus.neg_log_p !== undefined
                ? locus.neg_log_p
                : (locus.p_value
                    ? (-Math.log10(parseFloat(locus.p_value))).toFixed(2)
                    : null),
            status: locus.status,
            snp_count: locus.snp_count,
            selected_variants: locus.selected_variants || [],
            selected_region: locus.selected_region || null
        },
        gwas_associations: locus.gwas_traits || null,
        ...(screening ? {} : { candidate_genes: {
            top_prioritized: locus.top_gene || null,
            prioritized: locus.l2g_genes || null,
            enhancer_linked: locus.enhancer_genes || null
        } }),
        biological_annotations: {
            go_terms: locus.go_terms || null,
            reactome_pathways: locus.pathways || null,
            expression_tissues: locus.expression_tissues || null,
            related_drugs: locus.drugs || null
        },
        previous_evidence_summary: locus.llm_interpretation || null
    });
}

// ★ Add analysis chat message
function addAnalysisChatMessage(content, role, isStreaming = false) {
    const messagesArea = document.getElementById('analysis-chat-messages');
    
    // Remove welcome message
    const welcome = messagesArea.querySelector('.analysis-welcome');
    if (welcome) welcome.remove();
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `analysis-msg ${role}` + (isStreaming ? ' streaming' : '');
    if (content) {
        msgDiv.innerHTML = formatMarkdown(content);
    } else {
        const indicator = document.createElement('span');
        indicator.className = 'typing-indicator';
        indicator.textContent = 'Thinking...';
        msgDiv.appendChild(indicator);
    }
    
    messagesArea.appendChild(msgDiv);
    messagesArea.scrollTop = messagesArea.scrollHeight;
    
    return msgDiv;
}

// Render Markdown through the shared safe renderer.
function formatMarkdown(text) {
    if (!text) return '';
    const safeDom = window.GWASMindSafe;
    // Use marked.js only through the explicit parse-then-sanitize boundary.
    if (safeDom && typeof safeDom.renderMarkdown === 'function') {
        return safeDom.renderMarkdown(String(text));
    }
    // Fallback to simple formatting if marked.js not available
    const escaped = safeDom
        ? safeDom.escapeText(text)
        : String(text).replace(/[&<>"']/g, (character) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[character]));
    return escaped.replace(/\n/g, '<br>');
}

window.restoreWorkspaceAnalysis = async function (snapshot) {
    clearAnalysisResults();
    const savedRows = Array.isArray(snapshot.loci) && snapshot.loci.length ? snapshot.loci
        : (snapshot.results || []).map((locus, index) => ({ locus_index: (locus.locus_no || index + 1) - 1,
            locus_no: locus.locus_no || index + 1, locus_id: locus.lead_rsid || '', state: 'completed', timing: locus.timing }));
    getAutoLocusBoard()?.restore({ ...snapshot, loci: savedRows }, { historical: true });
    currentAnalysisTaskId = snapshot.task_id || null;
    currentAnalysisHistoryId = snapshot.history_id || null;
    analysisConversations = new Map(Object.entries(snapshot.chat_messages || {})
        .filter(([, messages]) => Array.isArray(messages))
        .map(([locus, messages]) => [Number(locus), messages]));
    analysisResults = snapshot.results || [];
    analysisResults.forEach((locus, index) => {
        const queryIndex = Number.isInteger(locus.locus_no) && locus.locus_no > 0 ? locus.locus_no - 1 : index;
        locus.selected_variants ||= snapshot.queries?.[queryIndex]?.variants || [];
        locus.source_phenotype ||= snapshot.source_phenotype || '';
        locus.query_phenotype ||= snapshot.target_phenotype || locus.source_phenotype;
    });
    uploadedSnpData = snapshot.snp_positions || [];
    document.getElementById('target-phenotype').value = snapshot.target_phenotype || snapshot.source_phenotype || '';
    document.getElementById('analysis-upload-section').style.display = 'none';
    await switchTab('analysis');
    showAnalysisResults('/api/history/analyses/' + snapshot.history_id + '/download');
    if (autoLocusBoard?.hasLoci) {
        const savedLoci = savedRows;
        const loci = savedLoci.length || autoLocusBoard.rows.size;
        const completed = savedLoci.filter(item => ['completed', 'failed'].includes(item.state)).length;
        handleAnalysisEvent({ type: 'stats', loci,
            snps: (snapshot.queries || []).reduce((sum, query) => sum + (query.variants?.length || 0), 0) });
        handleAnalysisEvent({ type: 'progress', percent: Math.round(completed / loci * 100) });
    }
    if (snapshot.status && snapshot.status !== 'completed') {
        document.querySelector('.results-title').textContent = 'Saved progress · ' + snapshot.status;
        const summary = document.getElementById('panel-batch-summary');
        if (autoLocusBoard?.hasLoci && summary) summary.textContent = 'Saved snapshot · ' + summary.textContent;
    }
    if (!analysisResults.length) document.querySelector('.results-title').textContent = snapshot.status || 'Saved analysis';
    highlightUploadedSnps(uploadedSnpData);
};

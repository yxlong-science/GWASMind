// Check database status on load
let gwasUploadGeneration = 0;
let gwasUploadController = null;
let gwasUploadRedirectTimer = null;
let demoLoadGeneration = 0;
let demoLoadController = null;

function abortableDelay(milliseconds, signal) {
    return new Promise((resolve, reject) => {
        if (signal?.aborted) {
            const error = new Error('The operation was cancelled');
            error.name = 'AbortError';
            reject(error);
            return;
        }
        const timer = setTimeout(() => {
            signal?.removeEventListener('abort', handleAbort);
            resolve();
        }, milliseconds);
        const handleAbort = () => {
            clearTimeout(timer);
            const error = new Error('The operation was cancelled');
            error.name = 'AbortError';
            reject(error);
        };
        signal?.addEventListener('abort', handleAbort, { once: true });
    });
}

function cancelGwasUpload() {
    gwasUploadGeneration += 1;
    if (gwasUploadController) gwasUploadController.abort();
    gwasUploadController = null;
    if (gwasUploadRedirectTimer !== null) {
        clearTimeout(gwasUploadRedirectTimer);
        gwasUploadRedirectTimer = null;
    }
    const label = document.getElementById('upload-label');
    const progress = document.getElementById('upload-progress');
    const input = document.getElementById('file-input');
    if (label) {
        label.textContent = 'Choose File';
        label.style.pointerEvents = 'auto';
    }
    if (progress) progress.style.display = 'none';
    if (input) input.value = '';
}

function cancelDemoLoad() {
    demoLoadGeneration += 1;
    if (demoLoadController) demoLoadController.abort();
    demoLoadController = null;
    setDemoControlsBusy(false);
}

function setDemoControlsBusy(busy) {
    document.querySelectorAll('[data-action="load-demo"]').forEach((element) => {
        element.setAttribute('aria-disabled', busy ? 'true' : 'false');
        element.style.pointerEvents = busy ? 'none' : '';
    });
}

fetch('/api/db-status').then(r => r.json()).then(d => {
    // The status describes the installed local bridge only. Upstream live
    // reachability is checked by each real analysis and is never fabricated.
    if (!d.bridge_available) document.getElementById('db-note').style.display = 'block';
}).catch(() => {
    document.getElementById('db-note').style.display = 'block';
});

async function loadDemo(name) {
    const progressDiv = document.getElementById('demo-progress');
    const progressFill = document.getElementById('demo-progress-fill');
    const progressText = document.getElementById('demo-progress-text');
    
    if (!['asthma', 'mdd'].includes(String(name))) {
        progressDiv.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Error: Unknown demo dataset';
        return;
    }

    cancelDemoLoad();
    cancelGwasUpload();
    const controller = new AbortController();
    const generation = ++demoLoadGeneration;
    demoLoadController = controller;
    const isCurrent = () => generation === demoLoadGeneration && !controller.signal.aborted;

    setDemoControlsBusy(true);
    progressDiv.style.display = 'block';
    
    try {
        // Stage 1: Preparing
        progressFill.style.width = '10%';
        progressText.textContent = '10% - Preparing...';
        await abortableDelay(200, controller.signal);
        
        // Stage 2: Loading
        progressFill.style.width = '30%';
        progressText.textContent = '30% - Loading demo data...';
        
        const response = await fetch(`/load-demo/${name}`, {
            method: 'POST',
            signal: controller.signal
        });
        if (!isCurrent()) return;
        
        // Stage 3: Processing
        progressFill.style.width = '60%';
        progressText.textContent = '60% - Processing data...';
        await abortableDelay(300, controller.signal);
        
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `Server error: ${response.status}`);
        if (!isCurrent()) return;
        
        if (result.success) {
            // Stage 4: Loading visualization
            progressFill.style.width = '85%';
            progressText.textContent = '85% - Loading visualization...';
            await abortableDelay(300, controller.signal);
            
            // Stage 5: Verify data is loaded before redirecting
            progressFill.style.width = '90%';
            progressText.textContent = '90% - Verifying data...';
            
            const verifyResponse = await fetch('/api/has-data', {
                method: 'GET',
                signal: controller.signal
            });
            const verifyData = await verifyResponse.json();
            if (!verifyResponse.ok) throw new Error(verifyData.error || 'Data verification failed: Server error');
            if (!verifyData || !verifyData.has_data) throw new Error('Data verification failed: No data found');
            if (!isCurrent()) return;

            progressFill.style.width = '100%';
            progressText.textContent = '100% - ' + result.message;
            await abortableDelay(500, controller.signal);
            if (isCurrent()) window.location.replace(result.redirect || '/app');
        } else {
            throw new Error(result.error || 'Failed to load demo');
        }
    } catch (error) {
        if (error.name === 'AbortError' || !isCurrent()) return;
        progressText.textContent = 'Error: ' + error.message;
        progressFill.style.width = '0%';
        setTimeout(() => {
            if (generation === demoLoadGeneration) progressDiv.style.display = 'none';
        }, 3000);
    } finally {
        if (demoLoadController === controller) {
            demoLoadController = null;
            setDemoControlsBusy(false);
        }
    }
}

document.getElementById('file-input').addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (!file) return;

    cancelDemoLoad();
    cancelGwasUpload();
    const controller = new AbortController();
    const generation = ++gwasUploadGeneration;
    gwasUploadController = controller;
    
    const uploadLabel = document.getElementById('upload-label');
    const progressDiv = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const errorDiv = document.getElementById('upload-error');
    const phenotypeInput = document.getElementById('source-phenotype');
    const sourcePhenotype = String(phenotypeInput?.value || '').trim().replace(/\s+/g, ' ');

    if (!sourcePhenotype) {
        errorDiv.textContent = 'Enter the phenotype measured by this GWAS before choosing a file.';
        errorDiv.style.display = 'block';
        e.target.value = '';
        phenotypeInput?.focus();
        if (gwasUploadController === controller) gwasUploadController = null;
        return;
    }
    
    if (file.size === 0) {
        errorDiv.textContent = 'Selected file is empty (0 bytes)';
        errorDiv.style.display = 'block';
        uploadLabel.textContent = 'Choose File';
        uploadLabel.style.pointerEvents = 'auto';
        progressDiv.style.display = 'none';
        if (gwasUploadController === controller) gwasUploadController = null;
        return;
    }
    
    uploadLabel.textContent = 'Processing...';
    uploadLabel.style.pointerEvents = 'none';
    progressDiv.style.display = 'block';
    errorDiv.style.display = 'none';
    
    try {
        // ★ All files processed client-side using chunked streaming
        const processedResult = await GWASProcessor.processFile(file, (percent, message) => {
            if (generation !== gwasUploadGeneration || controller.signal.aborted) return;
            progressFill.style.width = percent + '%';
            progressText.textContent = `${percent}% - ${message}`;
        }, { signal: controller.signal });
        if (generation !== gwasUploadGeneration || controller.signal.aborted) return;
        processedResult.metadata.source_phenotype = sourcePhenotype;
        
        // Upload sampled data to server
        progressFill.style.width = '96%';
        progressText.textContent = `96% - Uploading ${processedResult.data.length.toLocaleString()} SNPs...`;
        
        const response = await fetch('/upload-processed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(processedResult),
            signal: controller.signal
        });
        
        const result = await response.json();
        if (generation !== gwasUploadGeneration || controller.signal.aborted) return;
        
        if (response.ok && result.success) {
            progressFill.style.width = '100%';
            progressText.textContent = '100% - ' + result.message;
            gwasUploadRedirectTimer = setTimeout(() => {
                if (generation === gwasUploadGeneration) window.location.href = result.redirect;
            }, 800);
        } else {
            throw new Error(result.error || 'Upload failed');
        }
        
    } catch (error) {
        if (error.name === 'AbortError' || generation !== gwasUploadGeneration) return;
        console.error('Processing error:', error);
        errorDiv.textContent = error.message;
        errorDiv.style.display = 'block';
        uploadLabel.textContent = 'Choose File';
        uploadLabel.style.pointerEvents = 'auto';
        progressDiv.style.display = 'none';
        e.target.value = '';
    } finally {
        if (gwasUploadController === controller) gwasUploadController = null;
    }
});

window.addEventListener('beforeunload', () => {
    cancelDemoLoad();
    cancelGwasUpload();
});

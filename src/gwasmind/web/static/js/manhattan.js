/**
 * Manhattan Plot Module
 * Interactive GWAS visualization with Plotly.js
 */
const MANHATTAN_GENOME_WIDE_P_VALUE = globalThis.GWASMindConfig.get('analysis.gwas_significance_p');
const MANHATTAN_LOG_P_DECIMAL_PLACES = globalThis.GWASMindConfig.get('plot.threshold_decimal_places');
const MANHATTAN_P_VALUE_FLOOR = globalThis.GWASMindConfig.get('plot.p_value_floor');
const INITIAL_MANHATTAN_THRESHOLD = Number((-Math.log10(MANHATTAN_GENOME_WIDE_P_VALUE))
    .toFixed(MANHATTAN_LOG_P_DECIMAL_PLACES));
const ManhattanPlot = {
    container: null,
    rawData: [],
    plotData: [],
    chrBoundaries: [],
    // Match the original plot by attempting scattergl first. Plotly's configured renderer
    // is the authoritative capability check; a standalone canvas probe can
    // report a false negative for the same browser.
    webglSupported: new URLSearchParams(window.location.search).get('renderer') !== 'svg',
    
    // State
    threshold: INITIAL_MANHATTAN_THRESHOLD,
    markerSize: 5,
    focusedChromosome: null,
    selectedRegion: null,
    selectedPointIndices: null,
    dragMode: 'select',
    selectedVariantKeys: new Set(),
    selectionGeometry: [],
    genomeOffsets: {},
    userAnnotations: [],
    
    // Multi-region selection support
    multiSelectRegions: [],
    ctrlPressed: false,
    
    // SNP highlighting for analysis linkage
    highlightedSnps: {
        uploaded: new Set(),   // Green - uploaded SNPs
        selected: new Set()    // Yellow - selected locus SNPs
    },
    
    // Double-click detection
    lastClickTime: 0,
    lastClickedChr: null,
    _eventsBound: false,
    _plotInitialized: false,
    _plotOperation: Promise.resolve(true),
    _renderGeneration: 0,
    _markerRefreshTimer: null,
    _thresholdRefreshPending: false,
    _renderPoints: null,
    _traceGeometry: null,
    _svgFallbackNoticeShown: false,
    _exportInProgress: false,
    activePanel: 'chat',
    _panelStates: new Map(),
    _panelOperation: Promise.resolve(true),
    _datasetGeneration: 0,
    maxSvgPoints: globalThis.GWASMindConfig.get('plot.max_svg_points'),
    svgBackgroundFraction: 0.4,
    
    /**
     * Initialize the Manhattan plot
     */
    init(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container) throw new Error(`Plot container #${containerId} was not found`);
        if (this.container && this.container !== container) {
            this._eventsBound = false;
            this._plotInitialized = false;
        }
        this.container = container;
        this._plotInitialized = Boolean(container._fullLayout);
        if (this.rawData !== data) {
            this._datasetGeneration += 1;
            this._panelStates.clear();
            this.focusedChromosome = null;
            this.selectedRegion = null;
            this.selectedPointIndices = null;
            this.multiSelectRegions = [];
            this.selectionGeometry = [];
            this.lassoSelections = [];
            this.selectedVariantKeys.clear();
            this._lastSelectionKey = null;
            this._lastLassoPathKey = null;
            this.userAnnotations = [];
            this.highlightedRegionIndex = undefined;
            this.highlightedSnps.uploaded.clear();
            this.highlightedSnps.selected.clear();
            this.lastClickTime = 0;
            this.lastClickedChr = null;
            const zoomButton = document.getElementById('zoom-out-btn');
            if (zoomButton) zoomButton.style.display = 'none';
            this.hideSelectionInfo();
        }
        this.rawData = data;
        this.processData();
        this.updateThresholdDisplay();
        return this.render();
    },

    emptyPanelState() {
        return { dataset: this.rawData, focusedChromosome: null, selectedRegion: null,
            multiSelectRegions: [], selectionGeometry: [], userAnnotations: [],
            uploaded: [], selected: [], highlightedRegionIndex: undefined,
            threshold: INITIAL_MANHATTAN_THRESHOLD, markerSize: 5, dragMode: 'select', view: null };
    },

    capturePanelState() {
        const clone = value => JSON.parse(JSON.stringify(value));
        const copyRegion = region => region ? { ...region, snps: region.snps.map(snp => ({ ...snp })),
            pointIndices: undefined } : null;
        const layout = this.container?.layout || {};
        const full = this.container?._fullLayout || layout;
        const view = { title: layout.title?.text || full.title?.text };
        for (const axis of ['xaxis', 'yaxis']) {
            const range = full[axis]?.range || layout[axis]?.range;
            if (Array.isArray(range) && range.length === 2 && range.every(Number.isFinite)) {
                view[axis] = { range: [...range], autorange: false };
            }
        }
        if (Array.isArray(layout.annotations)) view.annotations = clone(layout.annotations);
        return { dataset: this.rawData, focusedChromosome: this.focusedChromosome,
            selectedRegion: copyRegion(this.selectedRegion),
            multiSelectRegions: this.multiSelectRegions.map(copyRegion),
            selectionGeometry: clone(this.selectionGeometry), userAnnotations: clone(this.userAnnotations),
            uploaded: [...this.highlightedSnps.uploaded], selected: [...this.highlightedSnps.selected],
            highlightedRegionIndex: this.highlightedRegionIndex,
            threshold: this.threshold, markerSize: this.markerSize, dragMode: this.dragMode, view };
    },

    restorePanelState(saved) {
        const state = saved?.dataset === this.rawData ? saved : this.emptyPanelState();
        const clone = value => JSON.parse(JSON.stringify(value));
        this.focusedChromosome = state.focusedChromosome;
        this.selectedRegion = state.selectedRegion ? clone(state.selectedRegion) : null;
        this.multiSelectRegions = clone(state.multiSelectRegions);
        this.selectionGeometry = clone(state.selectionGeometry);
        this.lassoSelections = [];
        this.userAnnotations = clone(state.userAnnotations);
        this.highlightedSnps.uploaded = new Set(state.uploaded);
        this.highlightedSnps.selected = new Set(state.selected);
        this.highlightedRegionIndex = state.highlightedRegionIndex;
        this.threshold = state.threshold; this.markerSize = state.markerSize; this.dragMode = state.dragMode;
        this._lastSelectionKey = null; this._lastLassoPathKey = null;
        this.lastClickTime = 0; this.lastClickedChr = null; this.ctrlPressed = false;
        this.processData();
        this.remapSelectedIndices();
        this.updateThresholdDisplay();
        for (const id of ['threshold-input', 'threshold-range']) {
            const input = document.getElementById(id); if (input) input.value = this.threshold;
        }
        const marker = document.getElementById('marker-size'); if (marker) marker.textContent = this.markerSize;
        const zoom = document.getElementById('zoom-out-btn'); if (zoom) zoom.style.display = this.focusedChromosome ? 'inline-block' : 'none';
        document.querySelectorAll('.plot-tools button[data-tool]').forEach(button => {
            button.setAttribute('aria-pressed', String(button.dataset.tool === this.dragMode));
        });
        if (this.multiSelectRegions.length) this.showSelectionInfo(); else this.hideSelectionInfo();
        return this.render(state.view).then(async rendered => {
            if (rendered && this.highlightedRegionIndex !== undefined) await this.highlightPointsOnly(this.highlightedRegionIndex);
            return rendered;
        });
    },

    switchPanel(panel) {
        if (!['chat', 'analysis'].includes(panel)) return Promise.resolve(false);
        const generation = this._datasetGeneration;
        const task = this._panelOperation.catch(() => false).then(async () => {
            await this._plotOperation;
            if (generation !== this._datasetGeneration) return false;
            if (panel === this.activePanel) return true;
            // Report-only Auto history has no plotting dataset, but its tab
            // must still open. There is no plot state to borrow or restore.
            if (!this.container || !this.rawData.length) {
                this.activePanel = panel;
                return true;
            }
            // A delayed marker refresh must not repaint the next panel with the
            // previous panel's controls. Its current values are in the snapshot.
            if (this._markerRefreshTimer !== null) clearTimeout(this._markerRefreshTimer);
            this._markerRefreshTimer = null; this._thresholdRefreshPending = false;
            const previousPanel = this.activePanel;
            const previousState = this.capturePanelState();
            this._panelStates.set(previousPanel, previousState);
            this.activePanel = panel;
            const restored = await this.restorePanelState(this._panelStates.get(panel));
            if (!restored && generation === this._datasetGeneration) {
                this.activePanel = previousPanel;
                await this.restorePanelState(previousState);
            }
            return restored;
        });
        this._panelOperation = task;
        return task;
    },

    setPanelHighlights(panel, snpPositions, group) {
        if (!['chat', 'analysis'].includes(panel) || !['uploaded', 'selected'].includes(group)) return;
        if (panel === this.activePanel) return this.highlightSnps(snpPositions, '', group);
        const saved = this._panelStates.get(panel);
        const state = saved?.dataset === this.rawData ? saved : this.emptyPanelState();
        state[group] = [...this.highlightKeys(snpPositions)];
        this._panelStates.set(panel, state);
    },

    clearPanelHighlights(panel) {
        this.setPanelHighlights(panel, [], 'uploaded');
        return this.setPanelHighlights(panel, [], 'selected');
    },

    restorePanelSelection(panel, regions) {
        if (!['chat', 'analysis'].includes(panel)) return Promise.resolve(false);
        const generation = this._datasetGeneration;
        const task = this._panelOperation.catch(() => false).then(async () => {
            await this._plotOperation;
            if (generation !== this._datasetGeneration) return false;
            const state = this.emptyPanelState();
            state.multiSelectRegions = regions;
            state.selectedRegion = regions.at(-1) || null;
            const chromosomes = new Set(regions.flatMap(region => region.snps.map(snp => String(snp.chromosome))));
            state.focusedChromosome = chromosomes.size === 1 ? [...chromosomes][0] : null;
            this._panelStates.set(panel, state);
            return this.activePanel === panel ? this.restorePanelState(state) : true;
        });
        this._panelOperation = task;
        return task;
    },

    reportPlotError(context, error) {
        const detail = error instanceof Error ? error.message : String(error || 'Unknown error');
        console.error(`[ManhattanPlot] ${context}:`, error);
        let message = document.getElementById('manhattan-plot-error');
        if (!message) {
            message = document.createElement('div');
            message.id = 'manhattan-plot-error';
            message.setAttribute('role', 'alert');
            message.style.cssText = 'position:fixed;right:24px;bottom:24px;z-index:20000;max-width:440px;padding:12px 16px;border-radius:7px;background:#b42318;color:#fff;box-shadow:0 4px 14px rgba(0,0,0,.22);font-size:14px;';
            document.body.appendChild(message);
        }
        message.textContent = `${context}: ${detail}`;
        clearTimeout(this._plotErrorTimer);
        this._plotErrorTimer = setTimeout(() => message.remove(), 7000);
    },

    enqueuePlotOperation(context, operation) {
        const task = this._plotOperation
            .catch(() => false)
            .then(operation)
            .catch((error) => {
                this.reportPlotError(context, error);
                return false;
            });
        this._plotOperation = task;
        return task;
    },
    
    /**
     * Process raw GWAS data for plotting
     */
    processData() {
        if (!this.rawData || this.rawData.length === 0) {
            this.plotData = [];
            this.chrBoundaries = [];
            this.invalidateRenderCache();
            return;
        }
        
        // Group by chromosome
        const chrGroups = {};
        for (let i = 0; i < this.rawData.length; i++) {
            const snp = this.rawData[i];
            if (!snp || !snp.chromosome) continue;
            const chr = snp.chromosome.toString();
            if (!chrGroups[chr]) chrGroups[chr] = [];
            chrGroups[chr].push(snp);
        }
        
        // Sort chromosomes
        const chromosomes = Object.keys(chrGroups).sort((a, b) => {
            const numA = parseInt(a);
            const numB = parseInt(b);
            if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
            if (!isNaN(numA)) return -1;
            if (!isNaN(numB)) return 1;
            return a.localeCompare(b);
        });
        
        // Filter if focused on one chromosome
        // Selection geometry uses the genome-wide axis as a stable coordinate
        // system, including while only one chromosome is displayed.
        let genomeEnd = 0;
        this.genomeOffsets = {};
        for (const chr of chromosomes) {
            this.genomeOffsets[chr] = genomeEnd;
            let maxPosition = 0;
            for (const snp of chrGroups[chr]) maxPosition = Math.max(maxPosition, Number(snp.position));
            genomeEnd += maxPosition * 1.02;
        }
        this.syncSelectedVariants();
        const displayChromosomes = this.focusedChromosome 
            ? (chrGroups[this.focusedChromosome] ? [this.focusedChromosome] : chromosomes)
            : chromosomes;
        
        // Calculate cumulative positions
        let cumulativePos = 0;
        this.chrBoundaries = [];
        this.plotData = [];
        
        for (let i = 0; i < displayChromosomes.length; i++) {
            const chr = displayChromosomes[i];
            const chrData = chrGroups[chr];
            if (!chrData || chrData.length === 0) continue;
            
            // Find min/max positions
            let minPos = Infinity, maxPos = -Infinity;
            for (let j = 0; j < chrData.length; j++) {
                const pos = chrData[j].position;
                if (pos < minPos) minPos = pos;
                if (pos > maxPos) maxPos = pos;
            }
            
            const chrStart = this.focusedChromosome ? minPos : cumulativePos;
            const chrEnd = this.focusedChromosome ? maxPos : cumulativePos + maxPos;
            const chrCenter = (chrStart + chrEnd) / 2;
            
            this.chrBoundaries.push({ chr, start: chrStart, end: chrEnd, center: chrCenter });
            
            // Add points
            for (let j = 0; j < chrData.length; j++) {
                const snp = chrData[j];
                const x = this.focusedChromosome ? snp.position : cumulativePos + snp.position;

                // Reuse the parsed SNP record. Creating a second object graph
                // doubled retained memory and made every focus/reset pay
                // another allocation cost.
                const pvalue = this.scientificPValue(snp);
                const logP = pvalue === null ? Number(snp.neg_log_p) || 0
                    : -Math.log10(Math.max(pvalue, MANHATTAN_P_VALUE_FLOOR));
                const y = Number(logP.toFixed(MANHATTAN_LOG_P_DECIMAL_PLACES));
                snp.x = x;
                snp.y = y;
                snp.chromosome = chr;
                snp.snp_id = snp.snp_id || 'Unknown';
                snp.ref = snp.ref || 'N';
                snp.alt = snp.alt || 'N';
                snp.aboveThreshold = this.isAboveThreshold(snp);
                this.plotData.push(snp);
            }
            
            if (!this.focusedChromosome) {
                cumulativePos = chrEnd + maxPos * 0.02;
            }
        }

        this.invalidateRenderCache();
    },

    scientificPValue(snp) {
        const supplied = snp.pvalue;
        if (!['number', 'string'].includes(typeof supplied)
                || (typeof supplied === 'string' && !supplied.trim())) return null;
        const value = Number(supplied);
        return Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
    },

    isAboveThreshold(snp, threshold = this.threshold) {
        const pvalue = this.scientificPValue(snp);
        if (threshold === INITIAL_MANHATTAN_THRESHOLD && pvalue !== null) {
            return pvalue <= MANHATTAN_GENOME_WIDE_P_VALUE;
        }
        // User-adjusted display lines and older records lacking P retain the
        // existing coordinate-based interaction, without inventing a P value.
        return snp.y >= threshold;
    },

    invalidateRenderCache() {
        this._renderPoints = null;
        this._traceGeometry = null;
    },

    variantKey(snp) {
        return `${String(snp.chromosome).replace(/^chr/i, '').toUpperCase()}_${snp.position}_${String(snp.ref || 'N').toUpperCase()}_${String(snp.alt || 'N').toUpperCase()}`;
    },

    hasHighlight(group, snp) {
        return group.has(this.variantKey(snp))
            || group.has(`${String(snp.chromosome).replace(/^chr/i, '').toUpperCase()}_${snp.position}`);
    },

    syncSelectedVariants() {
        this.selectedVariantKeys = new Set(this.getAllSelectedSNPs().map(snp => this.variantKey(snp)));
    },

    remapSelectedIndices(points = this.getRenderPoints()) {
        const indices = [];
        if (this.selectedVariantKeys.size) {
            points.forEach((snp, index) => {
                if (this.selectedVariantKeys.has(this.variantKey(snp))) indices.push(index);
            });
        }
        this.selectedPointIndices = indices.length ? indices : null;
    },

    projectSelectionGeometry() {
        const focus = this.focusedChromosome;
        const offset = focus ? (this.genomeOffsets[focus] || 0) : 0;
        const boundary = this.chrBoundaries.find(item => item.chr === focus);
        const selections = [];
        for (const geometry of this.selectionGeometry) {
            if (focus && geometry.chromosome && geometry.chromosome !== focus) continue;
            const xs = geometry.x.map(x => x - offset);
            if (boundary && (Math.max(...xs) < 0 || Math.min(...xs) > boundary.end)) continue;
            const style = { line: { color: '#f1c40f', width: 2, dash: 'dot' } };
            if (geometry.type === 'rect') {
                selections.push({ ...style, type: 'rect', x0: xs[0], x1: xs[1], y0: geometry.y[0], y1: geometry.y[1] });
            } else {
                const path = xs.map((x, i) => `${i ? 'L' : 'M'}${x},${geometry.y[i]}`).join('') + 'Z';
                selections.push({ ...style, type: 'path', path });
            }
        }
        this.lassoSelections = selections;
        return selections;
    },

    /**
     * Limit the SVG fallback so a missing WebGL context cannot create tens of
     * thousands of DOM nodes. Genome-wide significant points are retained
     * first; the remaining capacity is filled by a deterministic systematic
     * sample of the chromosome-sorted background points.
     */
    chooseSvgFallbackPoints() {
        const points = this.plotData;
        const limit = Math.max(1, Math.floor(this.maxSvgPoints));
        if (points.length <= limit) return points;

        const priorityIndices = [];
        let backgroundCount = 0;
        for (let index = 0; index < points.length; index += 1) {
            if (points[index].aboveThreshold) priorityIndices.push(index);
            else backgroundCount += 1;
        }

        const reservedBackground = backgroundCount > 0
            ? Math.min(
                backgroundCount,
                Math.max(1, Math.floor(limit * this.svgBackgroundFraction))
            )
            : 0;
        const priorityCapacity = priorityIndices.length >= limit
            ? limit - reservedBackground
            : priorityIndices.length;
        let retainedIndices = priorityIndices.length <= priorityCapacity
            ? priorityIndices.slice()
            : priorityIndices
                .sort((left, right) => points[right].y - points[left].y || left - right)
                .slice(0, priorityCapacity);
        const backgroundCapacity = Math.min(
            backgroundCount,
            limit - retainedIndices.length
        );

        if (backgroundCapacity > 0) {
            let backgroundOrdinal = 0;
            let targetOffset = 0;
            let targetOrdinal = Math.floor(
                ((targetOffset + 0.5) * backgroundCount) / backgroundCapacity
            );
            for (let index = 0; index < points.length && targetOffset < backgroundCapacity; index += 1) {
                if (points[index].aboveThreshold) continue;
                if (backgroundOrdinal === targetOrdinal) {
                    retainedIndices.push(index);
                    targetOffset += 1;
                    targetOrdinal = Math.floor(
                        ((targetOffset + 0.5) * backgroundCount) / backgroundCapacity
                    );
                }
                backgroundOrdinal += 1;
            }
        }

        // A background SNP selected in a chromosome view must not disappear
        // when the genome-wide SVG sample is rebuilt. Keep it ahead of the
        // ordinary sample without changing the submitted selection.
        if (this.selectedVariantKeys.size || this.highlightedSnps.selected.size) {
            const selected = [];
            points.forEach((point, index) => {
                if (this.selectedVariantKeys.has(this.variantKey(point)) || this.hasHighlight(this.highlightedSnps.selected, point)) selected.push(index);
            });
            const selectedSet = new Set(selected);
            const others = retainedIndices.filter(index => !selectedSet.has(index));
            retainedIndices = selected.concat(others.slice(0, Math.max(0, limit - selected.length)));
        }
        retainedIndices.sort((left, right) => left - right);
        if (!this._svgFallbackNoticeShown) {
            console.warn(
                `[ManhattanPlot] WebGL is unavailable; the SVG fallback is limited to ${limit.toLocaleString('en-US')} displayed points.`
            );
            this._svgFallbackNoticeShown = true;
        }
        return retainedIndices.map((index) => points[index]);
    },

    getRenderPoints() {
        if (!this._renderPoints) {
            this._renderPoints = this.webglSupported
                ? this.plotData
                : this.chooseSvgFallbackPoints();
        }
        return this._renderPoints;
    },

    getTraceGeometry(points) {
        if (this._traceGeometry && this._traceGeometry.points === points) {
            return this._traceGeometry;
        }
        const count = points.length;
        const x = new Float64Array(count);
        const y = new Float64Array(count);
        const customdata = new Array(count);
        for (let index = 0; index < count; index += 1) {
            const point = points[index];
            x[index] = point.x;
            y[index] = point.y;
            customdata[index] = [
                point.snp_id,
                point.chromosome,
                point.position,
                point.ref,
                point.alt,
                point.pvalue
            ];
        }
        this._traceGeometry = { points, x, y, customdata };
        return this._traceGeometry;
    },
    
    /**
     * Build Plotly traces.
     */
    buildTraces() {
        if (!this.plotData || this.plotData.length === 0) return [];
        
        // Build chromosome index map
        const chrToIndex = {};
        for (let i = 0; i < this.chrBoundaries.length; i++) {
            chrToIndex[this.chrBoundaries[i].chr] = i;
        }
        
        const allPoints = this.getRenderPoints();
        const n = allPoints.length;
        this.remapSelectedIndices(allPoints);
        
        // Skip position-key allocation when no highlights exist.
        const hasSelected = this.highlightedSnps.selected.size > 0;
        const hasUploaded = this.highlightedSnps.uploaded.size > 0;
        const hasHighlights = hasSelected || hasUploaded;
        
        const baseSize = this.markerSize;
        const thresholdSize = baseSize + 2;
        const geometry = this.getTraceGeometry(allPoints);
        const colors = new Array(n);
        const sizes = new Float32Array(n);
        const opacities = new Float32Array(n);
        
        for (let i = 0; i < n; i++) {
            const d = allPoints[i];
            let color = 'red';
            let size = d.aboveThreshold ? thresholdSize : baseSize;
            let opacity = d.y >= 3 ? 0.7 : 0.4;
            
            if (this.selectedVariantKeys.size && this.selectedVariantKeys.has(this.variantKey(d))) {
                color = '#f1c40f';
                size = baseSize + 4;
                opacity = 1;
            } else if (hasHighlights) {
                if (hasSelected && this.hasHighlight(this.highlightedSnps.selected, d)) {
                    color = '#f1c40f';
                    size = baseSize + 4;
                    opacity = 1.0;
                } else if (hasUploaded && this.hasHighlight(this.highlightedSnps.uploaded, d)) {
                    color = '#2ecc71';
                    size = thresholdSize;
                    opacity = 1.0;
                } else if (!d.aboveThreshold) {
                    color = (chrToIndex[d.chromosome] || 0) % 2 === 0 ? '#808080' : '#b0b0b0';
                }
            } else if (!d.aboveThreshold) {
                color = (chrToIndex[d.chromosome] || 0) % 2 === 0 ? '#808080' : '#b0b0b0';
            }
            
            colors[i] = color;
            sizes[i] = size;
            opacities[i] = opacity;
        }
        
        return [{
            x: geometry.x,
            y: geometry.y,
            type: this.webglSupported ? 'scattergl' : 'scatter',
            mode: 'markers',
            marker: {
                color: colors,
                size: sizes,
                opacity: opacities,
                line: { width: 0 }
            },
            customdata: geometry.customdata,
            hovertemplate: 
                '<b>%{customdata[0]}</b><br>' +
                'Chr: %{customdata[1]}<br>' +
                'Pos: %{customdata[2]}<br>' +
                'Ref: %{customdata[3]}<br>' +
                'Alt: %{customdata[4]}<br>' +
                '-log10(p): %{y:.2f}' +
                '<extra></extra>',
            selectedpoints: this.selectedPointIndices,
            selected: { marker: { color: '#f1c40f', opacity: 1 } },
            showlegend: false
        }];
    },
    
    /**
     * Build Plotly layout
     */
    buildLayout() {
        // Calculate X range
        let xRange = [0, 1];
        if (this.focusedChromosome && this.chrBoundaries.length > 0) {
            const chrBound = this.chrBoundaries.find(b => b.chr === this.focusedChromosome);
            if (chrBound) {
                xRange = [0, chrBound.end * 1.02];
            }
        } else if (this.chrBoundaries.length > 0) {
            xRange = [0, this.chrBoundaries[this.chrBoundaries.length - 1].end * 1.02];
        }
        
        // Calculate Y range
        let maxY = 10;
        if (this.plotData.length > 0) {
            maxY = this.plotData.reduce((max, d) => d.y > max ? d.y : max, 0);
        }
        const yRange = [0, maxY * 1.1];
        
        return {
            title: {
                text: this.focusedChromosome 
                    ? `Chromosome ${this.focusedChromosome}` 
                    : 'GWAS Manhattan Plot - Double-click chromosome to zoom (Shift+drag for multi-select)',
                font: { size: 18 }
            },
            xaxis: {
                title: this.focusedChromosome ? 'Position (bp)' : 'Chromosome',
                showgrid: false,
                zeroline: false,
                tickmode: this.focusedChromosome ? 'auto' : 'array',
                tickvals: this.focusedChromosome ? undefined : this.chrBoundaries.map(b => b.center),
                ticktext: this.focusedChromosome ? undefined : this.chrBoundaries.map(b => b.chr),
                tickformat: this.focusedChromosome ? ',.0f' : undefined,
                tickangle: 0,
                range: xRange,
                autorange: false
            },
            yaxis: {
                title: '-log₁₀(p-value)',
                showgrid: true,
                gridcolor: '#e0e0e0',
                range: yRange,
                autorange: false
            },
            shapes: [{
                type: 'line',
                x0: 0, x1: 1,
                xref: 'paper',
                y0: this.threshold, y1: this.threshold,
                line: { color: 'red', width: 2, dash: 'dash' }
            }],
            annotations: this.buildAnnotations(),
            hovermode: 'closest',
            dragmode: this.dragMode,
            selections: this.projectSelectionGeometry(),
            selectdirection: 'any',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            // height: 600,  <-- Removed to allow auto-sizing
            autosize: true,  // <-- Added
            margin: { l: 80, r: 30, t: 60, b: 70 }, // Adjusted margins
            showlegend: false
        };
    },
    
    /**
     * Build user annotations
     */
    buildAnnotations() {
        return this.userAnnotations.map(anno => {
            let plotX = anno.x;
            let visible = true;
            
            if (anno.chromosome) {
                if (this.focusedChromosome) {
                    visible = anno.chromosome === this.focusedChromosome;
                } else {
                    const boundary = this.chrBoundaries.find(b => b.chr === anno.chromosome);
                    if (boundary) {
                        plotX = boundary.start + anno.x;
                    } else {
                        visible = false;
                    }
                }
            }
            
            return {
                name: anno.id,
                x: plotX,
                y: anno.y,
                text: anno.text,
                visible,
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#333',
                ax: 0, ay: -40,
                bgcolor: '#ffffff',
                bordercolor: '#333',
                borderwidth: 1,
                borderpad: 6,
                captureevents: true,
                font: { size: 12, color: '#000' }
            };
        });
    },

    hasWebglRenderFailure(traces) {
        const usesWebgl = Array.isArray(traces) && traces.some(trace => trace.type === 'scattergl');
        return Boolean(
            usesWebgl &&
            this.container &&
            typeof this.container.querySelector === 'function' &&
            this.container.querySelector('.no-webgl')
        );
    },

    /**
     * Render the plot
     */
    render(view = null) {
        if (!this.container) return Promise.resolve(false);
        const generation = ++this._renderGeneration;
        const traces = this.buildTraces();
        const layout = this.buildLayout();
        if (view) {
            for (const axis of ['xaxis', 'yaxis']) if (view[axis]) Object.assign(layout[axis], view[axis]);
            if (view.title) layout.title.text = view.title;
            if (view.annotations) layout.annotations = view.annotations;
        }
        const config = {
            displayModeBar: false,
            displaylogo: false,
            locale: 'en',
            modeBarButtonsToRemove: ['select2d', 'autoScale2d', 'resetScale2d'],
            scrollZoom: true,
            doubleClick: false,
            responsive: true,
            edits: { annotationPosition: true },
            toImageButtonOptions: {
                format: 'png',
                filename: 'manhattan_plot',
                height: 1200,
                width: 1800,
                scale: 2
            }
        };
        
        return this.enqueuePlotOperation('Unable to render the Manhattan plot', async () => {
            const method = this._plotInitialized ? 'react' : 'newPlot';
            let renderedTraces = traces;
            try {
                await Plotly[method](this.container, renderedTraces, layout, config);
            } catch (error) {
                const rejectedWebgl = renderedTraces.some((trace) => trace.type === 'scattergl');
                if (!rejectedWebgl) throw error;
                this.webglSupported = false;
                this.invalidateRenderCache();
                renderedTraces = this.buildTraces();
                // Retry exactly once. A rejected SVG render propagates to the
                // serialized operation's normal error reporter.
                await Plotly.react(this.container, renderedTraces, layout, config);
            }
            if (this.hasWebglRenderFailure(renderedTraces)) {
                this.webglSupported = false;
                this.invalidateRenderCache();
                renderedTraces = this.buildTraces();
                await Plotly.react(this.container, renderedTraces, layout, config);
            }
            this._plotInitialized = true;
            this.bindEvents();
            if (generation === this._renderGeneration) {
                const loadingEl = document.getElementById('plot-loading');
                if (loadingEl) loadingEl.style.display = 'none';
            }
            return true;
        });
    },
    
    /**
     * Bind plot events
     */
    bindEvents() {
        if (this._eventsBound) return;
        if (!this.container || typeof this.container.on !== 'function') {
            throw new Error('Plotly event API is unavailable');
        }
        this._eventsBound = true;
        // Track Shift because it works consistently across supported browsers.
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Shift' || e.shiftKey) {
                this.ctrlPressed = true;
            }
        });
        document.addEventListener('keyup', (e) => {
            if (e.key === 'Shift') {
                this.ctrlPressed = false;
            }
        });
        
        // Read Shift state at the time Plotly emits the selection.
        this.container.on('plotly_selected', (eventData) => {
            this.saveLassoSelection(eventData, this.ctrlPressed);
            this.handleSelection(eventData, this.ctrlPressed);
        });
        
        // Capture Shift on pointer down as a fallback for event ordering.
        this.container.addEventListener('mousedown', (e) => {
            if (e.shiftKey) {
                this.ctrlPressed = true;
            }
        });
        this.container.addEventListener('mouseup', () => {
            // Reset after Plotly has emitted its selection event.
            setTimeout(() => {
                this.ctrlPressed = false;
            }, 100);
        });
        
        // Click event (for double-click zoom)
        this.container.on('plotly_click', (eventData) => {
            this.handleClick(eventData);
        });
        
        // Annotation click
        this.container.on('plotly_clickannotation', (eventData) => {
            this.handleAnnotationClick(eventData);
        });
    },
    
    /**
     * Preserve box/lasso geometry independently of the current viewport.
     */
    saveLassoSelection(eventData, isMultiSelect = false) {
        if (!eventData) return;
        if (Array.isArray(eventData.points) && eventData.points.length === 0) return;
        
        const coordinates = eventData.lassoPoints || eventData.range;
        const type = eventData.lassoPoints ? 'path' : 'rect';
        const minimum = type === 'path' ? 3 : 2;
        if (!coordinates?.x || !coordinates.y || coordinates.x.length < minimum ||
            coordinates.x.length !== coordinates.y.length ||
            ![...coordinates.x, ...coordinates.y].every(Number.isFinite)) return;
        const pathKey = `${this.focusedChromosome}:${isMultiSelect}:${type}:${JSON.stringify(coordinates)}`;
        if (this._lastLassoPathKey === pathKey) {
            return;
        }
        this._lastLassoPathKey = pathKey;
        
        setTimeout(() => {
            if (this._lastLassoPathKey === pathKey) {
                this._lastLassoPathKey = null;
            }
        }, 3000);
        
        const offset = this.focusedChromosome ? (this.genomeOffsets[this.focusedChromosome] || 0) : 0;
        const geometry = {
            type, chromosome: this.focusedChromosome,
            x: coordinates.x.map(x => x + offset), y: [...coordinates.y]
        };
        this.selectionGeometry = isMultiSelect ? [...this.selectionGeometry, geometry] : [geometry];
        const currentSelections = this.projectSelectionGeometry();

        // Plotly already displays the active lasso. Avoid a costly relayout for
        // the common single-selection path; only merge paths for Shift-select.
        if (isMultiSelect) {
            this.enqueuePlotOperation('Unable to preserve the lasso selection', () => Plotly.relayout(this.container, {
                selections: currentSelections
            }));
        }
    },
    
    /**
     * Clear every saved lasso path.
     */
    clearLassoSelections() {
        this.lassoSelections = [];
        this.selectionGeometry = [];
        if (this.container && this.container.layout) {
            this.enqueuePlotOperation('Unable to clear the lasso selection', () => (
                Plotly.relayout(this.container, { selections: [] })
            ));
        }
    },
    
    /**
     * Handle single- and Shift-multi-region SNP selection.
     */
    handleSelection(eventData, isCtrlSelect = false) {
        if (!eventData || !eventData.points || eventData.points.length === 0) return;
        
        // Plotly may emit the same selection again after a redraw, with entirely
        // different trace indices. Compare stable allelic identities first.
        const eventKeys = new Set(eventData.points.map(point => {
            const cd = point.customdata || [];
            return this.variantKey(Array.isArray(cd)
                ? { chromosome: cd[1], position: cd[2], ref: cd[3], alt: cd[4] }
                : cd);
        }));
        if (eventKeys.size === this.selectedVariantKeys.size &&
            [...eventKeys].every(key => this.selectedVariantKeys.has(key))) {
            this.remapSelectedIndices();
            return;
        }

        // Bound duplicate native events without conflating different chromosomes.
        let selectionHash = 2166136261;
        for (const key of eventKeys) {
            for (let i = 0; i < key.length; i++) {
                selectionHash ^= key.charCodeAt(i);
                selectionHash = Math.imul(selectionHash, 16777619);
            }
        }
        const selectionKey = `${isCtrlSelect || this.ctrlPressed}:${eventKeys.size}:${selectionHash >>> 0}`;
        if (this._lastSelectionKey === selectionKey) {
            return;
        }
        this._lastSelectionKey = selectionKey;
        
        // Permit selecting the same region again after a short debounce window.
        setTimeout(() => {
            if (this._lastSelectionKey === selectionKey) {
                this._lastSelectionKey = null;
            }
        }, 3000);
        
        const useMultiSelect = isCtrlSelect || this.ctrlPressed;
        
        const existingKeys = useMultiSelect
            ? this.selectedVariantKeys
            : null;
        const indices = [];
        const selectedSNPs = [];
        const chromosomes = new Set();
        let start = Infinity;
        let end = -Infinity;
        for (const point of eventData.points) {
            const pointIndex = Number(point.pointIndex ?? point.pointNumber);
            if (!Number.isInteger(pointIndex)) continue;
            const p = point;
            const cd = p.customdata || [];
            const snp = Array.isArray(cd)
                ? {
                    snp_id: cd[0],
                    chromosome: cd[1],
                    position: cd[2],
                    ref: cd[3],
                    alt: cd[4],
                    pvalue: cd[5],
                    neg_log_p: p.y
                }
                : {
                chromosome: cd.chromosome,
                position: cd.position,
                snp_id: cd.snp_id,
                pvalue: cd.pvalue,
                neg_log_p: cd.neg_log_p,
                ref: cd.ref,
                alt: cd.alt
            };
            if (!snp.chromosome || !Number.isFinite(Number(snp.position))) continue;
            if (existingKeys && existingKeys.has(this.variantKey(snp))) continue;
            snp.position = Number(snp.position);
            indices.push(pointIndex);
            selectedSNPs.push(snp);
            chromosomes.add(String(snp.chromosome));
            if (snp.position < start) start = snp.position;
            if (snp.position > end) end = snp.position;
        }
        
        if (selectedSNPs.length === 0) return;

        // A new plot locus invalidates any in-flight answer for the old context.
        if (this.activePanel === 'chat' && typeof window.cancelActiveChatRequest === 'function') {
            window.cancelActiveChatRequest();
        }
        
        const chromosomeValues = [...chromosomes];
        const chromosome = chromosomeValues.length === 1
            ? chromosomeValues[0]
            : chromosomeValues.join(',');
        
        const newRegion = { chromosome, start, end, snps: selectedSNPs, pointIndices: indices };
        
        if (useMultiSelect && this.multiSelectRegions.length > 0) {
            this.multiSelectRegions.push(newRegion);
            this.selectedPointIndices = (this.selectedPointIndices || []).concat(indices);
            
            this.addSystemMessage(`➕ Added ${selectedSNPs.length} SNPs in chr${chromosome}:${start.toLocaleString()}-${end.toLocaleString()}`);
        } else {
            this.multiSelectRegions = [newRegion];
            this.selectedPointIndices = indices;
            this.addSystemMessage(`✅ Selected ${selectedSNPs.length} SNPs in chr${chromosome}:${start.toLocaleString()}-${end.toLocaleString()}`);
        }
        
        this.selectedRegion = newRegion;
        this.syncSelectedVariants();
        this.remapSelectedIndices();
        this.updateHighlighting();
        
        // Update UI
        this.showSelectionInfo();
    },
    
    /**
     * Highlight all points selected across multiple regions.
     */
    highlightAllSelected() {
        this.remapSelectedIndices();
        if (!this.selectedPointIndices || this.selectedPointIndices.length === 0) return;
        
        return this.enqueuePlotOperation('Unable to highlight selected SNPs', () => Plotly.restyle(this.container, {
            selectedpoints: [this.selectedPointIndices],
            'selected.marker.color': '#f1c40f',
            'selected.marker.opacity': 1
        }));
    },
    
    /**
     * Get all selected SNPs from all regions
     */
    getAllSelectedSNPs() {
        if (this.multiSelectRegions.length === 0) return [];
        return this.multiSelectRegions.flatMap(r => r.snps);
    },
    
    /**
     * Handle plot click (double-click for zoom)
     */
    handleClick(eventData) {
        const currentTime = Date.now();
        
        // Determine clicked chromosome
        let clickedChr = null;
        
        if (eventData && eventData.points && eventData.points.length > 0) {
            const cd = eventData.points[0].customdata;
            if (Array.isArray(cd)) {
                clickedChr = cd[1];
            } else {
                clickedChr = cd?.chromosome;
            }
        } else if (!this.focusedChromosome && eventData && eventData.xvals && eventData.xvals.length > 0) {
            const xPos = eventData.xvals[0];
            for (const boundary of this.chrBoundaries) {
                if (xPos >= boundary.start && xPos <= boundary.end) {
                    clickedChr = boundary.chr;
                    break;
                }
            }
        }
        
        if (clickedChr) {
            // Double-click detection
            if (this.lastClickedChr === clickedChr && currentTime - this.lastClickTime < 500) {
                if (this.focusedChromosome === clickedChr) {
                    this.zoomOut();
                } else {
                    this.focusChromosome(clickedChr);
                }
                this.lastClickTime = 0;
                this.lastClickedChr = null;
            } else {
                this.lastClickTime = currentTime;
                this.lastClickedChr = clickedChr;
            }
        } else if (this.focusedChromosome && currentTime - this.lastClickTime < 500) {
            this.zoomOut();
            this.lastClickTime = 0;
        } else {
            this.lastClickTime = currentTime;
        }
    },
    
    /**
     * Handle annotation click (edit/delete)
     */
    handleAnnotationClick(eventData) {
        if (!eventData.annotation || !eventData.annotation.name) return;
        
        const annoId = eventData.annotation.name;
        const currentText = eventData.annotation.text.replace(/<br>/g, ';');
        
        const newText = prompt('Edit annotation (use ; for line breaks, clear to delete):', currentText);
        if (newText === null) return;
        
        if (newText.trim() === '') {
            this.userAnnotations = this.userAnnotations.filter(a => a.id !== annoId);
        } else {
            const formattedText = newText.replace(/;/g, '<br>');
            this.userAnnotations = this.userAnnotations.map(a => 
                a.id === annoId ? { ...a, text: formattedText } : a
            );
        }
        
        return this.render();
    },
    
    /**
     * Focus on a specific chromosome
     */
    focusChromosome(chr) {
        this.focusedChromosome = String(chr);
        this.processData();
        const result = this.render();
        
        document.getElementById('zoom-out-btn').style.display = 'inline-block';
        return result;
    },
    
    /**
     * Zoom out to all chromosomes
     */
    zoomOut() {
        this.focusedChromosome = null;
        this.processData();
        const result = this.render();
        
        document.getElementById('zoom-out-btn').style.display = 'none';
        return result;
    },
    
    /**
     * Reset axes while retaining saved selections.
     */
    resetAxes() {
        this.processData();
        return this.render();
    },
    
    /**
     * Restore a saved selection state.
     */
    restoreSelections(savedSelections, savedRegions, savedIndices) {
        // Restore the data model before the visual state.
        this.lassoSelections = savedSelections;
        this.multiSelectRegions = savedRegions;
        this.syncSelectedVariants();
        this.remapSelectedIndices();
        if (this.selectionGeometry.length) savedSelections = this.projectSelectionGeometry();
        savedIndices = this.selectedPointIndices;
        
        // Restore the selection UI and lasso paths.
        if (savedRegions.length > 0) {
            this.showSelectionInfo();
        }
        return this.enqueuePlotOperation('Unable to restore plot selections', async () => {
            if (savedSelections.length > 0) {
                await Plotly.relayout(this.container, { selections: savedSelections });
            }
            if (savedIndices && savedIndices.length > 0) {
                await Plotly.restyle(this.container, { selectedpoints: [savedIndices] });
            }
            return true;
        });
    },
    
    /**
     * Clear every selected region.
     */
    clearSelection() {
        if (this.activePanel === 'chat' && typeof window.cancelActiveChatRequest === 'function') {
            window.cancelActiveChatRequest();
        }
        this.selectedRegion = null;
        this.selectedPointIndices = null;
        this.multiSelectRegions = [];
        this.lassoSelections = [];
        this.selectionGeometry = [];
        this.selectedVariantKeys.clear();
        this._lastSelectionKey = null;
        this._lastLassoPathKey = null;
        
        this.enqueuePlotOperation('Unable to clear plot selections', async () => {
            await Plotly.restyle(this.container, { selectedpoints: [null] });
            await Plotly.relayout(this.container, { selections: [] });
            return true;
        });
        this.updateHighlighting();
        
        this.hideSelectionInfo();
        this.addSystemMessage('✖ Cleared all selections');
    },

    scheduleMarkerRefresh(updateThresholdLine = false) {
        this._thresholdRefreshPending = this._thresholdRefreshPending || updateThresholdLine;
        if (this._markerRefreshTimer !== null) clearTimeout(this._markerRefreshTimer);
        this._markerRefreshTimer = setTimeout(() => {
            this._markerRefreshTimer = null;
            const refreshThreshold = this._thresholdRefreshPending;
            this._thresholdRefreshPending = false;
            const markersUpdated = this.updateHighlighting();
            if (refreshThreshold) {
                Promise.resolve(markersUpdated).then(() => this.enqueuePlotOperation(
                    'Unable to update the significance threshold',
                    () => Plotly.relayout(this.container, {
                        'shapes[0].y0': this.threshold,
                        'shapes[0].y1': this.threshold
                    })
                ));
            }
        }, 60);
    },
    
    /**
     * Set threshold
     */
    setThreshold(value) {
        this.threshold = value;
        this.plotData.forEach(d => {
            d.aboveThreshold = this.isAboveThreshold(d, value);
        });
        if (this.webglSupported) {
            this.scheduleMarkerRefresh(true);
        } else {
            this.invalidateRenderCache();
            if (this._markerRefreshTimer !== null) clearTimeout(this._markerRefreshTimer);
            this._markerRefreshTimer = setTimeout(() => {
                this._markerRefreshTimer = null;
                this.render();
            }, 60);
        }
        this.updateThresholdDisplay();
    },
    
    /**
     * Change marker size
     */
    changeMarkerSize(delta) {
        this.markerSize = Math.max(2, Math.min(12, this.markerSize + delta));
        document.getElementById('marker-size').textContent = this.markerSize;
        return this.scheduleMarkerRefresh();
    },
    
    /**
     * Add annotation at selected region
     */
    addAnnotation() {
        if (!this.selectedRegion) return;
        
        const text = prompt('Enter label text (use ; for line breaks):');
        if (!text) return;
        
        const formattedText = text.replace(/;/g, '<br>');
        const centerPos = (this.selectedRegion.start + this.selectedRegion.end) / 2;
        
        let maxY = this.threshold;
        if (this.selectedRegion.snps && this.selectedRegion.snps.length > 0) {
            maxY = this.selectedRegion.snps.reduce((max, s) => (s.neg_log_p || 0) > max ? s.neg_log_p : max, 0);
        }
        
        this.userAnnotations.push({
            chromosome: this.selectedRegion.chromosome,
            x: centerPos,
            y: maxY + 1,
            text: formattedText,
            id: Date.now().toString()
        });
        
        return this.render();
    },
    
    /**
     * Show clickable selection summaries for one or more regions.
     */
    showSelectionInfo() {
        const infoDiv = document.getElementById('selection-info');
        const textSpan = document.getElementById('selection-text');
        const quickPrompt = document.getElementById('quick-prompt');
        const quickPromptBtn = document.getElementById('quick-prompt-btn');
        
        if (this.multiSelectRegions.length > 0) {
            const totalSNPs = this.getAllSelectedSNPs().length;
            textSpan.replaceChildren();
            this.multiSelectRegions.forEach((region, index) => {
                const item = document.createElement('div');
                item.className = 'multi-region-item clickable';
                item.dataset.action = 'highlight-card-left';
                item.dataset.regionIndex = String(index);
                item.textContent = `✅ Selected ${region.snps.length} SNPs in region chr${region.chromosome}:${region.start.toLocaleString()}-${region.end.toLocaleString()}`;
                textSpan.appendChild(item);
            });
            
            infoDiv.style.display = 'flex';
            quickPromptBtn.textContent = window.GWASMIND_UI_PROMPTS.describe_selected
                .replace('$count', String(totalSNPs));
            quickPrompt.style.display = 'block';
        }
    },
    
    /**
     * Hide selection info UI
     */
    hideSelectionInfo() {
        document.getElementById('selection-info').style.display = 'none';
        document.getElementById('quick-prompt').style.display = 'none';
    },
    
    /**
     * Update threshold display
     */
    updateThresholdDisplay() {
        const label = document.getElementById('threshold-pvalue');
        if (!label) return;
        const pvalue = (this.threshold === INITIAL_MANHATTAN_THRESHOLD
            ? MANHATTAN_GENOME_WIDE_P_VALUE : Math.pow(10, -this.threshold)).toExponential(2);
        label.textContent = `(p = ${pvalue})`;
    },
    
    /**
     * Add system message to chat
     */
    addSystemMessage(content) {
        if (this.activePanel !== 'chat') return;
        const messagesDiv = document.getElementById('chat-messages');
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system';
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = String(content ?? '');
        msgDiv.appendChild(messageContent);
        messagesDiv.appendChild(msgDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    },
    
    /**
     * Return one selected region or a merged multi-region view.
     */
    getSelectedRegion() {
        if (this.multiSelectRegions.length === 0) return null;
        
        // Preserve the original single-region shape.
        if (this.multiSelectRegions.length === 1) {
            return this.multiSelectRegions[0];
        }
        
        // Merge all region summaries for compatibility with chat requests.
        const allSNPs = this.getAllSelectedSNPs();
        const chromosomes = [...new Set(this.multiSelectRegions.map(r => r.chromosome))];
        
        return {
            chromosome: chromosomes.join(','),
            start: Math.min(...this.multiSelectRegions.map(r => r.start)),
            end: Math.max(...this.multiSelectRegions.map(r => r.end)),
            snps: allSNPs,
            regions: this.multiSelectRegions
        };
    },
    
    /**
     * Get all multi-select regions
     */
    getMultiSelectRegions() {
        return this.multiSelectRegions;
    },
    
    /**
     * Get variant IDs for every selected region.
     */
    getVariantIds() {
        const allSNPs = this.getAllSelectedSNPs();
        if (allSNPs.length === 0) return [];
        
        return allSNPs.map(snp => {
            const chr = snp.chromosome.toString().replace('chr', '');
            const ref = snp.ref?.toUpperCase();
            const alt = snp.alt?.toUpperCase();
            
            if (ref && alt) {
                return `${chr}_${snp.position}_${ref}_${alt}`;
            }
            return snp.snp_id || `chr${chr}:${snp.position}`;
        }).filter(id => id);
    },
    
    /**
     * Highlight one selected region without drawing another border.
     */
    highlightRegionPoints(index) {
        // Ignore requests made before plot data is ready.
        if (!this.plotData || this.plotData.length === 0) {
            return;
        }
        
        if (index >= this.multiSelectRegions.length) {
            return;
        }
        
        const region = this.multiSelectRegions[index];
        if (!region || !region.snps) {
            return;
        }
        
        // Store the active region for linked cards.
        this.highlightedRegionIndex = index;
        
        // Restyle marker arrays without rebuilding axes or event handlers.
        this.highlightPointsOnly(index);
    },
    
    /**
     * Highlight points without rebuilding the plot.
     */
    highlightPointsOnly(index, savedRegion = null) {
        const region = savedRegion || this.multiSelectRegions[index];
        if (!region || !region.snps) return;
        
        // Build an O(1) lookup for highlighted genomic positions.
        const highlightPositions = new Set();
        region.snps.forEach(snp => {
            highlightPositions.add(this.variantKey(snp));
        });
        
        // Build chromosome index map
        const chrToIndex = {};
        this.chrBoundaries.forEach((b, idx) => chrToIndex[b.chr] = idx);
        
        // Build marker updates only for the displayed points.
        const colors = [];
        const sizes = [];
        
        this.getRenderPoints().forEach(d => {
            const isHighlighted = highlightPositions.has(this.variantKey(d));
            
            if (isHighlighted) {
                colors.push('#ffc107');
                sizes.push(this.markerSize + 8);
            } else {
                // Dim points outside the linked region.
                colors.push(d.aboveThreshold ? 'rgba(255,100,100,0.3)' : 'rgba(150,150,150,0.25)');
                sizes.push(d.aboveThreshold ? this.markerSize + 2 : this.markerSize);
            }
        });
        
        // Update colors and sizes without rebuilding the plot.
        return this.enqueuePlotOperation('Unable to highlight the selected region', () => Plotly.restyle(this.container, {
            'marker.color': [colors],
            'marker.size': [sizes],
            selectedpoints: [null]
        }));
    },
    
    /**
     * Re-render a linked-region highlight while retaining lasso support.
     */
    renderWithHighlight() {
        const traces = this.buildTracesWithHighlight();
        const layout = this.buildLayout();
        
        // Keep lasso as the active drag mode.
        layout.dragmode = this.dragMode;
        
        // Restore previously saved lasso paths.
        if (this.lassoSelections && this.lassoSelections.length > 0) {
            layout.selections = this.lassoSelections;
        }
        
        const config = {
            responsive: true,
            displayModeBar: false,
            locale: 'en',
            modeBarButtonsToAdd: ['select2d', 'lasso2d'],
            modeBarButtonsToRemove: ['autoScale2d'],
            scrollZoom: true,
            toImageButtonOptions: {
                format: 'png',
                filename: 'manhattan_plot',
                height: 800,
                width: 1800,
                scale: 2
            }
        };
        
        return this.enqueuePlotOperation('Unable to update the highlighted region', async () => {
            await Plotly.react(this.container, traces, layout, config);
            this._plotInitialized = true;
            this.bindEvents();
            return true;
        });
    },
    
    /**
     * Build one trace with marker styling for the linked region.
     */
    buildTracesWithHighlight() {
        if (!this.plotData || this.plotData.length === 0) return [];
        
        const highlightIndex = this.highlightedRegionIndex;
        const highlightRegion = highlightIndex !== undefined ? this.multiSelectRegions[highlightIndex] : null;
        
        // Build an O(1) lookup for highlighted genomic positions.
        const highlightPositions = new Set();
        if (highlightRegion && highlightRegion.snps) {
            for (let i = 0; i < highlightRegion.snps.length; i++) {
                const snp = highlightRegion.snps[i];
                highlightPositions.add(this.variantKey(snp));
            }
        }
        const hasHighlight = highlightPositions.size > 0;
        
        // Build chromosome index map
        const chrToIndex = {};
        for (let i = 0; i < this.chrBoundaries.length; i++) {
            chrToIndex[this.chrBoundaries[i].chr] = i;
        }
        
        const allPoints = this.getRenderPoints();
        const n = allPoints.length;
        
        const geometry = this.getTraceGeometry(allPoints);
        const colors = new Array(n);
        const sizes = new Float32Array(n);
        const opacities = new Float32Array(n);
        
        for (let i = 0; i < n; i++) {
            const d = allPoints[i];
            const isHighlighted = highlightPositions.has(this.variantKey(d));
            
            if (isHighlighted || this.selectedVariantKeys.has(this.variantKey(d))) {
                colors[i] = '#ffc107';
                sizes[i] = this.markerSize + 8;
                opacities[i] = 1.0;
            } else if (hasHighlight) {
                colors[i] = d.aboveThreshold ? 'rgba(255,100,100,0.3)' : 'rgba(150,150,150,0.25)';
                sizes[i] = d.aboveThreshold ? this.markerSize + 2 : this.markerSize;
                opacities[i] = d.y >= 3 ? 0.3 : 0.2;
            } else {
                if (d.aboveThreshold) {
                    colors[i] = 'red';
                } else {
                    const chrIdx = chrToIndex[d.chromosome] || 0;
                    colors[i] = chrIdx % 2 === 0 ? '#808080' : '#b0b0b0';
                }
                sizes[i] = d.aboveThreshold ? this.markerSize + 2 : this.markerSize;
                opacities[i] = d.y >= 3 ? 0.7 : 0.4;
            }
        }
        
        return [{
            x: geometry.x,
            y: geometry.y,
            type: this.webglSupported ? 'scattergl' : 'scatter',
            mode: 'markers',
            marker: {
                color: colors,
                size: sizes,
                opacity: opacities,
                line: { width: 0 }
            },
            customdata: geometry.customdata,
            hovertemplate: 
                '<b>%{customdata[0]}</b><br>' +
                'Chr: %{customdata[1]}<br>' +
                'Pos: %{customdata[2]}<br>' +
                'Ref: %{customdata[3]}<br>' +
                'Alt: %{customdata[4]}<br>' +
                '-log10(p): %{y:.2f}' +
                '<extra></extra>',
            showlegend: false
        }];
    },
    
    /**
     * Restore the standard view of every selected point.
     */
    showAllSelectedPoints() {
        this.clearHighlight();
    },
    
    /**
     * Clear linked-region styling and restore normal markers.
     */
    clearHighlight() {
        this.highlightedRegionIndex = undefined;
        
        // Ignore requests made before plot data is ready.
        if (!this.plotData || this.plotData.length === 0) return;
        
        // Build chromosome index map
        const chrToIndex = {};
        for (let i = 0; i < this.chrBoundaries.length; i++) {
            chrToIndex[this.chrBoundaries[i].chr] = i;
        }
        
        // Restore the standard color, size, and opacity arrays.
        const renderPoints = this.getRenderPoints();
        const n = renderPoints.length;
        const colors = new Array(n);
        const sizes = new Float32Array(n);
        const opacities = new Float32Array(n);
        
        // Account for uploaded and selected SNP highlight groups.
        const hasSelected = this.highlightedSnps.selected.size > 0;
        const hasUploaded = this.highlightedSnps.uploaded.size > 0;
        const hasHighlights = hasSelected || hasUploaded;
        
        const baseSize = this.markerSize;
        const thresholdSize = baseSize + 2;
        
        for (let i = 0; i < n; i++) {
            const d = renderPoints[i];
            
            let color = 'red';
            let size = d.aboveThreshold ? thresholdSize : baseSize;
            let opacity = d.y >= 3 ? 0.7 : 0.4;
            
            if (this.selectedVariantKeys.size && this.selectedVariantKeys.has(this.variantKey(d))) {
                color = '#f1c40f';
                size = baseSize + 4;
                opacity = 1;
            } else if (hasHighlights) {
                if (hasSelected && this.hasHighlight(this.highlightedSnps.selected, d)) {
                    color = '#f1c40f';
                    size = baseSize + 4;
                    opacity = 1.0;
                } else if (hasUploaded && this.hasHighlight(this.highlightedSnps.uploaded, d)) {
                    color = '#2ecc71';
                    size = thresholdSize;
                    opacity = 1.0;
                } else if (!d.aboveThreshold) {
                    color = (chrToIndex[d.chromosome] || 0) % 2 === 0 ? '#808080' : '#b0b0b0';
                }
            } else if (!d.aboveThreshold) {
                color = (chrToIndex[d.chromosome] || 0) % 2 === 0 ? '#808080' : '#b0b0b0';
            }
            
            colors[i] = color;
            sizes[i] = size;
            opacities[i] = opacity;
        }
        
        // Restyle markers without rebuilding the trace geometry.
        const refreshed = this.enqueuePlotOperation('Unable to clear the region highlight', () => Plotly.restyle(this.container, {
            'marker.color': [colors],
            'marker.size': [sizes],
            'marker.opacity': [opacities]
        }));
        
        // Keep the Plotly selection state for multi-region selections.
        if (this.selectedPointIndices && this.selectedPointIndices.length > 0) {
            return refreshed.then(() => this.highlightAllSelected());
        }
        return refreshed;
    },
    
    /**
     * Return a chromosome's cumulative x-axis offset.
     */
    getChrOffset(chromosome) {
        const chr = chromosome.toString().replace(/^chr/i, '');
        for (const boundary of this.chrBoundaries) {
            if (String(boundary.chr) === chr) {
                return this.focusedChromosome ? 0 : boundary.start;
            }
        }
        return 0;
    },
    
    /**
     * Export the plot without transient lasso selection marks.
     * @param {string} format - 'png', 'svg', 'pdf'
     */
    async exportAs(format, options = {}) {
        if (!this.container) {
            alert('No plot to export');
            return false;
        }

        if (this._exportInProgress) {
            alert('A plot export is already in progress');
            return false;
        }
        this._exportInProgress = true;
        
        // Close the export menu.
        const exportMenu = document.getElementById('export-menu');
        if (exportMenu) exportMenu.style.display = 'none';
        const exportButton = document.querySelector('[data-action="toggle-export"]');
        if (exportButton) exportButton.setAttribute('aria-expanded', 'false');
        
        const filename = `manhattan_plot_${new Date().toISOString().slice(0, 10)}`;
        
        const dpi = Number(options.dpi || 300);
        if (![72, 96, 150, 300, 600].includes(dpi)) {
            this._exportInProgress = false;
            throw new Error('Choose a supported export resolution');
        }
        // Export participates in the same operation queue as redraws. Snapshot
        // geometry only after preceding redraws finish, and do not let a new
        // lasso or viewport update be overwritten by stale restoration.
        return this.enqueuePlotOperation('Unable to export the Manhattan plot', async () => {
            const layout = this.container._fullLayout || {};
            const ratio = Math.max(0.3, Math.min(3, (layout.height || 600) / (layout.width || 1200)));
            const exportConfig = { width: 1200, height: Math.round(1200 * ratio),
                scale: 12 * dpi / 1200, dpi };
            const savedSelections = this.container.layout?.selections || [];
            let failure = null;
            try {
                if (!this._plotInitialized) throw new Error('The plot has not finished rendering');
                if (savedSelections.length > 0) {
                    await Plotly.relayout(this.container, { selections: [] });
                }
                switch (format) {
                case 'png':
                    await this.exportPNG(filename, exportConfig);
                    break;
                case 'jpg':
                case 'jpeg':
                    await this.exportPNG(filename, { ...exportConfig, mime: 'image/jpeg' });
                    break;
                case 'svg':
                    await this.exportSVG(filename, exportConfig);
                    break;
                case 'pdf':
                    await this.exportPDF(filename, exportConfig);
                    break;
                default:
                    throw new Error('Unsupported format: ' + format);
                }
            } catch (error) {
                failure = error;
            }

            try {
                if (savedSelections.length > 0) {
                    await Plotly.relayout(this.container, { selections: savedSelections });
                }
            } catch (restoreError) {
                failure = failure || restoreError;
            } finally {
                this._exportInProgress = false;
            }

            if (failure) {
                console.error('Plot export failed:', failure);
                const detail = failure instanceof Error && failure.message
                    ? failure.message
                    : 'The browser could not render the requested file';
                alert('Export failed: ' + detail);
                return false;
            }
            return true;
        });
    },
    
    /**
     * Export as PNG.
     */
    async exportPNG(filename, config) {
        let dataUrl = await this.renderPngDataUrl(config);
        if (window.GWASMindFiles) dataUrl = window.GWASMindFiles.withDpi(dataUrl, config.dpi);
        this.downloadDataUrl(dataUrl, filename + (config.mime === 'image/jpeg' ? '.jpg' : '.png'));
    },

    /**
     * Render a Plotly figure to PNG without loading Plotly's temporary blob URL.
     *
     * Plotly's normal PNG path first creates a blob-backed SVG image. GWASMind's
     * strict local Content Security Policy deliberately does not allow blob:
     * image sources, so that path fails with an unhelpful DOM Event. Plotly can
     * still return the complete SVG as a data URL; loading that allowed data URL
     * into a canvas produces the same raster output without weakening the CSP.
     */
    async renderPngDataUrl(config) {
        const width = Math.max(1, Math.round(Number(config?.width) || 0));
        const height = Math.max(1, Math.round(Number(config?.height) || 0));
        const scale = Math.max(0.1, Number(config?.scale) || 1);
        const pixelWidth = Math.max(1, Math.round(width * scale));
        const pixelHeight = Math.max(1, Math.round(height * scale));

        const svgDataUrl = await Plotly.toImage(this.container, {
            format: 'svg',
            width,
            height,
            scale
        });
        if (typeof svgDataUrl !== 'string' || !svgDataUrl.startsWith('data:image/svg+xml')) {
            throw new Error('Plotly did not return a valid SVG image');
        }

        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = pixelWidth;
                canvas.height = pixelHeight;
                const context = canvas.getContext('2d');
                if (!context) {
                    reject(new Error('The browser could not create a 2D export canvas'));
                    return;
                }
                try {
                    context.fillStyle = '#ffffff';
                    context.fillRect(0, 0, pixelWidth, pixelHeight);
                    context.drawImage(image, 0, 0, pixelWidth, pixelHeight);
                    const mime = config.mime || 'image/png';
                    const pngDataUrl = canvas.toDataURL(mime, 0.95);
                    if (!pngDataUrl.startsWith(`data:${mime};base64,`)) {
                        throw new Error('The browser did not encode a valid image');
                    }
                    // Release the large backing store before the download begins.
                    canvas.width = 1;
                    canvas.height = 1;
                    resolve(pngDataUrl);
                } catch (error) {
                    reject(error instanceof Error ? error : new Error('PNG rendering failed'));
                }
            };
            image.onerror = () => reject(new Error('The browser could not decode the plot SVG'));
            image.src = svgDataUrl;
        });
    },

    /**
     * Download a trusted, locally generated data URL with an explicit extension.
     */
    downloadDataUrl(dataUrl, filename) {
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = filename;
        link.rel = 'noopener';
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        link.remove();
    },
    
    /**
     * Export as editable SVG.
     */
    async exportSVG(filename, config) {
        const dataUrl = await Plotly.toImage(this.container, {
            format: 'svg',
            filename: filename,
            width: config.width,
            height: config.height,
            scale: 1
        });
        this.downloadDataUrl(dataUrl, filename + '.svg');
    },
    
    /**
     * Export as PDF through jsPDF.
     */
    async exportPDF(filename, config) {
        // Fail clearly when the optional PDF library is unavailable.
        if (typeof window.jspdf === 'undefined') {
            throw new Error('The PDF export library is not available yet');
        }
            
        const { jsPDF } = window.jspdf;
            
        // Generate a compact, high-resolution source image through the same
        // CSP-compatible raster path used by direct PNG export.
        const imgData = await this.renderPngDataUrl({
            width: config.width,
            height: config.height,
            scale: 2
        });
            
        // Create a landscape A4 document.
        const pdf = new jsPDF({
                orientation: 'landscape',
                unit: 'mm',
                format: 'a4'
            });
            
        // A4 landscape dimensions: 297 mm x 210 mm.
        const pageWidth = 297;
        const pageHeight = 210;
        const margin = 35;
            
        // Fit the image while preserving its 2:1 aspect ratio.
        const imgRatio = config.width / config.height;
        const availableWidth = pageWidth - 2 * margin;
        const availableHeight = pageHeight - 2 * margin;
            
        let imgWidth, imgHeight;
        if (availableWidth / availableHeight > imgRatio) {
            imgHeight = availableHeight;
            imgWidth = imgHeight * imgRatio;
        } else {
            imgWidth = availableWidth;
            imgHeight = imgWidth / imgRatio;
        }
            
        // Center the plot on the page.
        const x = (pageWidth - imgWidth) / 2;
        const y = (pageHeight - imgHeight) / 2;
            
        // Add the image and download the document.
        pdf.addImage(imgData, 'PNG', x, y, imgWidth, imgHeight, undefined, 'FAST');
            
        pdf.save(filename + '.pdf');
    },
    
    // ============================================================
    // SNP Highlighting for Analysis Linkage
    // ============================================================
    
    /**
     * Highlight SNPs on the plot
     * @param {Array} snpPositions - Array of {chr, pos} objects
     * @param {string} color - Color for highlighting (used for identification, actual color is in buildTraces)
     * @param {string} group - 'uploaded' or 'selected'
     */
    highlightSnps(snpPositions, color, group) {
        if (!snpPositions || !Array.isArray(snpPositions)) return;
        if (group !== 'uploaded' && group !== 'selected') return;
        this.highlightedSnps[group] = this.highlightKeys(snpPositions);
        if (!this.webglSupported && group === 'selected') {
            const visible = new Set(this.getRenderPoints().flatMap(snp => [
                this.variantKey(snp), `${String(snp.chromosome).replace(/^chr/i, '').toUpperCase()}_${snp.position}`
            ]));
            if ([...this.highlightedSnps.selected].some(key => !visible.has(key))) {
                const view = this.capturePanelState().view;
                this.invalidateRenderCache();
                return this.render(view);
            }
        }
        return this.updateHighlighting();
    },

    highlightKeys(snpPositions) {
        const keys = new Set();
        if (!Array.isArray(snpPositions)) return keys;
        snpPositions.forEach(snp => {
            const chr = String(snp.chr || snp.chromosome || '').replace(/^chr/i, '').toUpperCase();
            const pos = Number(snp.pos || snp.position);
            if (chr && pos) {
                const key = snp.ref && snp.alt
                    ? this.variantKey({ chromosome: chr, position: pos, ref: snp.ref, alt: snp.alt })
                    : `${chr}_${pos}`;
                keys.add(key);
            }
        });
        
        return keys;
    },
    
    /**
     * Clear all highlights
     */
    clearHighlights() {
        this.highlightedSnps.uploaded.clear();
        this.highlightedSnps.selected.clear();
        this.updateHighlighting();
    },
    
    /**
     * Update plot colors without full redraw
     * ★ OPTIMIZED: Single pass instead of 3 separate .map() calls
     */
    updateHighlighting() {
        if (!this.container || !this.plotData.length) return;
        
        const chrToIndex = {};
        this.chrBoundaries.forEach((b, idx) => chrToIndex[b.chr] = idx);
        
        const renderPoints = this.getRenderPoints();
        const n = renderPoints.length;
        const newColors = new Array(n);
        const newSizes = new Array(n);
        const newOpacities = new Array(n);
        
        // Pre-check if we have any highlights
        const hasSelected = this.highlightedSnps.selected.size > 0;
        const hasUploaded = this.highlightedSnps.uploaded.size > 0;
        const hasHighlights = hasSelected || hasUploaded;
        
        const baseSize = this.markerSize;
        const thresholdSize = baseSize + 2;
        const uploadedSize = baseSize + 2;
        const selectedSize = baseSize + 4;
        
        for (let i = 0; i < n; i++) {
            const d = renderPoints[i];
            
            let isSelected = this.selectedVariantKeys.size > 0 && this.selectedVariantKeys.has(this.variantKey(d)), isUploaded = false;
            if (hasHighlights) {
                isSelected = isSelected || (hasSelected && this.hasHighlight(this.highlightedSnps.selected, d));
                isUploaded = !isSelected && hasUploaded && this.hasHighlight(this.highlightedSnps.uploaded, d);
            }
            
            // Color
            if (isSelected) {
                newColors[i] = '#f1c40f';
            } else if (isUploaded) {
                newColors[i] = '#2ecc71';
            } else if (d.aboveThreshold) {
                newColors[i] = 'red';
            } else {
                newColors[i] = (chrToIndex[d.chromosome] || 0) % 2 === 0 ? '#808080' : '#b0b0b0';
            }
            
            // Size
            if (isSelected) {
                newSizes[i] = selectedSize;
            } else if (isUploaded) {
                newSizes[i] = uploadedSize;
            } else {
                newSizes[i] = d.aboveThreshold ? thresholdSize : baseSize;
            }
            
            // Opacity
            newOpacities[i] = (isSelected || isUploaded) ? 1.0 : (d.y >= 3 ? 0.7 : 0.4);
        }
        
        // Use one serialized restyle for performance and to avoid racing a redraw.
        return this.enqueuePlotOperation('Unable to update plot highlighting', () => Plotly.restyle(this.container, {
            'marker.color': [newColors],
            'marker.size': [newSizes],
            'marker.opacity': [newOpacities],
            selectedpoints: [this.selectedPointIndices],
            'selected.marker.color': '#f1c40f',
            'selected.marker.opacity': 1
        }));
    },
    
    /**
     * Zoom to a specific genomic region
     * @param {string|number} chr - Chromosome
     * @param {number} startPos - Start position
     * @param {number} endPos - End position
     */
    zoomToRegion(chr, startPos, endPos) {
        if (!chr || !this.chrBoundaries.length) return;
        
        const chrStr = chr.toString();
        
        // First focus on the chromosome if not already
        if (!this.focusedChromosome || this.focusedChromosome !== chrStr) {
            this.focusedChromosome = chrStr;
            this.processData();
        }
        
        // Calculate X range with padding
        const padding = (endPos - startPos) * 0.1;
        const xStart = startPos - padding;
        const xEnd = endPos + padding;
        
        // Find Y range for this region
        let maxY = this.threshold;
        this.plotData.forEach(d => {
            if (d.position >= startPos && d.position <= endPos && d.y > maxY) {
                maxY = d.y;
            }
        });
        
        // Update layout and redraw
        const layout = this.buildLayout();
        layout.xaxis.range = [xStart, xEnd];
        layout.yaxis.range = [0, maxY * 1.2];
        layout.title.text = `Chromosome ${chrStr} - Region: ${startPos.toLocaleString()} - ${endPos.toLocaleString()}`;
        
        const result = this.enqueuePlotOperation('Unable to zoom to the selected region', async () => {
            await Plotly.react(this.container, this.buildTraces(), layout, {
                displayModeBar: false,
                locale: 'en',
                responsive: true,
                scrollZoom: true
            });
            this._plotInitialized = true;
            this.bindEvents();
            return true;
        });
        
        // Show zoom out button
        const zoomOutBtn = document.getElementById('zoom-out-btn');
        if (zoomOutBtn) zoomOutBtn.style.display = 'inline-block';
        return result;
    }
};

// Export for global access
window.ManhattanPlot = ManhattanPlot;

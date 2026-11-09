/**
 * Client-side GWAS data processor.
 *
 * The public browser API remains `GWASProcessor.processFile(file, onProgress)`.
 * Parsing and sampling are implemented as pure functions so the exact same
 * behavior can be exercised from Node without a DOM or FileReader.
 */
(function installGWASProcessor(root, factory) {
    const processor = factory();
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = processor;
    }
    if (root) {
        root.GWASProcessor = processor;
    }
})(typeof globalThis !== 'undefined' ? globalThis : this, function createGWASProcessor() {
    'use strict';

    // Scientific and resource limits come from the same validated server
    // configuration as retrieval. The rounded display threshold is derived.
    const config = globalThis.GWASMindConfig;
    const MAX_POINTS = config.get('plot.background_points');
    const SIGNIFICANCE_THRESHOLD = config.get('plot.suggestive_logp');
    const GENOME_WIDE_P_VALUE = config.get('analysis.gwas_significance_p');
    const LOG_P_DECIMAL_PLACES = config.get('plot.threshold_decimal_places');
    const GENOME_WIDE_THRESHOLD = Number((-Math.log10(GENOME_WIDE_P_VALUE))
        .toFixed(LOG_P_DECIMAL_PLACES));
    const BACKGROUND_BIN_SIZE = config.get('plot.background_bin_logp_width');
    const HARD_TOTAL_CAP = config.get('plot.max_total_points');
    const CHUNK_SIZE = config.get('plot.upload_chunk_bytes');
    const MAX_RECORD_CHARS = config.get('plot.max_record_chars');
    const P_VALUE_FLOOR = config.get('plot.p_value_floor');
    const DEFAULT_SEED = config.get('plot.sampling_seed');
    const NUMERIC_TOKEN = /^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$/;

    const COLUMN_MAPPING = Object.freeze({
        chr: 'chromosome', chrom: 'chromosome', '#chr': 'chromosome', '#chrom': 'chromosome',
        chromosome_name: 'chromosome',
        pos: 'position', bp: 'position', bp_pos: 'position', base_pair_location: 'position',
        p: 'pvalue', p_value: 'pvalue', 'p-value': 'pvalue', pval: 'pvalue',
        reference: 'ref', reference_allele: 'ref', ref_allele: 'ref',
        alternate: 'alt', alternate_allele: 'alt', alt_allele: 'alt',
        a1: 'allele1', a2: 'allele2', ea: 'effect_allele', effectallele: 'effect_allele',
        nea: 'other_allele', non_effect_allele: 'other_allele', otherallele: 'other_allele',
        snp: 'snp_id', rsid: 'snp_id', rs: 'snp_id', variant_id: 'snp_id', id: 'snp_id',
        oddsratio: 'odds_ratio', or: 'odds_ratio', standard_error: 'se'
    });

    function normalizeHeader(value) {
        return String(value == null ? '' : value)
            .replace(/^\uFEFF/, '')
            .trim()
            .toLowerCase()
            .replace(/\s+/g, '_');
    }

    function canonicalHeader(value) {
        const normalized = normalizeHeader(value);
        return COLUMN_MAPPING[normalized] || normalized;
    }

    function parseDelimitedRecord(record, delimiter) {
        if (delimiter !== ',' && delimiter !== '\t') {
            throw new Error('Delimiter must be comma or tab');
        }
        const fields = [];
        let field = '';
        let inQuotes = false;
        let afterClosingQuote = false;
        let quoted = false;

        for (let index = 0; index < record.length; index += 1) {
            const character = record[index];
            if (inQuotes) {
                if (character === '"') {
                    if (record[index + 1] === '"') {
                        field += '"';
                        index += 1;
                    } else {
                        inQuotes = false;
                        afterClosingQuote = true;
                    }
                } else {
                    field += character;
                }
                continue;
            }

            if (afterClosingQuote) {
                if (character === delimiter) {
                    fields.push(field);
                    field = '';
                    quoted = false;
                    afterClosingQuote = false;
                } else if (character !== ' ' && character !== '\t') {
                    throw new Error('Unexpected text after a quoted field');
                }
                continue;
            }

            if (character === delimiter) {
                fields.push(quoted ? field : field.trim());
                field = '';
                quoted = false;
            } else if (character === '"') {
                if (field.trim()) {
                    throw new Error('Unexpected quote in an unquoted field');
                }
                field = '';
                quoted = true;
                inQuotes = true;
            } else {
                field += character;
            }
        }

        if (inQuotes) {
            throw new Error('Unterminated quoted field');
        }
        fields.push(quoted || afterClosingQuote ? field : field.trim());
        return fields;
    }

    function countDelimiterOutsideQuotes(record, delimiter) {
        let count = 0;
        let inQuotes = false;
        for (let index = 0; index < record.length; index += 1) {
            const character = record[index];
            if (character === '"') {
                if (inQuotes && record[index + 1] === '"') {
                    index += 1;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (!inQuotes && character === delimiter) {
                count += 1;
            }
        }
        if (inQuotes) {
            throw new Error('Unterminated quoted header field');
        }
        return count;
    }

    function detectDelimiter(headerRecord) {
        const commaCount = countDelimiterOutsideQuotes(headerRecord, ',');
        const tabCount = countDelimiterOutsideQuotes(headerRecord, '\t');
        if (commaCount === 0 && tabCount === 0) {
            throw new Error('Could not detect CSV or TSV delimiter');
        }
        if (commaCount === tabCount) {
            throw new Error('Header has an ambiguous comma/tab delimiter');
        }
        return tabCount > commaCount ? '\t' : ',';
    }

    function buildColumnIndex(headers) {
        if (!Array.isArray(headers)) {
            throw new Error('Headers must be an array');
        }
        const columnIndex = Object.create(null);
        for (let index = 0; index < headers.length; index += 1) {
            const canonical = canonicalHeader(headers[index]);
            if (!canonical) {
                continue;
            }
            if (Object.prototype.hasOwnProperty.call(columnIndex, canonical)) {
                throw new Error(`Duplicate column after normalization: ${canonical}`);
            }
            columnIndex[canonical] = index;
        }
        const required = ['chromosome', 'position', 'pvalue'];
        const missing = required.filter((name) => columnIndex[name] === undefined);
        if (missing.length) {
            throw new Error(`Missing required columns: ${missing.join(', ')}. Found: ${headers.join(', ')}`);
        }
        return columnIndex;
    }

    function consumeCompleteRecords(buffer, final, onRecord) {
        let start = 0;
        let inQuotes = false;
        for (let index = 0; index < buffer.length; index += 1) {
            const character = buffer[index];
            if (character === '"') {
                if (inQuotes && buffer[index + 1] === '"') {
                    index += 1;
                } else {
                    inQuotes = !inQuotes;
                }
                continue;
            }
            if (!inQuotes && (character === '\n' || character === '\r')) {
                if (index - start > MAX_RECORD_CHARS) {
                    throw new Error('A delimited record exceeds 1 MB');
                }
                onRecord(buffer.slice(start, index));
                if (character === '\r' && buffer[index + 1] === '\n') {
                    index += 1;
                }
                start = index + 1;
            }
        }

        if (final) {
            if (inQuotes) {
                throw new Error('Unterminated quoted field at end of file');
            }
            if (buffer.length - start > MAX_RECORD_CHARS) {
                throw new Error('A delimited record exceeds 1 MB');
            }
            if (start < buffer.length) {
                onRecord(buffer.slice(start));
            }
            return '';
        }

        const leftover = buffer.slice(start);
        if (leftover.length > MAX_RECORD_CHARS) {
            throw new Error('A delimited record exceeds 1 MB');
        }
        return leftover;
    }

    function normalizeChromosome(value) {
        let chromosome = String(value == null ? '' : value).trim().replace(/^chr/i, '').toUpperCase();
        if (chromosome === 'M') {
            chromosome = 'MT';
        }
        if (/^0+[1-9]\d*$/.test(chromosome)) {
            chromosome = String(Number(chromosome));
        }
        if (/^(?:[1-9]|1\d|2[0-2]|X|Y|MT)$/.test(chromosome)) {
            return chromosome;
        }
        return '';
    }

    function normalizeAllele(value) {
        const allele = String(value == null ? '' : value).trim().toUpperCase();
        return allele.length <= 32 && /^(?:[ACGTN]+|-)$/.test(allele) ? allele : 'N';
    }

    function normalizeVariantId(value, chromosome, position, ref, alt) {
        const supplied = String(value == null ? '' : value).trim();
        const rsMatch = /^rs(\d+)$/i.exec(supplied);
        if (rsMatch) {
            return `rs${rsMatch[1]}`;
        }
        const coordinate = supplied.replace(/^chr/i, '').split(/[_:]/);
        if (coordinate.length === 4) {
            const coordinateChromosome = normalizeChromosome(coordinate[0]);
            const coordinatePosition = parsePositionStrict(coordinate[1]);
            const coordinateRef = normalizeAllele(coordinate[2]);
            const coordinateAlt = normalizeAllele(coordinate[3]);
            if (coordinateChromosome && coordinatePosition !== null) {
                return `${coordinateChromosome}_${coordinatePosition}_${coordinateRef}_${coordinateAlt}`;
            }
        }
        return `${chromosome}_${position}_${ref}_${alt}`;
    }

    function parsePositionStrict(value) {
        const text = String(value == null ? '' : value).trim();
        if (!NUMERIC_TOKEN.test(text)) {
            return null;
        }
        const position = Number(text);
        return Number.isSafeInteger(position) && position > 0 ? position : null;
    }

    function parsePValueStrict(value) {
        const text = String(value == null ? '' : value).trim();
        if (!NUMERIC_TOKEN.test(text)) {
            return null;
        }
        const pValue = Number(text);
        // Negative values can underflow to JavaScript's -0. They are invalid
        // P values, not extremely significant positive observations.
        const negativeUnderflow = pValue === 0 && text.startsWith('-')
            && /[1-9]/.test(text.split(/[eE]/)[0]);
        if (!Number.isFinite(pValue) || pValue < 0 || pValue > 1 || negativeUnderflow) {
            return null;
        }
        return pValue === 0 ? P_VALUE_FLOOR : Math.max(pValue, P_VALUE_FLOOR);
    }

    function parseSnpValues(values, columnIndex) {
        const requiredMaximum = Math.max(
            columnIndex.chromosome,
            columnIndex.position,
            columnIndex.pvalue
        );
        if (!Array.isArray(values) || values.length <= requiredMaximum) {
            return null;
        }
        const chromosome = normalizeChromosome(values[columnIndex.chromosome]);
        const position = parsePositionStrict(values[columnIndex.position]);
        const pValue = parsePValueStrict(values[columnIndex.pvalue]);
        if (!chromosome || position === null || pValue === null) {
            return null;
        }

        const refIndex = columnIndex.ref;
        const altIndex = columnIndex.alt;
        const snpIndex = columnIndex.snp_id;
        const ref = refIndex === undefined ? 'N' : normalizeAllele(values[refIndex]);
        const alt = altIndex === undefined ? 'N' : normalizeAllele(values[altIndex]);
        const suppliedId = snpIndex === undefined ? '' : values[snpIndex];
        // An explicit genomic ID must agree with the row supplying its P value.
        // Invalid labels still fall back to the row's canonical identity; a
        // well-formed but conflicting coordinate ID is a malformed observation.
        const coordinate = String(suppliedId || '').trim().replace(/^chr/i, '').split(/[_:]/);
        if (coordinate.length === 4 && normalizeChromosome(coordinate[0])
                && parsePositionStrict(coordinate[1]) !== null) {
            if (normalizeChromosome(coordinate[0]) !== chromosome
                    || parsePositionStrict(coordinate[1]) !== position
                    || (ref !== 'N' && normalizeAllele(coordinate[2]) !== ref)
                    || (alt !== 'N' && normalizeAllele(coordinate[3]) !== alt)) return null;
        }
        const snpId = normalizeVariantId(
            suppliedId,
            chromosome,
            position,
            ref,
            alt
        ).slice(0, 120);

        const sourceEffects = {};
        for (const field of ['beta', 'odds_ratio', 'se']) {
            const raw = columnIndex[field] === undefined ? '' : String(values[columnIndex[field]] ?? '').trim();
            if (!raw || !NUMERIC_TOKEN.test(raw)) continue;
            const value = Number(raw);
            if (!Number.isFinite(value) || (field === 'odds_ratio' && value <= 0)
                    || (field === 'se' && value < 0)) continue;
            sourceEffects[field] = value;
        }
        // Effect coding and unassigned A1/A2 columns are not reference-build
        // alleles. Keep them independently, even when no effect size is given.
        for (const field of ['effect_allele', 'other_allele', 'allele1', 'allele2']) {
            const raw = columnIndex[field] === undefined ? '' : String(values[columnIndex[field]] ?? '').trim().toUpperCase();
            if (/^[ACGT]{1,32}$/.test(raw)) sourceEffects[field] = raw;
        }
        return {
            chromosome,
            position,
            pvalue: pValue,
            neg_log_p: -Math.log10(pValue),
            snp_id: snpId,
            ref,
            alt,
            sourceEffects
        };
    }

    function chromosomeRank(chromosome) {
        if (/^\d+$/.test(chromosome)) {
            return Number(chromosome);
        }
        return { X: 23, Y: 24, MT: 25 }[chromosome] || 99;
    }

    function compareChromosomes(left, right) {
        const rankDifference = chromosomeRank(String(left)) - chromosomeRank(String(right));
        return rankDifference || String(left).localeCompare(String(right));
    }

    function compareSnps(left, right) {
        return compareChromosomes(left.chromosome, right.chromosome)
            || left.position - right.position
            || left.snp_id.localeCompare(right.snp_id);
    }

    function hashSeed(value) {
        if (value == null) {
            return DEFAULT_SEED >>> 0;
        }
        if (typeof value === 'number' && Number.isFinite(value)) {
            return value >>> 0;
        }
        const text = String(value);
        let hash = 2166136261;
        for (let index = 0; index < text.length; index += 1) {
            hash ^= text.charCodeAt(index);
            hash = Math.imul(hash, 16777619);
        }
        return hash >>> 0;
    }

    function createDeterministicRng(seed = DEFAULT_SEED) {
        let state = hashSeed(seed);
        return function nextRandom() {
            state = (state + 0x6D2B79F5) >>> 0;
            let value = state;
            value = Math.imul(value ^ (value >>> 15), value | 1);
            value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
            return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
        };
    }

    function reservoirConsider(reservoir, item, seen, limit, random) {
        if (limit <= 0) {
            return;
        }
        if (reservoir.length < limit) {
            reservoir.push(item);
            return;
        }
        const candidate = Math.floor(random() * seen);
        if (candidate < limit) {
            reservoir[candidate] = item;
        }
    }

    function shrinkReservoir(reservoir, limit, random) {
        while (reservoir.length > limit) {
            const removeAt = Math.floor(random() * reservoir.length);
            reservoir[removeAt] = reservoir[reservoir.length - 1];
            reservoir.pop();
        }
    }

    function deterministicReservoirSample(items, limit, seed = DEFAULT_SEED) {
        if (!Number.isInteger(limit) || limit < 0) {
            throw new Error('Reservoir limit must be a non-negative integer');
        }
        const random = createDeterministicRng(seed);
        const reservoir = [];
        let seen = 0;
        for (const item of items) {
            seen += 1;
            reservoirConsider(reservoir, item, seen, limit, random);
        }
        return reservoir;
    }

    class BoundedPrioritySampler {
        constructor(backgroundLimit, threshold, seed, hardTotalCap = HARD_TOTAL_CAP) {
            if (!Number.isInteger(backgroundLimit) || backgroundLimit <= 0) {
                throw new Error('Background sample limit must be a positive integer');
            }
            if (!Number.isInteger(hardTotalCap) || hardTotalCap <= 0) {
                throw new Error('Hard total cap must be a positive integer');
            }
            this.backgroundLimit = backgroundLimit;
            this.hardTotalCap = hardTotalCap;
            this.threshold = threshold;
            this.seed = hashSeed(seed);
            this.random = createDeterministicRng(this.seed);
            this.genomeWide = [];
            this.priority = [];
            this.significantSeen = 0;
            this.genomeWideSeen = 0;
            this.prioritySeen = 0;
            this.backgroundSeen = 0;
            this.binSize = BACKGROUND_BIN_SIZE;
            this.binCount = Math.max(1, Math.ceil(this.threshold / this.binSize));
            this.samplesPerBin = Math.floor(this.backgroundLimit / this.binCount);
            this.backgroundBins = Array.from({ length: this.binCount }, () => []);
            this.backgroundBinSeen = new Uint32Array(this.binCount);
            this.backgroundBinCapacity = new Uint32Array(this.binCount);
            this._lastBackgroundCapacity = -1;
            this.rebalanceBackground();
        }

        consider(snp) {
            // The display line is rounded; scientific membership must use P.
            // The configured rendering floor never exceeds this P threshold.
            const genomeWide = snp.pvalue <= GENOME_WIDE_P_VALUE;
            if (snp.neg_log_p >= this.threshold || genomeWide) {
                this.significantSeen += 1;
                if (genomeWide) {
                    this.genomeWideSeen += 1;
                    reservoirConsider(
                        this.genomeWide,
                        snp,
                        this.genomeWideSeen,
                        this.hardTotalCap,
                        this.random
                    );
                    shrinkReservoir(this.priority, this.priorityCapacity(), this.random);
                } else {
                    this.prioritySeen += 1;
                    reservoirConsider(
                        this.priority,
                        snp,
                        this.prioritySeen,
                        this.priorityCapacity(),
                        this.random
                    );
                }
                this.rebalanceBackground();
            } else {
                this.backgroundSeen += 1;
                const binIndex = Math.min(
                    this.binCount - 1,
                    Math.max(0, Math.floor(snp.neg_log_p / this.binSize))
                );
                this.backgroundBinSeen[binIndex] += 1;
                if (this.backgroundBinSeen[binIndex] === 1) {
                    this.rebalanceBackground();
                }
                reservoirConsider(
                    this.backgroundBins[binIndex],
                    snp,
                    this.backgroundBinSeen[binIndex],
                    this.backgroundBinCapacity[binIndex],
                    this.random
                );
            }
        }

        backgroundCapacity() {
            return Math.min(
                this.backgroundLimit,
                Math.max(0, this.hardTotalCap - Math.min(this.significantSeen, this.hardTotalCap))
            );
        }

        priorityCapacity() {
            return Math.max(
                0,
                this.hardTotalCap - Math.min(this.genomeWideSeen, this.hardTotalCap)
            );
        }

        rebalanceBackground() {
            const totalCapacity = this.backgroundCapacity();
            if (totalCapacity === this._lastBackgroundCapacity) return;
            this._lastBackgroundCapacity = totalCapacity;

            const baseCapacity = Math.min(
                this.samplesPerBin,
                Math.floor(totalCapacity / this.binCount)
            );
            let remainder = Math.min(
                totalCapacity - baseCapacity * this.binCount,
                (this.samplesPerBin - baseCapacity) * this.binCount
            );
            for (let index = 0; index < this.binCount; index += 1) {
                this.backgroundBinCapacity[index] = baseCapacity;
            }
            // A reduced pathological-cap budget is spread across all ten
            // canonical background bands, with rounding assigned from high to low.
            for (let index = this.binCount - 1; index >= 0 && remainder > 0; index -= 1) {
                if (this.backgroundBinCapacity[index] < this.samplesPerBin) {
                    this.backgroundBinCapacity[index] += 1;
                    remainder -= 1;
                }
            }
            for (let binIndex = 0; binIndex < this.binCount; binIndex += 1) {
                shrinkReservoir(
                    this.backgroundBins[binIndex],
                    this.backgroundBinCapacity[binIndex],
                    this.random
                );
            }
        }

        backgroundValues() {
            return this.backgroundBins.flat();
        }

        significantValues() {
            return this.genomeWide.concat(this.priority);
        }

        values() {
            return this.significantValues().concat(this.backgroundValues());
        }
    }

    function createProgressReporter(callback) {
        let last = 0;
        const report = typeof callback === 'function' ? callback : function noop() {};
        return {
            emit(percent, message) {
                const numeric = Number.isFinite(Number(percent)) ? Math.round(Number(percent)) : last;
                last = Math.max(last, Math.min(100, Math.max(0, numeric)));
                report(last, String(message || ''));
                return last;
            },
            value() {
                return last;
            }
        };
    }

    function createParserState(options = {}) {
        const maxPoints = options.maxPoints === undefined ? MAX_POINTS : Number(options.maxPoints);
        const hardTotalCap = options.hardTotalCap === undefined
            ? HARD_TOTAL_CAP
            : Number(options.hardTotalCap);
        const threshold = options.significanceThreshold === undefined
            ? SIGNIFICANCE_THRESHOLD
            : Number(options.significanceThreshold);
        if (!Number.isInteger(maxPoints) || maxPoints <= 0) {
            throw new Error('maxPoints must be a positive integer');
        }
        if (!Number.isInteger(hardTotalCap) || hardTotalCap <= 0) {
            throw new Error('hardTotalCap must be a positive integer');
        }
        if (!Number.isFinite(threshold) || threshold < 0) {
            throw new Error('significanceThreshold must be non-negative');
        }
        return {
            buffer: '',
            firstText: true,
            finalized: false,
            headers: null,
            canonicalHeaders: null,
            delimiter: null,
            columnIndex: null,
            totalRows: 0,
            validRows: 0,
            rejectedRows: 0,
            repeatedHeaders: 0,
            zeroPValuesCapped: 0,
            sampler: new BoundedPrioritySampler(maxPoints, threshold, options.seed, hardTotalCap),
            maxPoints,
            hardTotalCap,
            threshold
        };
    }

    function isPreambleComment(record) {
        const trimmed = record.trim();
        return trimmed.startsWith('##') || /^#\s/.test(trimmed);
    }

    function isRepeatedHeader(values, state) {
        if (values.length !== state.canonicalHeaders.length) {
            return false;
        }
        return values.every((value, index) => canonicalHeader(value) === state.canonicalHeaders[index]);
    }

    function processLogicalRecord(state, record) {
        if (!record.trim()) {
            return;
        }
        if (state.headers === null) {
            if (isPreambleComment(record)) {
                return;
            }
            state.delimiter = detectDelimiter(record);
            state.headers = parseDelimitedRecord(record, state.delimiter).map(normalizeHeader);
            state.canonicalHeaders = state.headers.map(canonicalHeader);
            state.columnIndex = buildColumnIndex(state.headers);
            return;
        }

        let values;
        try {
            values = parseDelimitedRecord(record, state.delimiter);
        } catch (_error) {
            state.totalRows += 1;
            state.rejectedRows += 1;
            return;
        }
        if (isRepeatedHeader(values, state)) {
            state.repeatedHeaders += 1;
            return;
        }

        state.totalRows += 1;
        const rawPValue = values[state.columnIndex.pvalue];
        const numericPValue = Number(String(rawPValue == null ? '' : rawPValue).trim());
        const snp = parseSnpValues(values, state.columnIndex);
        if (!snp) {
            state.rejectedRows += 1;
            return;
        }
        if (numericPValue === 0) {
            state.zeroPValuesCapped += 1;
        }
        state.validRows += 1;
        state.sampler.consider(snp);
    }

    function feedParser(state, text, final = false) {
        if (state.finalized) {
            throw new Error('Parser has already been finalized');
        }
        let decoded = String(text == null ? '' : text);
        if (state.firstText) {
            decoded = decoded.replace(/^\uFEFF/, '');
            state.firstText = false;
        }
        state.buffer += decoded;
        state.buffer = consumeCompleteRecords(
            state.buffer,
            final,
            (record) => processLogicalRecord(state, record)
        );
        if (final) {
            state.finalized = true;
        }
    }

    function finalizeParser(state, filename, processingTime) {
        if (!state.finalized) {
            feedParser(state, '', true);
        }
        if (!state.headers) {
            throw new Error('GWAS file has no header');
        }
        if (state.validRows === 0) {
            throw new Error('GWAS file contains no valid data rows');
        }

        const finalData = state.sampler.values();
        const significantSeen = state.sampler.significantSeen;
        const significantRetained = state.sampler.genomeWide.length + state.sampler.priority.length;
        const significantOmitted = Math.max(0, significantSeen - significantRetained);
        const samplingStrategy = state.validRows <= finalData.length
            ? 'none'
            : significantOmitted > 0
                ? state.sampler.genomeWideSeen > state.sampler.hardTotalCap
                    ? 'hard-cap-genome-wide-reservoir'
                    : 'hard-cap-genome-wide-then-moderate-reservoir'
                : state.sampler.backgroundCapacity() < state.sampler.backgroundLimit
                    ? 'hard-cap-retain-all-priority-then-reduced-background'
                    : `retain-all-priority-then-${state.sampler.samplesPerBin}-per-bin-background-reservoir`;
        finalData.sort(compareSnps);
        const chromosomes = Array.from(new Set(finalData.map((item) => item.chromosome)))
            .sort(compareChromosomes);
        const compactData = finalData.map((item) => [
            item.chromosome,
            item.position,
            Number(item.neg_log_p.toFixed(LOG_P_DECIMAL_PLACES)),
            item.snp_id,
            item.pvalue,
            item.ref,
            item.alt,
            ...(Object.keys(item.sourceEffects || {}).length ? [item.sourceEffects] : [])
        ]);

        return {
            data: compactData,
            format: 'array',
            metadata: {
                filename: String(filename || 'gwas-data').slice(0, 240),
                data_source: 'user_upload',
                total_snps: compactData.length,
                original_snps: state.validRows,
                chromosomes,
                sampled: state.validRows > compactData.length,
                processing_time: String(processingTime == null ? '0.0' : processingTime),
                rejected_rows: state.rejectedRows,
                repeated_headers: state.repeatedHeaders,
                zero_pvalues_capped: state.zeroPValuesCapped,
                sampling_strategy: samplingStrategy,
                priority_threshold: state.sampler.threshold,
                priority_seen: significantSeen,
                priority_retained: significantRetained,
                priority_omitted: significantOmitted,
                priority_truncated: significantOmitted > 0,
                priority_label: 'priority_or_suggestive',
                // Compatibility aliases for existing server/UI consumers.
                significant_seen: significantSeen,
                significant_retained: significantRetained,
                significant_omitted: significantOmitted,
                significant_truncated: significantOmitted > 0,
                significant_label: 'priority_or_suggestive',
                genome_wide_threshold: GENOME_WIDE_THRESHOLD,
                genome_wide_seen: state.sampler.genomeWideSeen,
                genome_wide_retained: state.sampler.genomeWide.length,
                moderate_seen: state.sampler.prioritySeen,
                moderate_retained: state.sampler.priority.length,
                background_seen: state.sampler.backgroundSeen,
                background_retained: finalData.length - significantRetained,
                background_bin_size: state.sampler.binSize,
                background_per_bin_cap: state.sampler.samplesPerBin,
                background_bin_capacity: Array.from(state.sampler.backgroundBinCapacity),
                background_bin_seen: Array.from(state.sampler.backgroundBinSeen),
                background_bin_retained: state.sampler.backgroundBins.map((bin) => bin.length),
                hard_total_cap: state.sampler.hardTotalCap,
                sampling_seed: state.sampler.seed
            }
        };
    }

    function processTextChunks(chunks, options = {}) {
        if (!chunks || typeof chunks[Symbol.iterator] !== 'function') {
            throw new Error('chunks must be iterable');
        }
        const chunkList = Array.from(chunks, (chunk) => String(chunk));
        const totalLength = chunkList.reduce((sum, chunk) => sum + chunk.length, 0);
        if (totalLength === 0) {
            throw new Error('GWAS file is empty');
        }
        const progress = createProgressReporter(options.onProgress);
        const state = createParserState(options);
        throwIfAborted(options.signal);
        progress.emit(1, 'Starting parser...');
        let consumed = 0;
        for (const chunk of chunkList) {
            throwIfAborted(options.signal);
            feedParser(state, chunk, false);
            consumed += chunk.length;
            progress.emit(Math.round((consumed / totalLength) * 85), 'Processing records...');
        }
        throwIfAborted(options.signal);
        feedParser(state, '', true);
        progress.emit(88, 'Combining results...');
        const result = finalizeParser(state, options.filename || 'gwas-data', '0.0');
        progress.emit(95, `Processed ${result.data.length.toLocaleString()} SNPs`);
        return result;
    }

    function decodeUtf8Chunks(byteChunks) {
        const decoder = new TextDecoder('utf-8', { fatal: true });
        const decoded = [];
        for (const bytes of byteChunks) {
            const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
            decoded.push(decoder.decode(view, { stream: true }));
        }
        decoded.push(decoder.decode());
        return decoded;
    }

    function createAbortError() {
        const error = new Error('The operation was cancelled');
        error.name = 'AbortError';
        return error;
    }

    function throwIfAborted(signal) {
        if (signal?.aborted) throw createAbortError();
    }

    async function readBlobBytes(blob, signal) {
        throwIfAborted(signal);
        if (blob && typeof blob.arrayBuffer === 'function') {
            const result = new Uint8Array(await blob.arrayBuffer());
            throwIfAborted(signal);
            return result;
        }
        if (typeof FileReader === 'undefined') {
            throw new Error('This browser cannot read local file chunks');
        }
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            const cleanup = () => signal?.removeEventListener('abort', handleAbort);
            const handleAbort = () => {
                if (reader.readyState === 1) reader.abort();
                cleanup();
                reject(createAbortError());
            };
            reader.onload = () => {
                cleanup();
                resolve(new Uint8Array(reader.result));
            };
            reader.onerror = () => {
                cleanup();
                reject(new Error('Failed to read file chunk'));
            };
            reader.onabort = () => {
                cleanup();
                reject(createAbortError());
            };
            signal?.addEventListener('abort', handleAbort, { once: true });
            if (signal?.aborted) {
                handleAbort();
                return;
            }
            reader.readAsArrayBuffer(blob);
        });
    }

    function firstLogicalRecord(text) {
        let record = null;
        let endPosition = 0;
        let inQuotes = false;
        for (let index = 0; index < text.length; index += 1) {
            const character = text[index];
            if (character === '"') {
                if (inQuotes && text[index + 1] === '"') {
                    index += 1;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (!inQuotes && (character === '\n' || character === '\r')) {
                record = text.slice(0, index);
                endPosition = index + 1;
                if (character === '\r' && text[index + 1] === '\n') {
                    endPosition += 1;
                }
                break;
            }
        }
        if (record === null && text && !inQuotes) {
            record = text;
            endPosition = text.length;
        }
        if (record === null) {
            throw new Error('Cannot find a complete header record');
        }
        return { record, endPosition };
    }

    const GWASProcessor = {
        MAX_POINTS,
        SIGNIFICANCE_THRESHOLD,
        GENOME_WIDE_THRESHOLD,
        BACKGROUND_BIN_SIZE,
        HARD_TOTAL_CAP,
        CHUNK_SIZE,
        COLUMN_MAPPING,

        async processFile(file, onProgress, options = {}) {
            if (!file || typeof file.slice !== 'function' || !Number.isFinite(file.size) || file.size <= 0) {
                throw new Error('Selected GWAS file is empty or unreadable');
            }
            const startedAt = Date.now();
            const filename = String(file.name || 'gwas-data');
            const progress = createProgressReporter(onProgress);
            const signal = options.signal;
            const parser = createParserState({
                maxPoints: this.MAX_POINTS,
                significanceThreshold: this.SIGNIFICANCE_THRESHOLD,
                seed: DEFAULT_SEED
            });
            const decoder = new TextDecoder('utf-8', { fatal: true });
            const totalChunks = Math.ceil(file.size / this.CHUNK_SIZE);
            let offset = 0;
            let chunkNumber = 0;

            throwIfAborted(signal);
            progress.emit(1, `Starting to process ${(file.size / 1024 / 1024).toFixed(1)}MB file...`);
            while (offset < file.size) {
                throwIfAborted(signal);
                chunkNumber += 1;
                const chunkEnd = Math.min(offset + this.CHUNK_SIZE, file.size);
                const bytes = await readBlobBytes(file.slice(offset, chunkEnd), signal);
                throwIfAborted(signal);
                feedParser(parser, decoder.decode(bytes, { stream: true }), false);
                progress.emit(
                    Math.round((chunkEnd / file.size) * 85),
                    `Processing... ${parser.validRows.toLocaleString()} SNPs (chunk ${chunkNumber}/${totalChunks})`
                );
                offset = chunkEnd;
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
            throwIfAborted(signal);
            feedParser(parser, decoder.decode(), true);
            progress.emit(88, 'Combining results...');
            const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
            const result = finalizeParser(parser, filename, elapsed);
            const diagnostics = [];
            if (result.metadata.sampled) {
                diagnostics.push(`sampled from ${result.metadata.original_snps.toLocaleString()} valid rows`);
            }
            if (result.metadata.rejected_rows) {
                diagnostics.push(`rejected ${result.metadata.rejected_rows.toLocaleString()} malformed rows`);
            }
            if (result.metadata.zero_pvalues_capped) {
                diagnostics.push(`capped ${result.metadata.zero_pvalues_capped.toLocaleString()} zero p-values`);
            }
            const suffix = diagnostics.length ? ` (${diagnostics.join('; ')})` : '';
            progress.emit(95, `Prepared ${result.data.length.toLocaleString()} SNPs${suffix}`);
            return result;
        },

        async readChunk(file, start, end) {
            const bytes = await readBlobBytes(file.slice(start, end));
            return new TextDecoder('utf-8', { fatal: true })
                .decode(bytes)
                .replace(/^\uFEFF/, '');
        },

        parseHeaders(chunk) {
            const first = firstLogicalRecord(String(chunk).replace(/^\uFEFF/, ''));
            const delimiter = detectDelimiter(first.record);
            const headers = parseDelimitedRecord(first.record, delimiter).map(normalizeHeader);
            buildColumnIndex(headers);
            return { headers, delimiter, headerEndPos: first.endPosition };
        },

        buildColumnIndex,

        parseLine(line, delimiter, columnIndex) {
            try {
                return parseSnpValues(parseDelimitedRecord(String(line), delimiter), columnIndex);
            } catch (_error) {
                return null;
            }
        },

        // Pure helpers intentionally exposed for deterministic Node tests.
        parseDelimitedRecord,
        detectDelimiter,
        normalizeChromosome,
        parsePositionStrict,
        parsePValueStrict,
        normalizeVariantId,
        compareChromosomes,
        createDeterministicRng,
        deterministicReservoirSample,
        createProgressReporter,
        processTextChunks,
        decodeUtf8Chunks
    };

    return GWASProcessor;
});

(function () {
    'use strict';

    function element(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function dialog(title) {
        const node = element('dialog', 'workspace-dialog');
        const heading = element('header');
        const close = element('button', 'dialog-close', '×');
        close.type = 'button';
        close.setAttribute('aria-label', 'Close');
        close.onclick = () => node.close();
        heading.append(element('h2', '', title), close);
        node.append(heading);
        const opener = document.activeElement;
        node.addEventListener('close', () => { node.remove(); opener?.focus?.(); });
        node.addEventListener('click', event => {
            if (event.target === node) {
                const rect = node.getBoundingClientRect();
                if (event.clientX < rect.left || event.clientX > rect.right
                        || event.clientY < rect.top || event.clientY > rect.bottom) node.close();
            }
        });
        document.body.append(node);
        node.showModal();
        return node;
    }

    function choice(parent, label, values) {
        const field = element('label', '', label);
        const select = element('select');
        for (const [value, text] of values) {
            const option = element('option', '', text);
            option.value = value;
            select.append(option);
        }
        field.append(select);
        parent.append(field);
        return select;
    }

    function download(content, filename, mime) {
        const url = URL.createObjectURL(new Blob([content], { type: mime }));
        const link = element('a');
        link.href = url;
        link.download = filename;
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30_000);
    }

    // Canvas defaults to 96 DPI. Write physical resolution as well as pixels.
    function withDpi(dataUrl, dpi = 300) {
        const [prefix, encoded] = dataUrl.split(',');
        const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
        let output;
        if (prefix.includes('image/png')) {
            const chunk = new Uint8Array(21);
            const view = new DataView(chunk.buffer);
            view.setUint32(0, 9);
            chunk.set([112, 72, 89, 115], 4); // pHYs
            view.setUint32(8, Math.round(dpi / 0.0254));
            view.setUint32(12, Math.round(dpi / 0.0254));
            chunk[16] = 1;
            let crc = 0xffffffff;
            for (const byte of chunk.slice(4, 17)) {
                crc ^= byte;
                for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
            }
            view.setUint32(17, (crc ^ 0xffffffff) >>> 0);
            const parts = [bytes.slice(0, 33), chunk]; // after signature + IHDR
            let offset = 33;
            while (offset + 12 <= bytes.length) {
                const length = new DataView(bytes.buffer).getUint32(offset) + 12;
                if (offset + length > bytes.length) throw new Error('Invalid PNG chunk');
                if (String.fromCharCode(...bytes.slice(offset + 4, offset + 8)) !== 'pHYs') {
                    parts.push(bytes.slice(offset, offset + length));
                }
                offset += length;
            }
            output = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
            offset = 0;
            parts.forEach(part => { output.set(part, offset); offset += part.length; });
        } else if (prefix.includes('image/jpeg')) {
            output = bytes;
            const jfif = bytes[2] === 255 && bytes[3] === 224
                && String.fromCharCode(...bytes.slice(6, 11)) === 'JFIF\0';
            if (!jfif) {
                const header = Uint8Array.from([255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 1,
                    dpi >> 8, dpi & 255, dpi >> 8, dpi & 255, 0, 0]);
                output = new Uint8Array(bytes.length + header.length);
                output.set(bytes.slice(0, 2)); output.set(header, 2); output.set(bytes.slice(2), 20);
            } else {
                output[13] = 1;
                new DataView(output.buffer).setUint16(14, dpi);
                new DataView(output.buffer).setUint16(16, dpi);
            }
        } else return dataUrl;
        const blocks = [];
        for (let i = 0; i < output.length; i += 8192) blocks.push(String.fromCharCode(...output.subarray(i, i + 8192)));
        return prefix + ',' + btoa(blocks.join(''));
    }
    window.GWASMindFiles = { download, withDpi };

    window.openPlotExport = function () {
        const plot = window.ManhattanPlot;
        if (!plot?._plotInitialized) throw new Error('Load a plot before downloading');
        const modal = dialog('Download plot');
        const format = choice(modal, 'Format', [['png', 'PNG'], ['jpg', 'JPG'], ['svg', 'SVG'], ['pdf', 'PDF']]);
        const dpi = choice(modal, 'Resolution (DPI)', [['300', '300'], ['600', '600'], ['150', '150'], ['96', '96'], ['72', '72']]);
        const status = element('p', 'dialog-status', 'Raster image width: 12 inches. Current zoom and gene labels are retained.');
        const submit = element('button', 'dialog-submit', 'Download');
        modal.append(status, submit);
        format.onchange = () => { dpi.parentElement.hidden = !['png', 'jpg'].includes(format.value); };
        submit.onclick = async () => {
            submit.disabled = true;
            status.classList.remove('dialog-error');
            status.textContent = 'Preparing download…';
            try {
                const ok = await plot.exportAs(format.value, { dpi: Number(dpi.value) });
                if (!ok) throw new Error('Plot export failed. Please retry after the plot finishes rendering.');
                modal.close();
            } catch (error) {
                status.textContent = error.message;
                status.classList.add('dialog-error');
            } finally { submit.disabled = false; }
        };
    };

    window.runPlotTool = function (tool) {
        const plot = window.ManhattanPlot;
        if (!plot?._plotInitialized) return;
        const figure = plot.container;
        const update = {};
        if (['pan', 'zoom', 'lasso', 'select'].includes(tool)) {
            plot.dragMode = tool;
            update.dragmode = tool;
            document.querySelectorAll('.plot-tools button[data-tool]').forEach(button => {
                button.setAttribute('aria-pressed', String(button.dataset.tool === tool));
            });
        } else if (['in', 'out'].includes(tool)) {
            const factor = tool === 'in' ? .75 : 1 / .75;
            for (const axis of ['xaxis', 'yaxis']) {
                const range = figure._fullLayout[axis].range;
                const center = (range[0] + range[1]) / 2;
                const half = (range[1] - range[0]) * factor / 2;
                update[axis + '.range'] = [center - half, center + half];
            }
        }
        return plot.enqueuePlotOperation('Unable to update plot tool', () => Plotly.relayout(figure, update));
    };

    function installResizer() {
        const main = document.getElementById('main-app');
        const handle = document.getElementById('panel-resizer');
        if (!main || !handle) return;
        let dragging = false, timer = null, resizing = false, resizeAgain = false;
        const resize = async () => {
            timer = null;
            if (resizing) { resizeAgain = true; return; }
            const figure = document.getElementById('manhattan-plot');
            if (!figure?._fullLayout) return;
            resizing = true;
            try { await Plotly.Plots.resize(figure); }
            catch (error) { console.warn('Plot resize failed:', error); }
            finally {
                resizing = false;
                if (resizeAgain) { resizeAgain = false; schedule(0); }
            }
        };
        const schedule = (delay = 100) => {
            if (timer !== null) clearTimeout(timer);
            timer = setTimeout(resize, delay);
        };
        const setWidth = width => {
            const available = main.clientWidth - 40;
            const value = Math.max(Math.min(300, available * .5), Math.min(available * .7, available - 300, width));
            main.style.setProperty('--panel-width', Math.round(value) + 'px');
            handle.setAttribute('aria-valuenow', String(Math.round(value / available * 100)));
            // Debounce GPU layout during pointer motion; always render the final width.
            schedule();
        };
        handle.addEventListener('pointerdown', event => {
            if (event.button !== 0) return;
            dragging = true;
            handle.setPointerCapture(event.pointerId);
            main.classList.add('is-resizing');
            event.preventDefault();
        });
        window.addEventListener('pointermove', event => {
            if (dragging) setWidth(main.getBoundingClientRect().right - event.clientX - 20);
        });
        const finish = () => {
            if (!dragging) return;
            dragging = false;
            main.classList.remove('is-resizing');
            schedule(0);
        };
        window.addEventListener('pointerup', finish);
        window.addEventListener('pointercancel', finish);
        window.addEventListener('blur', finish);
        handle.addEventListener('keydown', event => {
            if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
            event.preventDefault();
            const width = document.getElementById('right-panel').getBoundingClientRect().width;
            if (event.key === 'Home') {
                main.style.removeProperty('--panel-width');
                handle.setAttribute('aria-valuenow', '33');
                schedule();
            } else setWidth(width + (event.key === 'ArrowLeft' ? 30 : -30));
        });
        if (typeof ResizeObserver !== 'undefined') {
            const observer = new ResizeObserver(() => schedule());
            observer.observe(main.querySelector('.plot-wrapper'));
        }
    }

    window.decorateReportStatus = function (target) {
        if (!target || target.querySelector('.report-status-details')) return;
        const nodes = Array.from(target.children), quality = nodes.findIndex(node => node.classList.contains('report-quality'));
        const end = nodes.findIndex((node, index) => index > quality && node.textContent.trim() === 'Summary');
        if (quality < 0 || end < 0) return;
        const metadata = nodes.slice(quality + 1, end).filter(node => /^(?:Analysis status|Ranking review|Report fact check|Explanation source|Analysis scope|LLM controls):/.test(node.textContent.trim()));
        if (!metadata.length) return;
        const text = metadata.map(node => node.textContent).join(' ');
        const count = text.match(/\b(\d+) ranked\b/), candidates = text.match(/\b(\d+) candidate genes\b/);
        const ranking = target.scientificReviewReceipt?.ranking_review;
        const flagged = ranking?.fact_issues?.unresolved_count;
        const needsReview = /Ranking review:\s*Needs review/.test(text);
        const note = element('span', 'report-status-summary');
        note.textContent = ' · ' + (count && Number(count[1]) > 0 ? `${count[1]} genes prioritised${candidates ? ` from ${candidates[1]} candidates` : ''}` : 'See analysis details')
            + (needsReview ? flagged ? `; ${flagged} ranking statements flagged below.` : '; ranking review details below.' : '.');
        nodes[quality].appendChild(note);
        const details = element('details', 'report-status-details');
        details.appendChild(element('summary', '', 'Analysis details'));
        for (const node of metadata) {
            const line = element('p', '', node.textContent);
            details.appendChild(line); node.remove();
        }
        if (needsReview && !ranking) details.appendChild(element('p', '', 'Statement-level ranking findings are unavailable in this saved view.'));
        target.insertBefore(details, nodes[end]);
    };

    window.decorateReportWarnings = function (target, markdown, receipt = target?.scientificReviewReceipt) {
        if (receipt && target && typeof target.insertBefore === 'function' && !target.querySelector(':scope > .report-warning')) {
            const review = window.GWASMindProgress?.publicReviewUpdate?.(receipt);
            const structured = window.GWASMindProgress?.createReportReview?.(receipt);
            const nodes = Array.from(target.children);
            const quality = nodes.findIndex(node => /^(?:Report quality|Validation status):\s*(?:PASS|PASS_WITH_WARNINGS|STOPPED)$/.test(String(node.textContent || '').trim()));
            if (review && (Array.isArray(review.academic_warnings) || review.output_review) && quality >= 0) {
                if (structured) target.insertBefore(structured, nodes[quality + 1] || null);
                const end = nodes.findIndex((node, index) => index > quality && String(node.textContent || '').trim() === 'Summary');
                if (end > quality + 1 && /^Warning:/.test(String(nodes[quality + 1]?.textContent || '').trim())) {
                    // The structured W cards are the user-facing warning list.
                    // Keep the original interpretation/audit prose in the saved
                    // receipt and Markdown, without another renamed disclosure.
                    for (const node of nodes.slice(quality + 1, end)) node.remove();
                }
                return true;
            }
        }
        // Only the program's leading warning block is a presentation notice.
        // Never infer a warning section from words appearing in scientific prose.
        if (!/^\s*\*\*(?:Report quality|Validation status):\s*PASS_WITH_WARNINGS\*\*\s*\n\s*\n\*\*Warning:\*\*[^]*?\n\s*\n\*\*Summary\*\*(?:\s*\n|\s*$)/.test(String(markdown || ''))
                || !target || typeof target.insertBefore !== 'function'
                || target.querySelector(':scope > .report-warning')) return false;
        const nodes = Array.from(target.children);
        const quality = nodes.findIndex(node => /^(?:Report quality|Validation status):\s*PASS_WITH_WARNINGS$/.test(String(node.textContent || '').trim()));
        if (quality < 0 || !/^Warning:/.test(String(nodes[quality + 1]?.textContent || '').trim())) return false;
        const end = nodes.findIndex((node, index) => index > quality + 1 && (String(node.textContent || '').trim() === 'Summary'
            || String(node.textContent || '').trim().startsWith('Analysis status:')));
        if (end < 0) return false;
        const block = element('div', 'report-warning');
        const warningText = nodes.slice(quality + 1, end).map(node => String(node.textContent || '')).join(' ');
        const categories = window.GWASMindProgress?.reportWarningCategories?.(warningText) || [];
        if (/^Warning:\s*Review limitations: /.test(warningText)
                && target.querySelector('a[href="#gwasmind-review-details"]')) {
            // The program already supplied a short opening. Keep its explicit
            // interpretation status visible; complete reasons are below.
            block.append(element('p', 'report-warning-summary', warningText.replace(/^Warning:\s*/, '')));
            target.insertBefore(block, nodes[quality + 1]);
            for (const node of nodes.slice(quality + 1, end)) node.remove();
            return true;
        }
        const summaryText = categories.length ? 'Warnings: ' + categories.slice(0, 3).join(' · ')
            + (categories.length > 3 ? ` · ${categories.length - 3} more areas` : '') : 'Warnings affect the interpretation of these findings.';
        block.append(element('p', 'report-warning-summary', summaryText));
        const details = element('details', 'report-warning-details');
        details.append(element('summary', '', 'View full warning details'));
        const body = element('div', 'report-warning-body');
        details.append(body); block.append(details);
        target.insertBefore(block, nodes[quality + 1]);
        // Move the already sanitized DOM intact. Markdown, scientific text,
        // links, and the original copy/export source remain unchanged.
        for (const node of nodes.slice(quality + 1, end)) body.append(node);
        return true;
    };

    window.decorateReportReviewDetails = function (target, markdown) {
        if (!target?.querySelectorAll || !/^\s*\*\*(?:Report quality|Validation status):\s*PASS_WITH_WARNINGS\*\*/.test(String(markdown || ''))) return;
        for (const link of target.querySelectorAll('a[href="#gwasmind-review-details"]')) {
            if (link.textContent.trim() !== 'Review details') continue;
            const marker = link.parentElement;
            const list = marker?.nextElementSibling;
            if (marker?.parentElement !== target || marker.tagName?.toUpperCase() !== 'P'
                    || list?.tagName?.toUpperCase() !== 'UL' || !list.children.length) continue;
            const prior = Array.from(target.children).slice(0, Array.from(target.children).indexOf(marker));
            const headings = prior.map(node => node.textContent.trim());
            if (!headings.includes('Limitations & Uncertainty') || headings.lastIndexOf('References') > headings.lastIndexOf('Limitations & Uncertainty')) continue;
            const review = window.GWASMindProgress?.publicReviewUpdate?.(target.scientificReviewReceipt);
            if (review && Array.isArray(review.academic_warnings)) {
                // These audit notes must not reappear as a second warning list.
                marker.remove(); list.remove(); continue;
            }
            // Move sanitized list nodes intact; do not admit any new HTML tags
            // or parse model text as markup a second time.
            const details = element('details', 'report-warning-details report-review-details');
            details.append(element('summary', '', 'Review details'));
            const academicGroups = new Map();
            for (const item of Array.from(list.children)) {
                const match = /^\s*(W[1-9])\s*[·:–—-]\s*/.exec(String(item.textContent || ''));
                const label = match && window.GWASMindProgress?.academicWarningLabel?.(match[1]);
                if (!label) continue;
                let group = academicGroups.get(match[1]);
                if (!group) {
                    const wrapper = element('li', 'report-academic-group');
                    const disclosure = element('details', 'academic-warning');
                    disclosure.append(element('summary', '', `${match[1]} · ${label}`));
                    const body = element('ul'); disclosure.append(body); wrapper.append(disclosure);
                    list.insertBefore(wrapper, item);
                    group = { body, messages: new Set() }; academicGroups.set(match[1], group);
                }
                const message = String(item.textContent || '').trim();
                if (group.messages.has(message)) item.remove();
                else { group.messages.add(message); group.body.append(item); }
            }
            target.insertBefore(details, marker);
            details.append(list);
            marker.remove();
        }
    };

    window.decorateQtlEvidence = function (target, { cards = target?.geneEvidenceCards,
        normalizations = target?.reportVariantNormalizations } = {}) {
        const api = window.GWASMindProgress;
        if (!target?.querySelectorAll || !api?.createQtlDisplayRow) return;
        const aliases = new Map((api.publicVariantNormalizations?.(normalizations) || []).map(row => [row.original_id, row.canonical_id]));
        const rows = (cards || []).flatMap(card => card.categories?.qtl?.rows || []);
        const pattern = /^\s*((?:sc-)?(?:e|s|p|tu|mol)?QTL)\s*\((.+?)\)\s*:\s*(.*?)\s*\(displayed statistics:\s*representative (source-selected variant|LD proxy|analyzed variant)\s+([0-9XYMT]+_\d+_[ACGT]+_[ACGT]+)(?:\s*;\s*study\s+([^\s)]+))?\s*\)/i;
        const leadPattern = /^\s*((?:sc-)?(?:e|s|p|tu|mol)?QTL)\s*\((.+?)\)\s*:\s*(.*?)\s*\(displayed statistics:\s*(credible-set lead variant)\s+([0-9XYMT]+_\d+_[ACGT]+_[ACGT]+)(?:\s*;\s*study\s+([^\s;)]+))?(?:\s*;\s*locus overlap:\s*([^)]*))?\s*\)/i;
        for (const item of target.querySelectorAll('li')) {
            let source = item.qtlDisplaySource;
            if (!source) {
                const sourceMarker = [...item.querySelectorAll('a')].find(anchor =>
                    anchor.getAttribute('href') === '#gwasmind-evidence-source');
                const measured = (sourceMarker?.getAttribute('title') || String(item.textContent || ''))
                    .replace(/`/g, '').replace(/^\s*[-*+]\s+/, '');
                const match = leadPattern.exec(measured) || pattern.exec(measured);
                if (!match) continue;
                const origin = match[4].toLowerCase();
                const stats = match[3];
                source = { qtl_type: match[1], biosample: match[2], locus_variant_id: match[5], variant_id: match[5],
                    variant_origin: origin === 'source-selected variant' ? 'source_selected' : origin === 'ld proxy' ? 'ld_proxy' : 'unknown',
                    study_id: match[6] || '', p_value: /\bP\s*=\s*([^,\s]+)/i.exec(stats)?.[1] || '',
                    beta: /\bBeta\s*=\s*([^,\s]+)/i.exec(stats)?.[1] || '', provenance: [], citations: [] };
                if (origin === 'credible-set lead variant') {
                    source.measurement_scope = 'credible_set_lead'; source.cs_lead_variant_id = match[5];
                    source.locus_variant_id = '';
                    source.locus_anchor_witnesses = [...(match[7] || '').matchAll(/(user-selected variant|LD-linked variant|analyzed variant)\s+([0-9XYMT]+_\d+_[ACGT]+_[ACGT]+)/gi)]
                        .map(witness => ({ variant_id: witness[2], locus_variant_id: witness[2],
                            variant_origin: witness[1].toLowerCase() === 'user-selected variant' ? 'source_selected'
                                : witness[1].toLowerCase() === 'ld-linked variant' ? 'ld_proxy' : 'unknown' }));
                    if (!source.locus_anchor_witnesses.length && sourceMarker) {
                        for (const group of measured.matchAll(/\b(user-selected variants|LD-linked variants|analyzed variants):\s*([^;]+)/gi)) {
                            const variantOrigin = group[1].toLowerCase() === 'user-selected variants' ? 'source_selected'
                                : group[1].toLowerCase() === 'ld-linked variants' ? 'ld_proxy' : 'unknown';
                            for (const witness of group[2].matchAll(/\b[0-9XYMT]+_\d+_[ACGT]+_[ACGT]+\b/g))
                                source.locus_anchor_witnesses.push({ variant_id: witness[0], locus_variant_id: witness[0], variant_origin: variantOrigin });
                        }
                        const betaOrientation = measured.match(/; Beta: (.*)$/)?.[1];
                        if (betaOrientation) source.provenance.push(`Beta: ${betaOrientation}`);
                    }
                }
                for (const anchor of item.querySelectorAll('a')) {
                    if (anchor.getAttribute('href') === '#gwasmind-evidence-source') continue;
                    if (anchor.getAttribute('href') === '#gwasmind-evidence-origin')
                        source.provenance.push(...(anchor.getAttribute('title') || '').split('; '));
                    else source.citations.push(anchor);
                }
                for (const disclosure of item.querySelectorAll('.evidence-origin-details'))
                    for (const entry of disclosure.querySelectorAll('p')) source.provenance.push(String(entry.textContent || ''));
                item.qtlDisplaySource = source;
            }
            const bound = rows.filter(row => row.study_id === source.study_id
                && (source.measurement_scope === 'credible_set_lead'
                    ? row.measurement_scope === 'credible_set_lead' && row.variant_id === source.variant_id
                    : row.locus_variant_id === source.variant_id || row.variant_id === source.variant_id)
                && row.qtl_type?.toLowerCase() === source.qtl_type.toLowerCase()
                && row.biosample === source.biosample);
            const detail = bound.length === 1 ? bound[0] : {};
            // The visible P/beta always stay attached to the original report's
            // measured representative; a card only supplies its proven identity/details.
            const verifiedDetail = detail.variant_id === source.variant_id || detail.variant_origin === source.variant_origin;
            const variantId = (verifiedDetail && detail.variant_id)
                || (source.variant_origin === 'source_selected' && aliases.get(source.variant_id)) || source.variant_id;
            const row = { ...detail, ...source, variant_id: variantId,
                ...(source.measurement_scope === 'credible_set_lead' && detail.locus_anchor_witnesses
                    ? { locus_anchor_witnesses: detail.locus_anchor_witnesses } : {}),
                variant_label: source.measurement_scope === 'credible_set_lead' ? 'Credible-set lead'
                    : { source_selected: 'User-selected variant', ld_proxy: 'LD-linked variant' }[source.variant_origin] || 'Analyzed variant' };
            item.replaceChildren(api.createQtlDisplayRow(row, { citations: source.citations, provenance: source.provenance }));
            item.classList.add('qtl-evidence-item');
        }
        if (target.querySelector('.qtl-evidence-item') && !target.querySelector('.qtl-origin-legend')) {
            const legend = element('p', 'qtl-origin-legend', 'LD-linked variant: an additional variant included through LD with a user-selected variant.');
            const first = target.querySelector('.qtl-evidence-item');
            const list = first.parentElement;
            list?.parentElement?.insertBefore(legend, list);
        }
    };

    window.decorateEvidenceOrigins = function (target) {
        if (!target?.querySelectorAll) return;
        window.decorateQtlEvidence(target);
        for (const link of [...target.querySelectorAll('a')].filter(anchor =>
            anchor.getAttribute('href') === '#gwasmind-evidence-source')) {
            const description = link.getAttribute('title');
            if (link.textContent.trim() !== 'Source' || !description || !link.parentElement) continue;
            const details = element('details', 'qtl-source-details evidence-source-details');
            const summary = element('summary', '', 'Source');
            const body = element('div', 'qtl-source-body');
            body.append(element('p', '', description)); details.append(summary, body);
            const parent = link.parentElement;
            parent.insertBefore(details, link); link.remove();
            // Older saved Markdown put Source before its trailing citations.
            // Move only this disclosure, preserving the original reference links.
            const following = [...(parent.children || [])].slice([...(parent.children || [])].indexOf(details) + 1);
            const citations = following.filter(node => node.tagName?.toLowerCase() === 'a'
                && /^\[?\d+\]?$/.test(String(node.textContent || '').trim()));
            if (citations.length) {
                const next = citations.at(-1).nextSibling;
                if (next) parent.insertBefore(details, next); else parent.appendChild(details);
            }
        }
        for (const link of target.querySelectorAll('a[href="#gwasmind-evidence-origin"]')) {
            const label = String(link.textContent || '').trim();
            const detail = link.getAttribute('title') || '';
            if (!['Source + LD proxy evidence', 'Indirect LD proxy evidence'].includes(label)
                    || !detail.startsWith('Source SNPs: ') || !detail.includes('; Verified LD proxies: ')) continue;
            const disclosure = element('details', 'evidence-origin-details');
            disclosure.append(element('summary', '', label));
            const content = element('div', 'evidence-origin-body');
            for (const text of detail.split('; ')) content.append(element('p', '', text));
            disclosure.append(content);
            link.replaceWith(disclosure);
        }
    };

    const reportPartTitles = new Set([
        'Summary', 'Biological Context', 'Evidence', 'Limitations & Uncertainty',
        'Limitations and Uncertainty', 'References', 'Interpretation', 'Conclusion',
        'Conclusions', 'Methods', 'Next Steps', 'Recommendations', 'Details', 'Limitations'
    ]);
    const reportRootTitles = new Set(['Summary', 'Biological Context', 'Evidence',
        'Limitations & Uncertainty', 'Limitations and Uncertainty', 'References']);

    function reportHeading(node) {
        const tag = String(node.tagName || '').toUpperCase();
        if (!/^H[1-6]$/.test(tag) && tag !== 'P') return null;
        const title = String(node.textContent || '').trim().replace(/:$/, '');
        if (reportRootTitles.has(title) || (tag === 'P' && reportPartTitles.has(title))) return { title, level: 2 };
        // Semantic headings apply to every intent: variants, studies, targets,
        // diseases, drugs, or general explanations. Entity-specific compatibility
        // below is limited to legacy bold paragraphs without heading semantics.
        if (/^H[1-6]$/.test(tag)) return { title, level: Math.max(2, Number(tag.slice(1))) };
        if (/\bENSG\d{11}\b/.test(title) && title.length < 180
                && node.querySelector?.('strong')) return { title, level: 3 };
        if (/^\d+\.\s+(?:Molecular QTL(?:\/Colocalization)?|Colocalization|Regulatory Region-to-Gene(?: \(E2G\))?|Expression|Function|Coding(?: and Splicing)?|Genetic Associations|Disease Associations)$/i.test(title)) {
            return { title, level: 4 };
        }
        return null;
    }

    window.decorateReportSections = function (target, { collapsible = true } = {}) {
        if (!target?.children || target.querySelector?.(':scope > .report-section')) return;
        const nodes = Array.from(target.children);
        // Older saved reports placed context under Evidence and used bold
        // paragraphs for their main headings. Normalize presentation only.
        const summaryIndex = nodes.findIndex(node => reportHeading(node)?.title === 'Summary');
        if (summaryIndex < 0 && !nodes.some(node => reportHeading(node))) return;
        const contextIndex = nodes.findIndex(node => reportHeading(node)?.title === 'Biological Context');
        if (summaryIndex >= 0 && contextIndex >= 0) {
            let contextEnd = contextIndex + 1;
            while (contextEnd < nodes.length && !reportHeading(nodes[contextEnd])) contextEnd++;
            const contextNodes = nodes.splice(contextIndex, contextEnd - contextIndex);
            const summaryStart = nodes.findIndex(node => reportHeading(node)?.title === 'Summary');
            let afterSummary = summaryStart + 1;
            while (afterSummary < nodes.length && (reportHeading(nodes[afterSummary])?.level || 7) > 2) afterSummary++;
            nodes.splice(afterSummary, 0, ...contextNodes);
        }
        const stack = [];
        let summaryReached = false;
        for (const node of nodes) {
            const section = reportHeading(node);
            if (!section) {
                (stack.at(-1)?.body || target).appendChild(node);
                continue;
            }
            while (stack.length && stack.at(-1).level >= section.level) stack.pop();
            const heading = element(`h${section.level}`, 'report-part-heading');
            // Move already sanitized child nodes, preserving emphasis, IDs and
            // links without interpreting model text as HTML a second time.
            if (node.firstChild) while (node.firstChild) heading.appendChild(node.firstChild);
            else heading.textContent = String(node.textContent || '').trim();
            if (node.id) heading.id = node.id;
            if (!collapsible) {
                target.appendChild(heading); node.remove();
                continue;
            }
            const details = element('details', 'report-section');
            details.dataset.level = String(section.level);
            details.dataset.part = section.title;
            details.open = !summaryReached || section.title === 'Summary';
            if (section.title === 'Summary' || summaryIndex < 0) summaryReached = true;
            const disclosure = element('summary', 'report-part-toggle');
            disclosure.appendChild(heading);
            const body = element('div', 'report-part-body');
            details.append(disclosure, body);
            (stack.at(-1)?.body || target).appendChild(details);
            stack.push({ level: section.level, body });
            node.remove();
        }
        target.classList.add('structured-report');
    };

    function reportReviewText(value) {
        // Review facts are literal text in the UI. Preserve that contract in
        // Markdown exports instead of interpreting provider text as markup.
        return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/[\\`*_{}\[\]]/g, '\\$&');
    }

    const legacyPriorReportReference = /^\[(\d+)\] Published GWAS supporting the exact selected lead-SNP association, retrieved through Open Targets\. (https:\/\/pubmed\.ncbi\.nlm\.nih\.gov\/(\d+)\/?)[.]?$/;

    function compactPriorReportReference(line) {
        const match = legacyPriorReportReference.exec(line.trim());
        return match ? `[${match[1]}] [PMID: ${match[3]}](${match[2]})` : line;
    }

    window.decoratePriorReportReferences = function (target) {
        // Older reports retained only the PMID. Preserve its identity and number
        // without repeating the retrieval provider as if it were the publication.
        for (const paragraph of target.querySelectorAll('p')) {
            if (paragraph.closest?.('pre, code, blockquote')) continue;
            const match = legacyPriorReportReference.exec(paragraph.textContent.trim());
            if (!match) continue;
            const link = element('a', '', `PMID: ${match[3]}`);
            link.href = match[2]; link.target = '_blank'; link.rel = 'noopener noreferrer';
            paragraph.replaceChildren(document.createTextNode(`[${match[1]}] `), link);
        }
    };

    function reportExportSource(target) {
        let fenced = false;
        const original = String(target.dataset.reportMarkdown || '').split('\n').map(line => {
            if (/^\s*```/.test(line)) { fenced = !fenced; return line; }
            if (!fenced) line = compactPriorReportReference(line);
            if (fenced || !/^\s*[-*]\s/.test(line)) return line;
            return line.replace(/(\[Source\]\(#gwasmind-evidence-source "(?:[^"\\\n]|\\.)*"\))((?:[ \t]+\[\[\d+\]\]\(https?:\/\/[^\s)]+\))+)/g,
                (_match, source, citations) => citations.trim() + ' ' + source);
        }).join('\n');
        const review = window.GWASMindProgress?.publicReviewUpdate?.(target.scientificReviewReceipt);
        if (!review) return original;
        const sections = [];
        for (const item of review.academic_warnings || []) {
            const code = item.rule_codes?.length ? item.rule_codes.join(', ') : item.code;
            const lines = [`**${code} · ${reportReviewText(item.label)}**`];
            if (item.facts.length) lines.push('**Observed facts**', ...item.facts.map(fact => '- ' + reportReviewText(fact)));
            if (item.impact) lines.push('**What this limits:** ' + reportReviewText(item.impact));
            for (const [key, label] of Object.entries({ variant_ids: 'Variants', gene_ids: 'Genes', study_ids: 'Studies', modules: 'Retrieval modules' })) {
                const values = item.affected?.[key] || [], count = Math.max(item.affected_count?.[key] || 0, values.length);
                if (!count) continue;
                lines.push(`**${label} (${count}):** ` + (values.length ? values.map(reportReviewText).join(' · ') : 'Identifiers not included in the saved review.'));
                if (count > values.length && values.length) {
                    const missing = count - values.length;
                    lines.push(`${missing} additional identifier${missing === 1 ? ' was' : 's were'} not included in the saved review.`);
                }
            }
            if (item.refs.length) lines.push('**Evidence and retrieval records:** ' + item.refs.map(ref => reportReviewText(
                ref.type === 'reference' ? ref.id : `${ref.type}: ${ref.id}`)).join(' · '));
            sections.push(lines.join('\n\n'));
        }
        let warnings = sections.length ? `**Scientific evidence review · ${sections.length} limitation${sections.length === 1 ? '' : 's'}**\n\n${sections.join('\n\n')}\n\n` : '';
        const output = review.output_review, reportFacts = output?.fact_issues;
        const rankingFacts = review.ranking_review?.fact_issues;
        const facts = rankingFacts ? { items: [...(reportFacts?.items || []), ...rankingFacts.items],
            unresolved_count: (reportFacts?.unresolved_count || 0) + rankingFacts.unresolved_count,
            corrected_count: (reportFacts?.corrected_count || 0) + rankingFacts.corrected_count,
            omitted_count: (reportFacts?.omitted_count || 0) + rankingFacts.omitted_count } : reportFacts;
        const factSections = [];
        for (const disposition of ['unverified', 'corrected']) {
            const items = facts?.items.filter(item => item.disposition === disposition) || [];
            if (!items.length) continue;
            const heading = disposition === 'unverified'
                ? `Objective fact verification · ${window.GWASMindProgress.factIssueSummary(facts)}`
                : `${facts.corrected_count} factual correction${facts.corrected_count === 1 ? '' : 's'} applied`;
            const lines = [`**${heading}**`];
            for (const item of items) {
                lines.push(`**${reportReviewText(window.GWASMindProgress.factIssueTitle(item))}**`);
                for (const [label, value] of [
                    ['Where', [item.target === 'candidate_ranking' ? 'Gene prioritization' : 'Final report', item.location].filter(Boolean).join(' · ')],
                    ['LLM-stated value', item.original_value],
                    [disposition === 'unverified' ? 'Why not verified' : 'Why corrected', item.reason],
                    ['How handled', window.GWASMindProgress.factIssueHandling(item)],
                    ['Original statement', item.excerpt], ['Variants', item.variant_ids?.join(' · ')],
                    ['Evidence records', item.evidence_ids?.join(' · ')],
                    ['Numeric source facts', item.numeric_fact_ids?.join(' · ')]
                ]) if (value) lines.push(`**${label}:** ${reportReviewText(value)}`);
            }
            factSections.push(lines.join('\n\n'));
        }
        if (facts?.omitted_count) factSections.push(`${facts.omitted_count} additional fact-review items are retained in the review record.`);
        if (output?.status === 'PASS_WITH_WARNINGS' && !facts?.unresolved_count) {
            factSections.push('**Objective fact verification needs attention**', ...(output.warnings?.length ? output.warnings
                : ['The saved review did not include the affected facts. Open the original analysis record for details.']).map(reportReviewText));
        }
        if (factSections.length) warnings += factSections.join('\n\n') + '\n\n';
        let source = original;
        // These exact program markers are the same audit-only material removed
        // by the page decorators. Never search arbitrary scientific prose for
        // warning words or change the original Markdown stored on the report.
        const quality = /^\*\*(?:Report quality|Validation status):[ \t]*(?:PASS|PASS_WITH_WARNINGS|STOPPED)\*\*[ \t]*\r?$/m.exec(source);
        const summary = /^(?:#{1,6}[ \t]+)?(?:\*\*Summary\*\*|Summary)[ \t]*\r?$/m.exec(source);
        if (quality && summary && quality.index < summary.index) {
            const start = quality.index + quality[0].length;
            const metadata = source.slice(start, summary.index);
            const genericWarning = /^\*\*Warning:\*\* Review limitations:[^\r\n]*(?:\r?\n|$)(?:[ \t]*\r?\n)*/m;
            const reviewed = genericWarning.test(metadata) ? metadata.replace(genericWarning, () => warnings) : metadata + warnings;
            source = source.slice(0, start) + reviewed + source.slice(summary.index);
        } else if (warnings) source = warnings + source;
        const limits = source.indexOf('**Limitations & Uncertainty**');
        const references = source.indexOf('**References**', Math.max(0, limits));
        const marker = source.indexOf('[Review details](#gwasmind-review-details)', Math.max(0, limits));
        if (limits >= 0 && marker > limits && (references < 0 || marker < references)) {
            const afterMarker = source.slice(marker);
            const auditList = /^\[Review details\]\(#gwasmind-review-details\)[ \t]*\r?\n(?:[ \t]*\r?\n)*((?:[-*] [^\r\n]*(?:\r?\n|$)(?:[ \t]*\r?\n)*)+)/.exec(afterMarker);
            if (auditList) source = source.slice(0, marker) + afterMarker.slice(auditList[0].length);
        }
        return source;
    }

    window.reportExportSource = reportExportSource;

    window.attachReportActions = function (target, markdown) {
        if (!target || !String(markdown || '').trim()) return;
        target.dataset.reportMarkdown = String(markdown);
        window.decorateReportWarnings(target, markdown);
        window.decorateReportReviewDetails(target, markdown);
        window.decorateEvidenceOrigins(target);
        window.decoratePriorReportReferences(target);
        window.decorateReportSections(target);
        target.querySelector(':scope > .report-actions')?.remove();
        const bar = element('div', 'report-actions');
        const copy = element('button', '', 'Copy');
        const exportButton = element('button', '', 'Export');
        copy.type = exportButton.type = 'button';
        bar.append(copy, exportButton);
        target.prepend(bar);
        copy.onclick = async event => {
            event.stopPropagation();
            const raw = reportExportSource(target);
            const source = window.GWASMindVariants?.markdown(raw, target.variantLabels) || raw;
            try {
                await navigator.clipboard.writeText(source);
                copy.textContent = 'Copied';
            } catch (error) {
                const text = element('textarea');
                text.value = source;
                text.style.cssText = 'position:fixed;left:-9999px;';
                document.body.append(text); text.select();
                const copied = document.execCommand('copy'); text.remove();
                copy.textContent = copied ? 'Copied' : 'Copy failed';
            }
            setTimeout(() => { copy.textContent = 'Copy'; }, 1600);
        };
        exportButton.onclick = event => {
            event.stopPropagation();
            const modal = dialog('Export response');
            const format = choice(modal, 'Format', [['md', 'Markdown (.md)'], ['html', 'Formatted HTML (.html)'], ['txt', 'Plain text (.txt)']]);
            const submit = element('button', 'dialog-submit', 'Export');
            modal.append(submit);
            submit.onclick = () => {
                const raw = reportExportSource(target);
                const source = window.GWASMindVariants?.markdown(raw, target.variantLabels) || raw;
                let content = source, mime = 'text/markdown;charset=utf-8';
                if (format.value === 'html') {
                    const holder = element('div');
                    // Bind QTL statistics and Source identity before replacing only
                    // the visible variant label. The parser owns canonical IDs.
                    holder.innerHTML = window.GWASMindSafe.renderMarkdown(raw);
                    holder.geneEvidenceCards = target.geneEvidenceCards;
                    holder.reportVariantNormalizations = target.reportVariantNormalizations;
                    window.decorateReportReviewDetails(holder, raw);
                    window.decorateEvidenceOrigins(holder);
                    window.decorateReportSections(holder, { collapsible: false });
                    window.GWASMindVariants?.decorate(holder, target.variantLabels);
                    content = '<!doctype html><html lang="en"><meta charset="utf-8"><title>GWASMind report</title><style>body{max-width:900px;margin:40px auto;font:16px/1.6 system-ui;color:#27384a;padding:0 24px}a{color:#4d6f90}h1{font-size:1.5rem}h2{font-size:1.25rem}h3{font-size:1.1rem}h4,h5,h6{font-size:1rem}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}.qtl-compact-line{display:flex;flex-wrap:wrap;align-items:baseline}.qtl-separator{white-space:pre;color:#9aa9b5}.qtl-variant-origin{font-size:.8em;border:1px solid #cbd6df;padding:0 4px}.qtl-source-details{font-size:.86em}.qtl-source-details[open]{flex-basis:100%}.qtl-source-body{padding:6px 10px;background:#f4f7fa;overflow-wrap:anywhere}.qtl-source-body p{margin:3px 0}.qtl-origin-legend{font-size:.85em;color:#687b8a}</style><body>'
                        + holder.innerHTML + '</body></html>';
                    mime = 'text/html;charset=utf-8';
                } else if (format.value === 'txt') {
                    const holder = element('div');
                    holder.innerHTML = window.GWASMindSafe.renderMarkdown(raw);
                    holder.geneEvidenceCards = target.geneEvidenceCards;
                    holder.reportVariantNormalizations = target.reportVariantNormalizations;
                    window.decorateEvidenceOrigins(holder);
                    window.GWASMindVariants?.decorate(holder, target.variantLabels);
                    content = holder.textContent || holder.innerText;
                    mime = 'text/plain;charset=utf-8';
                }
                download(content, `GWASMind_report_${new Date().toISOString().slice(0, 10)}.${format.value}`, mime);
                modal.close();
            };
        };
    };

    let datasetId = null, sourcePhenotype = '', conversationId = null, turns = [], datasetGeneration = 0;
    let saveChain = Promise.resolve(), restoring = false, conversationEpoch = 0;
    let historySaveError = null;
    let analysisSaveChain = Promise.resolve();
    let currentRunId = null;
    let currentAnalysisStatus = null;
    let currentComputedFacts = null;
    let currentVariantNormalizations = [];
    let currentRankingEvidence = [];
    let currentGeneEvidenceCards = null;
    let currentVariantLabels = [];
    let currentSelectionSnapshot = null;
    const currentProgress = new Map();
    const idPattern = /^[a-f0-9]{32}$/;

    function syncConversationLocation(id = null) {
        if (!window.history?.replaceState) return;
        const params = new URLSearchParams(location.search);
        params.delete('history_kind'); params.delete('history_id');
        if (idPattern.test(id || '')) {
            params.set('history_kind', 'chats'); params.set('history_id', id);
        }
        const query = params.toString();
        window.history.replaceState(null, '', (location.pathname || '/app') + (query ? '?' + query : '') + (location.hash || ''));
    }

    async function api(path, options) {
        const response = await fetch(path, options);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `History request failed (${response.status})`);
        return payload;
    }

    function saveConversation() {
        if (restoring || !turns.length) return;
        const epoch = conversationEpoch;
        saveChain = saveChain.catch(() => {}).then(async () => {
            const id = epoch === conversationEpoch ? conversationId : null;
            const snapshot = { dataset_id: datasetId, source_phenotype: sourcePhenotype,
                messages: turns.map(turn => ({ ...turn })), status: 'saved' };
            // A conversation is named once, after its opening question. A later
            // save must not rename a record the user already recognises.
            if (!id) snapshot.title = String(turns.find(turn => turn.role === 'user')?.content || 'Chat')
                .split(/\s+/).filter(Boolean).slice(0, 10).join(' ') || 'Chat';
            const result = await api('/api/history/chats' + (id ? '/' + id : ''), {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(snapshot)
            });
            if (epoch === conversationEpoch) {
                conversationId = result.history_id;
                syncConversationLocation(conversationId);
                historySaveError = null;
                const button = document.querySelector('[data-kind="chats"]');
                if (button) { button.title = 'Open saved conversations'; button.textContent = 'History'; }
            }
        }).catch(error => {
            console.warn('Chat history could not be saved:', error.message);
            if (epoch === conversationEpoch) historySaveError = error;
            const button = document.querySelector('[data-kind="chats"]');
            if (button) { button.title = 'History save failed: ' + error.message; button.textContent = 'History · save failed'; }
        });
    }

    async function preserveConversation() {
        await saveChain;
        if (!historySaveError) return;
        // Retry once before replacing the visible conversation. A failed disk
        // save must not turn New Chat or Restore into silent history loss.
        saveConversation();
        await saveChain;
        if (historySaveError) throw new Error('The current conversation has not been saved. '
            + 'Please copy or export it before leaving, or retry when local history is available.');
    }

    window.preserveWorkspaceHistory = async function () {
        await preserveConversation();
        await analysisSaveChain;
    };

    window.recordWorkspaceChatEvent = function (event) {
        if (restoring || !event) return;
        if (event.type === 'run' && idPattern.test(event.run_id || '')) {
            currentRunId = event.run_id;
        } else if (event.type === 'ranking_evidence') {
            const descriptor = window.GWASMindProgress?.publicRankingEvidence?.(event);
            if (!descriptor || descriptor.run_id !== currentRunId) return;
            const key = row => `${row.run_id}:${row.query_index}`;
            const index = currentRankingEvidence.findIndex(row => key(row) === key(descriptor));
            if (index < 0 && currentRankingEvidence.length < 256) currentRankingEvidence.push(descriptor);
            else if (index >= 0) currentRankingEvidence[index] = descriptor;
        } else if (event.type === 'gene_evidence_cards') {
            currentGeneEvidenceCards = window.GWASMindProgress?.publicGeneEvidenceCards?.(event.cards) || null;
            if (Array.isArray(event.variant_labels)) currentVariantLabels = window.GWASMindVariants?.cleanLabels(event.variant_labels) || [];
        } else if (event.type === 'operation_timings') {
            const previous = currentProgress.get(event.stage);
            const timings = window.GWASMindProgress?.publicOperationTimings?.(event.operation_timings);
            if (previous && timings?.length) currentProgress.set(event.stage, { ...previous, operation_timings: timings });
        } else if (event.type === 'variant_normalization') {
            const incoming = window.GWASMindProgress?.publicVariantNormalizations?.(event.normalizations);
            if (!incoming) return;
            const combined = new Map(currentVariantNormalizations.map(item => [item.original_id, item]));
            for (const item of incoming) {
                const previous = combined.get(item.original_id);
                if (previous && (previous.canonical_id !== item.canonical_id || previous.genome_build !== item.genome_build)) return;
                combined.set(item.original_id, item);
            }
            if (combined.size <= 2000) currentVariantNormalizations = [...combined.values()];
        } else if (event.type === 'analysis_status' && event.analysis_status && typeof event.analysis_status === 'object') {
            const fields = ['execution', 'retrieval', 'ranking', 'ranking_model', 'ranking_review_status', 'explanation', 'explanation_model', 'fact_check', 'scope', 'safety_method', 'safety_model_review'];
            currentAnalysisStatus = Object.fromEntries(fields.filter(key => typeof event.analysis_status[key] === 'string'
                && /^[A-Za-z0-9_.:/ -]{0,120}$/.test(event.analysis_status[key])).map(key => [key, event.analysis_status[key]]));
            const facts = event.computed_facts || {};
            currentComputedFacts = Object.fromEntries(['source_variant_count', 'verified_proxy_count', 'analysis_variant_count', 'candidate_pool_count', 'shortlist_count', 'ranked_count', 'displayed_ranked_count']
                .filter(key => Number.isSafeInteger(facts[key]) && facts[key] >= 0).map(key => [key, facts[key]]));
        } else if (event.type === 'review_update' && event.stage === 'safety' && Array.isArray(event.warnings)) {
            const previous = currentProgress.get('safety');
            const details = (previous?.details || []).filter(detail => !detail.startsWith('Safety warning: '));
            details.push(...event.warnings.map(warning => `Safety warning: ${warning}`));
            currentProgress.set('safety', { ...previous, type: 'stage', stage: 'safety', state: event.state,
                summary: event.summary || previous?.summary, details: details.slice(-24),
                metrics: { ...(previous?.metrics || {}), ...event.metrics },
                ...(event.requested_task ? { requested_task: event.requested_task } : {}),
                review_update: { ...event } });
        } else if (event.type === 'stage' && ['plan', 'retrieve', 'safety', 'response'].includes(event.stage)) {
            const previous = currentProgress.get(event.stage);
            const details = previous ? [...previous.details] : [];
            if (event.detail && !details.includes(event.detail)) details.push(event.detail);
            const progress = window.GWASMindProgress?.publicRetrievalProgress?.(event.progress);
            const substeps = [...(previous?.substeps || [])];
            if (progress) {
                const key = window.GWASMindProgress.retrievalStepKey(progress);
                const index = substeps.findIndex(item => window.GWASMindProgress.retrievalStepKey(item) === key);
                if (index < 0) substeps.push(progress);
                else {
                    const terminal = ['completed', 'incomplete', 'failed', 'unavailable', 'blocked', 'not_run', 'cached'];
                    const stale = progress.operation_id && terminal.includes(substeps[index].status) && !terminal.includes(progress.status);
                    if (!stale) substeps[index] = { ...substeps[index], ...progress };
                    if (['completed', 'cached'].includes(progress.status) && !progress.reason) delete substeps[index].reason;
                }
            }
            const substage = ['inference', 'fact_verification'].includes(event.substage) ? event.substage : previous?.substage;
            currentProgress.set(event.stage, { type: 'stage', stage: event.stage, state: event.state,
                ...(window.GWASMindProgress?.publicModelUsage?.(event.model_usage) || previous?.model_usage
                    ? { model_usage: window.GWASMindProgress?.publicModelUsage?.(event.model_usage) || previous.model_usage } : {}),
                ...(event.requested_task || previous?.requested_task ? { requested_task: event.requested_task || previous.requested_task } : {}),
                ...(event.stage === 'response' && ['inference', 'fact_verification'].includes(substage) ? { substage } : {}),
                ...(event.stage === 'response' && (event.ranking_unavailable === true || previous?.ranking_unavailable === true) ? { ranking_unavailable: true } : {}),
                summary: event.summary, details: details.slice(-24),
                metrics: { ...(previous?.metrics || {}), ...event.metrics },
                ...(previous?.review_update ? { review_update: previous.review_update } : {}),
                ...(event.operation_timings || previous?.operation_timings ? { operation_timings:
                    window.GWASMindProgress?.publicOperationTimings?.(event.operation_timings || previous.operation_timings) || [] } : {}),
                ...(progress || previous?.progress ? { progress: progress
                    ? substeps.find(item => window.GWASMindProgress.retrievalStepKey(item)
                        === window.GWASMindProgress.retrievalStepKey(progress)) : previous.progress } : {}),
                ...(substeps.length ? { substeps } : {}) });
            const saved = currentProgress.get(event.stage).metrics;
            if (Number.isInteger(event.metrics?.output_tokens)) delete saved.estimated_output_tokens;
            else if (Number.isInteger(event.metrics?.estimated_output_tokens)) delete saved.output_tokens;
        }
    };
    window.recordWorkspaceChatTurn = function (role, content) {
        if (restoring) return;
        if (role === 'user') { currentRunId = null; currentProgress.clear(); currentAnalysisStatus = null; currentComputedFacts = null; currentVariantNormalizations = []; currentRankingEvidence = []; currentGeneEvidenceCards = null; currentVariantLabels = []; }
        const plot = window.ManhattanPlot;
        const region = plot?.getSelectedRegion?.();
        const selection = plot?.getVariantIds?.() || [];
        if (role === 'user' || !currentSelectionSnapshot) {
            const regions = plot?.getMultiSelectRegions?.() || (region ? [region] : []);
            currentSelectionSnapshot = {
                selected_variants: [...selection],
                selected_region: region ? { chromosome: region.chromosome, start: region.start, end: region.end } : null,
                selected_regions: regions.map(item => ({
                    chromosome: item.chromosome, start: item.start, end: item.end,
                    selected_variants: Array.isArray(item.snps) ? item.snps.map(snp =>
                        `${String(snp.chromosome).replace(/^chr/i, '')}_${snp.position}_${snp.ref}_${snp.alt}`) : [...selection]
                }))
            };
        }
        const metadata = role === 'assistant' ? {
            run_id: currentRunId, progress: Array.from(currentProgress.values()),
            ...(currentAnalysisStatus ? { analysis_status: currentAnalysisStatus, computed_facts: currentComputedFacts } : {}),
            ...(currentVariantNormalizations.length ? { variant_normalizations: currentVariantNormalizations.map(item => ({ ...item })) } : {}),
            ...(currentRankingEvidence.length ? { ranking_evidence: currentRankingEvidence.map(item => ({ ...item })) } : {}),
            ...(currentGeneEvidenceCards?.length ? { gene_evidence_cards: currentGeneEvidenceCards.map(item => ({ ...item })) } : {}),
            ...(currentVariantLabels.length ? { variant_labels: currentVariantLabels.map(item => ({ ...item })) } : {}),
            report_quality: String(content).match(/Report quality:\s*\**(PASS_WITH_WARNINGS|PASS|STOPPED)/)?.[1] || null
        } : {};
        turns.push({ role, content: String(content), timestamp: new Date().toISOString(), ...metadata,
            ...currentSelectionSnapshot });
        saveConversation();
    };
    window.newWorkspaceChat = async function () {
        window.cancelActiveChatRequest?.();
        await preserveConversation();
        conversationEpoch += 1; conversationId = null; turns = [];
        syncConversationLocation();
        currentAnalysisStatus = null; currentComputedFacts = null; currentVariantNormalizations = [];
        currentRunId = null; currentRankingEvidence = []; currentGeneEvidenceCards = null; currentVariantLabels = [];
        currentSelectionSnapshot = null;
        window.clearChat?.();
    };

    window.saveWorkspaceAnalysisConversation = function (historyId, locusNumber, messages) {
        if (!idPattern.test(historyId || '') || !Number.isInteger(locusNumber) || locusNumber < 1) {
            return Promise.reject(new Error('The analysis must be saved before its discussion can be saved.'));
        }
        const turnsSnapshot = messages.map(message => ({ ...message }));
        analysisSaveChain = analysisSaveChain.catch(() => {}).then(async () => {
            await api('/api/history/analyses/' + historyId + '/discussion', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ locus_no: locusNumber, messages: turnsSnapshot })
            });
        });
        return analysisSaveChain;
    };

    async function restoreSnapshot(kind, id) {
        if (!idPattern.test(id)) throw new Error('Invalid history identifier');
        window.cancelActiveChatRequest?.();
        if (typeof cancelAnalysisChatRequest === 'function') cancelAnalysisChatRequest();
        await preserveConversation();
        await analysisSaveChain;
        if (typeof abandonAnalysisRun === 'function') abandonAnalysisRun();
        await api(`/api/history/${kind}/${id}/restore`, { method: 'POST' });
        location.href = '/app' + (kind === 'datasets' ? '' : `?history_kind=${kind}&history_id=${id}`);
    }

    function historyEntry(item, kind, open) {
        const button = element('button', 'history-entry');
        button.append(element('strong', '', item.title || item.source_phenotype || 'Untitled'));
        const unit = kind === 'datasets' ? 'SNPs' : kind === 'chats' ? 'messages' : 'loci';
        const conversations = kind === 'datasets' && Number.isFinite(item.chat_count)
            ? ` · ${item.chat_count} conversation${item.chat_count === 1 ? '' : 's'}` : '';
        button.append(element('small', '', `${new Date(item.updated_at || item.created_at).toLocaleString()}`
            + ` · ${item.source_phenotype || ''} · ${item.item_count || 0} ${unit}${conversations}`));
        button.onclick = () => open(button);
        return button;
    }

    window.openWorkspaceHistory = async function (kind = 'datasets') {
        if (!['datasets', 'chats', 'analyses'].includes(kind)) return;
        const modal = dialog({ datasets: 'GWAS data history', chats: 'Chat history', analyses: 'Auto Analysis history' }[kind]);
        const list = element('div', 'history-list');
        const status = element('p', 'dialog-status', 'Loading history…');
        const more = element('button', 'dialog-submit', 'Load more');
        more.hidden = true;
        modal.append(status, list, more);
        // A dataset is the unit a user recognises: its upload is one history
        // entry and every saved conversation hangs inside that entry.
        const scoped = kind === 'chats' && datasetId ? `&dataset_id=${datasetId}` : '';
        let offset = 0;
        const listed = [];
        const summaryText = () => !listed.length ? 'No saved records yet.'
            : kind === 'datasets' ? 'Newest upload first · conversations are stored inside their data'
            : scoped ? 'Conversations saved with the loaded GWAS data' : 'Newest first · stored on this computer';
        async function openEntry(entryKind, id, button) {
            button.disabled = true; status.textContent = 'Restoring…'; status.classList.remove('dialog-error');
            try { await restoreSnapshot(entryKind, id); }
            catch (error) { status.textContent = error.message; status.classList.add('dialog-error'); button.disabled = false; }
        }
        async function load() {
            more.disabled = true;
            try {
                await saveChain;
                const result = await api(`/api/history/${kind}?limit=30&offset=${offset}${scoped}`);
                listed.push(...result.items);
                status.textContent = summaryText();
                for (const item of result.items) {
                    if (kind !== 'datasets') {
                        list.append(historyEntry(item, kind, button => openEntry(kind, item.id, button)));
                        continue;
                    }
                    const group = element('div', 'history-entry-group');
                    const chats = element('div', 'history-chat-list'); chats.hidden = true;
                    const toggle = element('button', 'history-entry-toggle');
                    // A listing served before this change carries no count; keep
                    // the entry usable instead of claiming it has no conversations.
                    const counted = Number.isFinite(item.chat_count);
                    const label = hidden => !counted ? `${hidden ? 'Show' : 'Hide'} conversations`
                        : item.chat_count ? `${hidden ? 'Show' : 'Hide'} ${item.chat_count} conversation${item.chat_count === 1 ? '' : 's'}`
                        : 'No conversations yet';
                    toggle.textContent = label(true);
                    toggle.disabled = counted && !item.chat_count;
                    let loaded = false, loading = false, chatOffset = 0;
                    const seenChats = new Set();
                    const nextChats = element('button', 'history-entry-toggle', 'Load more conversations');
                    async function loadChats() {
                        if (loading) return;
                        loading = true; nextChats.disabled = true;
                        status.textContent = 'Loading conversations…'; status.classList.remove('dialog-error');
                        try {
                            const found = await api(`/api/history/chats?limit=30&offset=${chatOffset}&dataset_id=${item.id}`);
                            // A service older than this change ignores the filter,
                            // so the entry only lists conversations that name it.
                            const owned = found.items.filter(chat => chat.dataset_id === undefined || chat.dataset_id === item.id);
                            nextChats.remove();
                            for (const chat of owned) {
                                if (seenChats.has(chat.id)) continue;
                                seenChats.add(chat.id);
                                const row = historyEntry(chat, 'chats', button => openEntry('chats', chat.id, button));
                                row.classList.add('history-chat-entry');
                                chats.append(row);
                            }
                            chatOffset += found.items.length;
                            if (found.has_more && found.items.length) chats.append(nextChats);
                            else if (!seenChats.size) chats.append(element('p', 'history-chat-empty', 'No conversations saved for this data.'));
                            loaded = true;
                            toggle.textContent = label(chats.hidden);
                            status.textContent = summaryText();
                        } catch (error) {
                            status.textContent = error.message; status.classList.add('dialog-error');
                            if (!nextChats.parentNode) chats.append(nextChats);
                        }
                        finally { loading = false; nextChats.disabled = false; }
                    }
                    nextChats.onclick = loadChats;
                    toggle.onclick = async () => {
                        chats.hidden = !chats.hidden;
                        toggle.textContent = label(chats.hidden);
                        if (!chats.hidden && !loaded) await loadChats();
                    };
                    group.append(historyEntry(item, 'datasets', button => openEntry('datasets', item.id, button)), toggle, chats);
                    list.append(group);
                }
                offset += result.items.length; more.hidden = !result.has_more;
            } catch (error) { status.textContent = error.message; status.classList.add('dialog-error'); }
            finally { more.disabled = false; }
        }
        more.onclick = load;
        await load();
    };

    window.addEventListener('gwasmind-dataset-ready', async event => {
        const restoreDatasetGeneration = ++datasetGeneration;
        const plotMetadata = event.detail?.metadata || {};
        datasetId = plotMetadata.history_dataset_id || null;
        sourcePhenotype = plotMetadata.source_phenotype || '';
        const params = new URLSearchParams(location.search);
        const kind = params.get('history_kind'), id = params.get('history_id');
        if (!['chats', 'analyses'].includes(kind) || !idPattern.test(id || '')) return;
        const restoreEpoch = conversationEpoch;
        const existingTurnCount = turns.length;
        try {
            const snapshot = await api(`/api/history/${kind}/${id}${kind === 'chats' ? '?include_ranking_review=1' : ''}`);
            if (restoreDatasetGeneration !== datasetGeneration
                || restoreEpoch !== conversationEpoch || existingTurnCount !== turns.length) return;
            if (kind === 'analyses' && !snapshot.dataset_id) {
                const restoredForThisAnalysis = plotMetadata.history_analysis_id === id
                    && ['dataset', 'analysis_selection'].includes(plotMetadata.history_plot_scope);
                const plotDataAvailable = event.detail?.plotDataAvailable ?? Boolean(window.ManhattanPlot?.rawData?.length);
                if (!restoredForThisAnalysis && (datasetId || plotMetadata.history_analysis_id || plotDataAvailable)) {
                    throw new Error('This analysis history has no dataset binding. Open it from History to restore its saved plotting scope.');
                }
            } else if (Object.hasOwn(snapshot, 'dataset_id') && (snapshot.dataset_id || null) !== datasetId) {
                if (!datasetId && snapshot.dataset_id) throw new Error('No GWAS dataset is loaded. Open this entry from History to restore its saved data.');
                throw new Error('This history entry belongs to different GWAS data. Open it from History to restore its original dataset.');
            }
            if (kind === 'chats') {
                restoring = true;
                conversationEpoch += 1; conversationId = id;
                // A report-only conversation may have no plotting dataset. Keep
                // its saved source phenotype, never the unrelated active plot.
                sourcePhenotype = snapshot.source_phenotype || sourcePhenotype;
                turns = snapshot.messages || [];
                window.restoreWorkspaceChat?.(turns);
                const last = [...turns].reverse().find(turn => Array.isArray(turn.selected_variants));
                if (last) {
                    const plot = window.ManhattanPlot;
                    const rawData = plot?.rawData || [];
                    const byId = new Map();
                    rawData.forEach(snp => {
                        byId.set(`${snp.chromosome}_${snp.position}_${snp.ref}_${snp.alt}`, snp);
                        if (!byId.has(snp.snp_id)) byId.set(snp.snp_id, snp);
                    });
                    const savedRegions = Array.isArray(last.selected_regions) && last.selected_regions.length
                        ? last.selected_regions
                        : [{ ...(last.selected_region || {}), selected_variants: last.selected_variants }];
                    const regions = savedRegions.map(saved => {
                        const snps = [...new Set(saved.selected_variants || [])].map(id => byId.get(id)).filter(Boolean);
                        if (!snps.length) return null;
                        return { ...saved, snps,
                            chromosome: [...new Set(snps.map(snp => snp.chromosome))].join(','),
                            start: Number.isFinite(saved.start) ? saved.start : snps.reduce((min, snp) => Math.min(min, snp.position), Infinity),
                            end: Number.isFinite(saved.end) ? saved.end : snps.reduce((max, snp) => Math.max(max, snp.position), -Infinity)
                        };
                    }).filter(Boolean);
                    if (regions.length) {
                        if (plot.restorePanelSelection) {
                            await plot.restorePanelSelection('chat', regions);
                        } else {
                            plot.selectedRegion = regions.at(-1); plot.multiSelectRegions = regions;
                            plot.syncSelectedVariants(); plot.remapSelectedIndices();
                            plot.showSelectionInfo();
                            // Keep older cached plot scripts usable during a reload.
                            const chromosomes = new Set(regions.flatMap(region => region.snps.map(snp => String(snp.chromosome))));
                            const chromosome = chromosomes.size === 1 ? [...chromosomes][0] : null;
                            if (chromosome && plot.focusedChromosome !== chromosome) {
                                await plot.focusChromosome(chromosome);
                            } else if (!chromosome && plot.focusedChromosome) {
                                await plot.zoomOut();
                            } else if (!plot.webglSupported && plot.selectedVariantKeys.size > (plot.selectedPointIndices?.length || 0)) {
                                plot.invalidateRenderCache();
                                await plot.render();
                            } else await plot.updateHighlighting();
                        }
                    }
                }
            } else await window.restoreWorkspaceAnalysis?.(snapshot);
        } catch (error) { window.addMessage?.('system', 'Unable to restore history: ' + error.message); }
        finally { restoring = false; }
    });

    installResizer();
}());

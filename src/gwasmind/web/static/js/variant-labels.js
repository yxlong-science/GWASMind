/* Presentation-only rsID labels. Fact review uses the immutable source identities. */
(function () {
    'use strict';
    const identity = /^(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)_[1-9][0-9]*_[ACGT]+_[ACGT]+$/;
    const rsid = /^rs[1-9][0-9]*$/;
    const token = /(?<![A-Za-z0-9_/:])(?:chr)?(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)[_:][1-9][0-9]*[_:][ACGT]+[_:][ACGT]+(?![A-Za-z0-9_])/gi;
    const displayToken = new RegExp(token.source + '|(?<![A-Za-z0-9_/:])rs[1-9][0-9]*(?![A-Za-z0-9_])', 'gi');
    const canonical = value => String(value).replace(/^chr/i, '').replaceAll(':', '_').toUpperCase();

    function cleanLabels(value) {
        if (!Array.isArray(value)) return [];
        const rows = new Map(), conflicts = new Set(), owners = new Map();
        for (const item of value) {
            if (!item || item.genome_build !== 'GRCh38' || !rsid.test(item.rsid)
                    || !identity.test(item.variant_id) || !identity.test(item.canonical_variant_id)) continue;
            const original = item.variant_id.split('_'), resolved = item.canonical_variant_id.split('_');
            if (original[0] !== resolved[0] || original[1] !== resolved[1]
                    || !(original[2] === resolved[2] && original[3] === resolved[3]
                        || original[2] === resolved[3] && original[3] === resolved[2])) continue;
            const row = { variant_id: item.variant_id, canonical_variant_id: item.canonical_variant_id,
                genome_build: 'GRCh38', rsid: item.rsid };
            const previous = rows.get(row.variant_id);
            if (previous && (previous.rsid !== row.rsid || previous.canonical_variant_id !== row.canonical_variant_id))
                conflicts.add(row.variant_id);
            rows.set(row.variant_id, row);
            if (!owners.has(row.rsid)) owners.set(row.rsid, new Set());
            owners.get(row.rsid).add(row.canonical_variant_id);
        }
        return [...rows.values()].filter(row => !conflicts.has(row.variant_id) && owners.get(row.rsid).size === 1);
    }

    function sourceLabel(row) {
        return `Source: ${row.genome_build} ${row.variant_id}`
            + (row.variant_id !== row.canonical_variant_id ? `; canonical ${row.canonical_variant_id}` : '');
    }

    function decorate(container, input) {
        if (!container) return;
        const labels = cleanLabels(input || container.variantLabels);
        container.variantLabels = labels;
        if (!document.createTreeWalker) return;
        const lookup = new Map(labels.map(row => [row.variant_id, row]));
        const walker = document.createTreeWalker(container, 4), nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        for (const node of nodes) {
            if (node.parentElement?.closest('a,pre,script,style,textarea,details:not(.report-section),.report-section[data-part="References"],.qtl-source-body,.evidence-preview-source')) continue;
            const text = node.nodeValue || '';
            const urls = [...text.matchAll(/https?:\/\/[^\s<]+/gi)];
            const matches = [...text.matchAll(displayToken)].filter(match =>
                (lookup.has(canonical(match[0])) || rsid.test(match[0].toLowerCase()))
                && !urls.some(url => match.index >= url.index && match.index < url.index + url[0].length));
            if (!matches.length) continue;
            const fragment = document.createDocumentFragment(); let position = 0;
            for (const match of matches) {
                fragment.appendChild(document.createTextNode(text.slice(position, match.index)));
                const row = lookup.get(canonical(match[0]));
                const label = row?.rsid || match[0].toLowerCase();
                const source = row ? sourceLabel(row) : `dbSNP: ${label}`;
                const link = document.createElement('a');
                link.className = 'variant-reference'; link.href = 'https://www.ncbi.nlm.nih.gov/snp/' + label;
                link.textContent = label; link.title = source;
                link.setAttribute('aria-label', label + '. ' + source);
                link.target = '_blank'; link.rel = 'noopener noreferrer';
                fragment.appendChild(link); position = match.index + match[0].length;
            }
            fragment.appendChild(document.createTextNode(text.slice(position)));
            node.replaceWith(fragment);
        }
    }

    function markdown(text, input) {
        const labels = cleanLabels(input), lookup = new Map(labels.map(row => [row.variant_id, row]));
        let fenced = false, references = false;
        return String(text || '').split('\n').map(line => {
            if (/^\s*```/.test(line)) { fenced = !fenced; return line; }
            if (!fenced && /^\s*#{1,6}\s+/.test(line)) references = /^\s*#{1,6}\s+References\s*$/i.test(line);
            if (fenced || references || /^\s*(?:[-*]\s*)?(?:Source|Canonical identity|Input locus identity)\s*:/i.test(line)) return line;
            // Preserve existing links, destinations, HTML and their source titles.
            return line.split(/(\[[^\]\n]*\]\((?:[^()"'\\\n]|\\.|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|\([^()\n]*\))*\)|https?:\/\/[^\s<]+|<[^>\n]*>)/gi).map((part, index) => {
                if (index % 2) return part;
                const replace = value => {
                    const row = lookup.get(canonical(value));
                    if (row) return `[${row.rsid}](https://www.ncbi.nlm.nih.gov/snp/${row.rsid} "${sourceLabel(row)}")`;
                    const label = value.toLowerCase();
                    return rsid.test(label) ? `[${label}](https://www.ncbi.nlm.nih.gov/snp/${label})` : value;
                };
                return part.split(/(`[^`\n]+`)/g).map((section, code) => code % 2
                    ? lookup.has(canonical(section.slice(1, -1))) || rsid.test(section.slice(1, -1).toLowerCase())
                        ? replace(section.slice(1, -1)) : section
                    : section.replace(displayToken, replace)).join('');
            }).join('');
        }).join('\n');
    }

    window.GWASMindVariants = Object.freeze({ cleanLabels, decorate, markdown, sourceLabel });
})();

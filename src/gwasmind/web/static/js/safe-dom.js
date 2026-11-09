(function () {
    "use strict";

    const allowedTags = new Set([
        "A", "B", "BLOCKQUOTE", "BR", "CODE", "DEL", "EM", "H1", "H2",
        "H3", "H4", "H5", "H6", "HR", "I", "IMG", "LI", "OL", "P",
        "PRE", "SPAN", "STRONG", "TABLE", "TBODY", "TD", "TH", "THEAD",
        "TR", "UL"
    ]);

    function safeUrl(value, image) {
        const text = String(value || "").trim();
        if (!text || /[\u0000-\u001F\u007F]/.test(text)) return null;
        if (image) return /^\/static\//i.test(text) ? text : null;
        if (text.startsWith("#") || /^\/(?!\/)/.test(text) || /^https?:\/\//i.test(text)) return text;
        return null;
    }

    function copyChildren(source, destination) {
        source.childNodes.forEach((child) => destination.appendChild(cleanNode(child)));
    }

    function cleanNode(node) {
        if (node.nodeType === Node.TEXT_NODE) return document.createTextNode(node.nodeValue || "");
        if (node.nodeType !== Node.ELEMENT_NODE) return document.createDocumentFragment();
        if (!allowedTags.has(node.tagName)) {
            const fragment = document.createDocumentFragment();
            copyChildren(node, fragment);
            return fragment;
        }

        const clean = document.createElement(node.tagName.toLowerCase());
        if (node.tagName === "A") {
            const href = safeUrl(node.getAttribute("href"), false);
            if (href) {
                clean.setAttribute("href", href);
                if (/^https?:\/\//i.test(href)) {
                    clean.setAttribute("target", "_blank");
                    clean.setAttribute("rel", "noopener noreferrer");
                }
            }
            if (node.getAttribute("title")) clean.setAttribute("title", node.getAttribute("title"));
        } else if (node.tagName === "IMG") {
            const src = safeUrl(node.getAttribute("src"), true);
            if (!src) return document.createDocumentFragment();
            clean.setAttribute("src", src);
            clean.setAttribute("alt", node.getAttribute("alt") || "");
            if (node.getAttribute("title")) clean.setAttribute("title", node.getAttribute("title"));
        } else if (node.tagName === "CODE") {
            const className = node.getAttribute("class") || "";
            if (/^language-[a-z0-9_-]+$/i.test(className)) clean.className = className;
        } else if (/^H[1-6]$/.test(node.tagName)) {
            const identifier = node.getAttribute("id") || "";
            if (/^[a-z0-9_-]{1,120}$/i.test(identifier)) clean.id = identifier;
        }
        copyChildren(node, clean);
        return clean;
    }

    function sanitizeHtml(value) {
        const template = document.createElement("template");
        template.innerHTML = String(value || "");
        const output = document.createElement("div");
        copyChildren(template.content, output);
        return output.innerHTML;
    }

    function escapeText(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }

    function renderMarkdown(source, options) {
        const parser = window.marked;
        if (!parser || typeof parser.parse !== "function") {
            throw new Error("Markdown renderer is unavailable");
        }
        const html = parser.parse(String(source ?? ""), options);
        if (typeof html !== "string") {
            throw new Error("Markdown renderer must return HTML synchronously");
        }
        return emphasizeReportGenes(sanitizeHtml(html));
    }

    function emphasizeReportGenes(html) {
        // Saved reports used italics alone. Only promote symbols declared by
        // report gene headings; ordinary emphasis, IDs and references stay intact.
        const holder = document.createElement("div");
        holder.innerHTML = html;
        const symbols = new Set();
        for (const heading of holder.querySelectorAll("h3,h4")) {
            const match = heading.textContent.match(/^\s*(?:\d+[.)]\s*)?([A-Za-z][A-Za-z0-9-]*)\s+\(?ENSG\d+\b/);
            if (match && !match[1].startsWith("ENSG")) symbols.add(match[1]);
        }
        if (!symbols.size) return html;
        for (const block of holder.children) {
            if (/^(?:P|H[1-6])$/.test(block.tagName) && block.textContent.trim() === "References") break;
            for (const emphasis of block.querySelectorAll("em,i")) {
                if (!symbols.has(emphasis.textContent) || emphasis.closest("code,pre,strong,b")
                        || emphasis.querySelector("strong,b")) continue;
                const strong = document.createElement("strong");
                while (emphasis.firstChild) strong.appendChild(emphasis.firstChild);
                emphasis.appendChild(strong);
            }
        }
        return holder.innerHTML;
    }

    // Marked v15 exposes `parse` as a getter-only property. Do not monkey-patch
    // the vendor namespace; keep parsing and DOM sanitization explicit instead.
    window.GWASMindSafe = { sanitizeHtml, escapeText, renderMarkdown };
}());

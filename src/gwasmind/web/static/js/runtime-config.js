/* Runtime values are supplied by the Python TOML loader, never duplicated here. */
(function installRuntimeConfig(root) {
    'use strict';

    function createRuntimeConfig(payload) {
        if (!payload || typeof payload !== 'object' || !payload.values || Array.isArray(payload.values)) {
            throw new Error('The server did not provide valid runtime parameters. Reload the page.');
        }
        const values = Object.freeze(Object.fromEntries(Object.entries(payload.values).map(([key, value]) =>
            [key, Array.isArray(value) ? Object.freeze([...value]) : value])));
        return Object.freeze({
            effectiveHash: String(payload.effective_sha256 || ''),
            get(key) {
                if (!Object.prototype.hasOwnProperty.call(values, key)) {
                    throw new Error(`Required runtime parameter is missing: ${key}`);
                }
                return values[key];
            }
        });
    }

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { createRuntimeConfig };
    }
    if (root?.document) {
        const node = root.document.getElementById('gwasmind-runtime-parameters');
        if (!node?.dataset?.parameters) throw new Error('Runtime configuration is unavailable. Reload the page.');
        root.GWASMindConfig = createRuntimeConfig(JSON.parse(node.dataset.parameters));
    }
})(typeof globalThis !== 'undefined' ? globalThis : this);

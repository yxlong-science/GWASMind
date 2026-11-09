(function loadRuntimeUiPrompts() {
    "use strict";

    const node = document.getElementById("gwasmind-ui-prompts");
    if (!node || !node.dataset.prompts) {
        throw new Error("The server did not provide the runtime UI prompt catalog.");
    }
    const prompts = JSON.parse(node.dataset.prompts);
    window.GWASMIND_UI_PROMPTS = Object.freeze(prompts);
})();

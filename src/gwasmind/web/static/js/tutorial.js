(function () {
    "use strict";

    const target = document.getElementById("tutorial-markdown");
    if (!target) return;

    fetch("/static/docs/tutorial.md")
        .then((response) => {
            if (!response.ok) throw new Error("Tutorial could not be loaded");
            return response.text();
        })
        .then((source) => {
            const safeDom = window.GWASMindSafe;
            if (!safeDom || typeof safeDom.renderMarkdown !== "function") {
                throw new Error("Tutorial renderer is unavailable");
            }
            target.innerHTML = safeDom.renderMarkdown(source);
            target.querySelectorAll("h1, h2, h3, h4, h5, h6").forEach((heading) => {
                if (heading.id) return;
                heading.id = (heading.textContent || "section")
                    .trim()
                    .toLowerCase()
                    .replace(/[^a-z0-9]+/g, "-")
                    .replace(/^-|-$/g, "")
                    .slice(0, 120);
            });
        })
        .catch((error) => {
            target.textContent = error.message;
        });
}());

(function () {
    "use strict";

    let errorRemovalTimer = null;

    function reportActionError(error, action) {
        const detail = error instanceof Error ? error.message : String(error || "Unknown error");
        const message = action ? `${action}: ${detail}` : detail;
        console.error("[GWASMind UI]", message, error);
        let alertBox = document.getElementById("gwasmind-action-error");
        if (!alertBox) {
            alertBox = document.createElement("div");
            alertBox.id = "gwasmind-action-error";
            alertBox.setAttribute("role", "alert");
            alertBox.setAttribute("aria-live", "assertive");
            alertBox.style.cssText = "position:fixed;right:24px;bottom:24px;z-index:20000;max-width:440px;padding:12px 16px;border-radius:7px;background:#b42318;color:#fff;box-shadow:0 4px 14px rgba(0,0,0,.22);font-size:14px;line-height:1.4;";
            document.body.appendChild(alertBox);
        }
        alertBox.textContent = message;
        if (errorRemovalTimer !== null) window.clearTimeout(errorRemovalTimer);
        errorRemovalTimer = window.setTimeout(() => {
            alertBox.remove();
            errorRemovalTimer = null;
        }, 7000);
    }

    function handleActionResult(result, action) {
        if (result && typeof result.then === "function") {
            result.catch((error) => reportActionError(error, action));
        }
        return result;
    }

    function callGlobal(name, ...args) {
        const fn = window[name];
        if (typeof fn !== "function") {
            reportActionError(new Error(`Required handler window.${name} is unavailable`), name);
            return undefined;
        }
        try {
            return handleActionResult(fn(...args), name);
        } catch (error) {
            reportActionError(error, name);
            return undefined;
        }
    }

    function withPlot(callback, action = "Plot action") {
        if (typeof ManhattanPlot === "undefined") {
            reportActionError(new Error("The Manhattan plot is not available"), action);
            return undefined;
        }
        try {
            return handleActionResult(callback(ManhattanPlot), action);
        } catch (error) {
            reportActionError(error, action);
            return undefined;
        }
    }

    function clickInput(id, action) {
        const input = document.getElementById(id);
        if (!input) {
            reportActionError(new Error(`Required input #${id} is unavailable`), action);
            return;
        }
        input.click();
    }

    document.addEventListener("click", (event) => {
        const target = event.target.closest("[data-action]");
        if (!target) return;
        const action = target.dataset.action;

        switch (action) {
            case "open-settings": callGlobal("openSettings"); break;
            case "test-connection": callGlobal("testConnection"); break;
            case "save-settings": callGlobal("saveSettings"); break;
            case "close-settings": callGlobal("closeSettings"); break;
            case "load-demo": callGlobal("loadDemo", target.dataset.demo); break;
            case "open-tutorial": callGlobal("openTutorialModal"); break;
            case "close-tutorial": callGlobal("closeTutorialModal"); break;
            case "open-contact": callGlobal("openContactModal"); break;
            case "close-contact": callGlobal("closeContactModal"); break;
            case "close-detail": callGlobal("closeDetailModal"); break;
            case "clear-data": callGlobal("clearData"); break;
            case "toggle-panel": callGlobal("toggleChat"); break;
            case "marker-size":
                withPlot((plot) => plot.changeMarkerSize(Number(target.dataset.delta)), action);
                break;
            case "reset-axes": withPlot((plot) => plot.resetAxes(), action); break;
            case "zoom-out": withPlot((plot) => plot.zoomOut(), action); break;
            case "toggle-export": callGlobal("openPlotExport"); break;
            case "plot-tool": callGlobal("runPlotTool", target.dataset.tool); break;
            case "open-history": callGlobal("openWorkspaceHistory", target.dataset.kind); break;
            case "new-chat": callGlobal("newWorkspaceChat"); break;
            case "export-plot":
                withPlot((plot) => plot.exportAs(target.dataset.format), action);
                break;
            case "add-annotation": withPlot((plot) => plot.addAnnotation(), action); break;
            case "clear-and-reset":
                withPlot((plot) => {
                    plot.clearSelection();
                    plot.resetAxes();
                }, action);
                break;
            case "clear-plot-selection": withPlot((plot) => plot.clearSelection(), action); break;
            case "highlight-card-left":
                callGlobal("highlightCardFromLeft", Number(target.dataset.regionIndex));
                break;
            case "highlight-from-card":
                callGlobal("highlightFromCard", Number(target.dataset.regionIndex));
                break;
            case "switch-tab": callGlobal("switchTab", target.dataset.tab); break;
            case "quick-prompt": callGlobal("sendQuickPrompt"); break;
            case "send-message": callGlobal("sendMessage"); break;
            case "browse-loci":
                clickInput("analysis-file-input", action);
                break;
            case "load-analysis-demo": callGlobal("loadDemoData"); break;
            case "start-quick-analysis": callGlobal("startQuickAnalysis"); break;
            case "reset-analysis": callGlobal("resetAnalysis"); break;
            case "analysis-suggestion":
                callGlobal(
                    "sendAnalysisSuggestion",
                    window.GWASMIND_UI_PROMPTS[target.dataset.promptKey]
                );
                break;
            case "send-analysis-message": callGlobal("sendAnalysisMessage"); break;
            case "choose-analysis-file": clickInput("file-input", action); break;
            case "start-analysis": callGlobal("startAnalysis"); break;
            default: reportActionError(new Error(`Unsupported action: ${action || "(empty)"}`), "UI action"); break;
        }
    });

    document.addEventListener("input", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLInputElement)) return;
        if (target.id === "threshold-range") {
            const numberInput = document.getElementById("threshold-input");
            if (numberInput) numberInput.value = target.value;
            withPlot((plot) => plot.setThreshold(Number(target.value)));
        }
    });

    document.addEventListener("change", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLInputElement)) return;
        if (target.id === "threshold-input") {
            withPlot((plot) => plot.setThreshold(Number(target.value)));
        } else if (target.id === "file-input" && target.files?.length
                && document.getElementById("upload-section")) {
            callGlobal("handleFileSelect", event);
        }
    });

    document.addEventListener("keydown", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) return;
        if (event.key === "Enter" && target.id === "chat-input") {
            event.preventDefault();
            callGlobal("sendMessage");
        } else if (event.key === "Enter" && target.id === "analysis-chat-input") {
            event.preventDefault();
            callGlobal("sendAnalysisMessage");
        } else if ((event.key === "Enter" || event.key === " ") && target.dataset.action) {
            event.preventDefault();
            target.click();
        }
    });
}());

// Companion runtime for the Shibuya copy-page-button.html override. Two
// non-obvious behaviors:
//
// * file:// guard: Shibuya's stock copy handler fails silently when fetch()
//   is blocked across file:// origins; we intercept the click to surface
//   the command the user needs to run to serve over HTTP.
// * label sync: mirror Shibuya's copy-button icon state into the text label
//   and hold the "Copied" state longer than the theme's 500ms default.

(function () {
    function attachFileProtocolGuard(wrapper) {
        const message = [
            "Copy doesn't work when opening the docs as a file:// URL.",
            "",
            "The browser blocks fetch() across file:// origins, so the page",
            "can't be read into the clipboard.",
            "",
            "Fix: serve the docs over HTTP. From the build output directory, run:",
            "",
            "    python -m http.server",
            "",
            "then open the printed http://localhost:8000 URL, where Copy works.",
        ].join("\n");

        wrapper.querySelectorAll(".js-copy").forEach((button) => {
            button.addEventListener(
                "click",
                (event) => {
                    event.stopImmediatePropagation();
                    event.preventDefault();
                    alert(message);
                },
                true,
            );
        });
    }

    // Shibuya reverts the check icon to copy after 500ms; we override that
    // to keep the success state visible for the full 2 seconds.
    const COPIED_HOLD_MS = 2000;

    function attachLabelSync(button) {
        const icon = button.querySelector("i.i-lucide");
        const label = button.querySelector("span:not(.iconify-icon)");
        if (!icon || !label) return;

        const restingLabel = label.textContent;
        // Non-null holdTimer means "we own the icon+label; undo Shibuya's
        // revert until the timer fires." Doubles as the "is this 'check'
        // attribute change our own re-set?" check below.
        let holdTimer = null;

        const observer = new MutationObserver(() => {
            const state = icon.dataset.icon;

            if (state === "loader") {
                clearTimeout(holdTimer);
                holdTimer = null;
                label.textContent = "Copying…";
            } else if (state === "check") {
                if (holdTimer !== null) return;
                label.textContent = "Copied";
                holdTimer = setTimeout(() => {
                    holdTimer = null;
                    // Resetting the icon triggers this observer again; the
                    // "copy" branch with holdTimer===null restores the label.
                    icon.dataset.icon = "copy";
                }, COPIED_HOLD_MS);
            } else if (state === "copy") {
                if (holdTimer !== null) {
                    icon.dataset.icon = "check";
                } else {
                    label.textContent = restingLabel;
                }
            }
        });
        observer.observe(icon, { attributes: true, attributeFilter: ["data-icon"] });
    }

    function init() {
        const wrapper = document.querySelector(".copy-page-wrapper");
        if (!wrapper) return;
        if (window.location.protocol === "file:") {
            attachFileProtocolGuard(wrapper);
        } else {
            wrapper.querySelectorAll(".js-copy").forEach(attachLabelSync);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();

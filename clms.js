(() => {
    const entries = Array.from(document.querySelectorAll(".entry"));
    const isEditable = (target) => target instanceof Element
        && target.closest(
            "input, textarea, select, button, [contenteditable]:not([contenteditable=\"false\"])",
        );

    function scrollToEntry(direction) {
        if (!entries.length) {
            return;
        }

        if (direction > 0) {
            const nextEntry = entries.find((entry) => entry.getBoundingClientRect().top > 1);
            if (nextEntry) {
                nextEntry.scrollIntoView({ block: "start" });
            } else {
                window.scrollTo({ top: document.documentElement.scrollHeight });
            }
            return;
        }

        for (let index = entries.length - 1; index >= 0; index -= 1) {
            if (entries[index].getBoundingClientRect().bottom < -1) {
                entries[index].scrollIntoView({ block: "start" });
                return;
            }
        }

        window.scrollTo({ top: 0 });
    }

    document.addEventListener("keydown", (event) => {
        if (event.defaultPrevented || isEditable(event.target)) {
            return;
        }

        if (event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey) {
            if (event.code === "KeyJ") {
                event.preventDefault();
                scrollToEntry(1);
            } else if (event.code === "KeyK") {
                event.preventDefault();
                scrollToEntry(-1);
            }
            return;
        }

        if (event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey) {
            const relation = event.code === "Comma"
                ? "prev"
                : event.code === "Period" ? "next" : null;
            if (!relation) {
                return;
            }

            const pageLink = document.querySelector(`.pagination a[rel="${relation}"]`);
            if (pageLink) {
                event.preventDefault();
                pageLink.click();
            }
        }
    });
})();

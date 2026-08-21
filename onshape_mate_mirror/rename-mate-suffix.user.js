// ==UserScript==
// @name         Onshape mate suffix renamer
// @namespace    robot-training
// @version      1.0.0
// @description  Rename every mate suffix in the active Onshape Assembly through the normal UI.
// @match        https://cad.onshape.com/documents/*
// @grant        none
// ==/UserScript==

(() => {
    "use strict";

    const BUTTON_ID = "robot-training-mate-renamer";
    const LABEL_SELECTOR = ".os-list-item-name";
    const ROW_SELECTOR = ".os-list-item";
    const MENU_ITEM_SELECTOR = ".context-menu-item";
    const WAIT_INTERVAL_MS = 50;
    const WAIT_TIMEOUT_MS = 3000;

    const textOf = (element) => (element?.textContent ?? "").trim();

    function escapeRegExp(value) {
        return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    function renamed(name, fromSuffix, toSuffix) {
        const pattern = new RegExp(`(^|[\\s_-])${escapeRegExp(fromSuffix)}$`, "i");
        return pattern.test(name) ? name.replace(pattern, `$1${toSuffix}`) : null;
    }

    function visible(element) {
        if (!element) return false;
        const box = element.getBoundingClientRect();
        const style = getComputedStyle(element);
        return box.width > 0 && box.height > 0 && style.display !== "none" && style.visibility !== "hidden";
    }

    function waitFor(find, timeoutMs = WAIT_TIMEOUT_MS) {
        return new Promise((resolve, reject) => {
            const deadline = Date.now() + timeoutMs;
            const check = () => {
                const result = find();
                if (result) return resolve(result);
                if (Date.now() >= deadline) return reject(new Error("Timed out waiting for the Onshape Rename control."));
                setTimeout(check, WAIT_INTERVAL_MS);
            };
            check();
        });
    }

    function mateSection() {
        const labels = [...document.querySelectorAll(LABEL_SELECTOR)];
        const headerIndex = labels.findIndex((label) => /^Mate features(?:\s*\(|$)/i.test(textOf(label)));
        if (headerIndex < 0) {
            throw new Error("Open an Assembly tab and expand Mate features before running the renamer.");
        }

        const header = labels[headerIndex];
        const headerRow = header.closest(ROW_SELECTOR);
        const headerText = textOf(header);
        const expectedCount = Number(headerText.match(/\((\d+)\)\s*$/)?.[1] ?? NaN);
        const rows = [];
        const seen = new Set();

        for (const label of labels.slice(headerIndex + 1)) {
            const row = label.closest(ROW_SELECTOR);
            if (!row || row === headerRow || seen.has(row)) continue;

            // The next non-selectable row is the next Assembly-list section.
            if (row.classList.contains("ns-list-item-not-selectable")) break;

            seen.add(row);
            rows.push({ row, label, name: textOf(label) });
        }

        return { expectedCount, rows };
    }

    function mateRenames(fromSuffix, toSuffix, requireEveryMate = true) {
        const section = mateSection();
        const changes = section.rows
            .map((entry) => ({ ...entry, newName: renamed(entry.name, fromSuffix, toSuffix) }))
            .filter((entry) => entry.newName !== null);

        if (Number.isFinite(section.expectedCount) && section.rows.length !== section.expectedCount) {
            throw new Error(
                `Onshape reports ${section.expectedCount} mates, but only ${section.rows.length} mate rows are loaded. ` +
                "Clear the tree filter, expand Mate features, and try again."
            );
        }
        if (requireEveryMate && Number.isFinite(section.expectedCount) && changes.length !== section.expectedCount) {
            const skipped = section.rows.filter((entry) => renamed(entry.name, fromSuffix, toSuffix) === null);
            throw new Error(
                `Refusing a partial rename: ${changes.length} of ${section.expectedCount} mate names end in ${fromSuffix}.` +
                (skipped.length ? `\n\nNot matched:\n${skipped.map((entry) => `- ${entry.name}`).join("\n")}` : "")
            );
        }
        return changes;
    }

    function dispatchEnter(element) {
        for (const type of ["keydown", "keypress", "keyup"]) {
            const event = new KeyboardEvent(type, {
                key: "Enter",
                code: "Enter",
                bubbles: true,
                cancelable: true,
            });
            Object.defineProperties(event, {
                keyCode: { get: () => 13 },
                which: { get: () => 13 },
            });
            element.dispatchEvent(event);
        }
    }

    function enterName(editor, value) {
        editor.focus();
        if (editor instanceof HTMLInputElement || editor instanceof HTMLTextAreaElement) {
            const prototype = editor instanceof HTMLTextAreaElement
                ? HTMLTextAreaElement.prototype
                : HTMLInputElement.prototype;
            Object.getOwnPropertyDescriptor(prototype, "value").set.call(editor, value);
        } else {
            editor.textContent = value;
        }
        editor.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
        editor.dispatchEvent(new Event("change", { bubbles: true }));
        dispatchEnter(editor);
        editor.blur();
    }

    function findMateRow(name) {
        return mateSection().rows.find((entry) => entry.name === name)?.row ?? null;
    }

    async function renameOne(oldName, newName) {
        const row = findMateRow(oldName);
        if (!row) throw new Error(`Could not find mate \"${oldName}\".`);

        row.scrollIntoView({ block: "nearest" });
        const box = row.getBoundingClientRect();
        row.dispatchEvent(new MouseEvent("contextmenu", {
            bubbles: true,
            cancelable: true,
            view: window,
            button: 2,
            buttons: 2,
            clientX: box.left + Math.min(20, box.width / 2),
            clientY: box.top + box.height / 2,
        }));

        const renameItem = await waitFor(() => [...document.querySelectorAll(MENU_ITEM_SELECTOR)]
            .find((item) => visible(item) && textOf(item) === "Rename"));
        renameItem.click();

        const editor = await waitFor(() => {
            const currentRow = findMateRow(oldName) ?? row;
            const local = currentRow.querySelector('input[type="text"], input:not([type]), textarea, [contenteditable="true"]');
            if (visible(local)) return local;
            const active = document.activeElement;
            return visible(active) && active.matches?.('input[type="text"], input:not([type]), textarea, [contenteditable="true"]')
                ? active
                : null;
        });
        enterName(editor, newName);

        await waitFor(() => findMateRow(newName), WAIT_TIMEOUT_MS);
    }

    function inferredTarget() {
        const tabName = textOf(document.querySelector(".os-tab-bar-tab.active .os-tab-name"));
        return tabName.match(/(?:^|[-_\s])(RF|RR|LF|LR)(?:[-_\s]|$)/i)?.[1]?.toUpperCase() ?? "LF";
    }

    async function run() {
        const button = document.getElementById(BUTTON_ID);
        const fromSuffix = "RF";
        const answer = prompt("Rename all mate names ending in RF to which leg suffix?", inferredTarget());
        if (answer === null) return;

        const toSuffix = answer.trim().toUpperCase();
        if (!/^(RF|RR|LF|LR)$/.test(toSuffix) || toSuffix === fromSuffix) {
            alert("Choose RR, LF, or LR.");
            return;
        }

        let changes;
        try {
            changes = mateRenames(fromSuffix, toSuffix);
        } catch (error) {
            alert(error.message);
            return;
        }
        if (!changes.length) {
            alert("No mate names ending in RF were found.");
            return;
        }

        const preview = changes.map(({ name, newName }) => `${name}  ->  ${newName}`).join("\n");
        if (!confirm(`Rename ${changes.length} mates using Onshape's normal Rename command?\n\n${preview}`)) return;

        button.disabled = true;
        const originalText = button.textContent;
        let completed = 0;
        try {
            for (const { name, newName } of changes) {
                button.textContent = `Renaming ${completed + 1}/${changes.length}…`;
                await renameOne(name, newName);
                completed += 1;
            }

            const remaining = mateRenames(fromSuffix, toSuffix, false);
            if (remaining.length) throw new Error(`${remaining.length} RF mate names remain.`);
            alert(`Renamed all ${completed} mates from RF to ${toSuffix}. No Onshape API calls were used.`);
        } catch (error) {
            alert(
                `Stopped after ${completed} of ${changes.length} mates.\n\n${error.message}\n\n` +
                "No API calls were made. You can correct the visible mate and run the tool again."
            );
        } finally {
            button.disabled = false;
            button.textContent = originalText;
        }
    }

    function installButton() {
        if (document.getElementById(BUTTON_ID)) return;
        const button = document.createElement("button");
        button.id = BUTTON_ID;
        button.type = "button";
        button.textContent = "Rename RF mates";
        button.title = "Rename every RF mate suffix in the active Assembly tab";
        Object.assign(button.style, {
            position: "fixed",
            right: "18px",
            bottom: "58px",
            zIndex: "2147483647",
            padding: "9px 13px",
            border: "1px solid #1261a0",
            borderRadius: "4px",
            color: "white",
            background: "#1676bd",
            font: "600 13px Arial, sans-serif",
            cursor: "pointer",
            boxShadow: "0 2px 8px rgba(0, 0, 0, .3)",
        });
        button.addEventListener("click", run);
        document.body.appendChild(button);
    }

    installButton();
})();

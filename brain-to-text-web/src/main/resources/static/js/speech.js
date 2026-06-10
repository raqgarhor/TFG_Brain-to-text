function speakFromButton(button) {
    const text = button.dataset.speak;
    if (!text) {
        return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
}

document.addEventListener("DOMContentLoaded", function () {
    const viewer = document.querySelector("[data-step-viewer]");
    if (!viewer) {
        return;
    }

    const cards = Array.from(viewer.querySelectorAll(".phase-slide"));
    const nextButton = viewer.querySelector("[data-next-step]");
    const prevButton = viewer.querySelector("[data-prev-step]");
    const label = viewer.querySelector("[data-step-label]");
    let current = 0;

    function render() {
        cards.forEach(function (card, index) {
            card.classList.toggle("active", index === current);
        });

        if (label) {
            label.textContent = (current + 1) + " / " + cards.length;
        }

        if (nextButton) {
            nextButton.disabled = current === cards.length - 1;
        }

        if (prevButton) {
            prevButton.disabled = current === 0;
        }
    }

    if (nextButton) {
        nextButton.addEventListener("click", function () {
            current = Math.min(current + 1, cards.length - 1);
            render();
        });
    }

    if (prevButton) {
        prevButton.addEventListener("click", function () {
            current = Math.max(current - 1, 0);
            render();
        });
    }

    render();
});

document.querySelectorAll("[data-copy]").forEach(button => {
  button.addEventListener("click", async () => {
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      button.textContent = "Copied";
    } catch (_error) {
      button.textContent = "Select text";
    }
    window.setTimeout(() => { button.textContent = original; }, 1600);
  });
});

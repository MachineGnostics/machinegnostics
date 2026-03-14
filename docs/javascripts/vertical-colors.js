
// Define colors for each vertical
const verticalAccents = {
  "/da/": "orange",
  "/models/": "blue",
  "/magnet/": "deep-purple"
};

// Default accents from mkdocs.yml configuration
// - scheme: slate -> accent: green
// - scheme: default -> accent: cyan
const defaultAccents = {
  "slate": "green",
  "default": "cyan"
};

function setVerticalAccent() {
  const path = window.location.pathname;
  const body = document.querySelector("body");
  const scheme = body.getAttribute("data-md-color-scheme") || "default";
  const currentAccent = body.getAttribute("data-md-color-accent");
  
  let targetColor = null;

  // Check if we are in a vertical
  for (const [key, color] of Object.entries(verticalAccents)) {
    if (path.includes(key)) {
      targetColor = color;
      break;
    }
  }
  
  // If not in a vertical, fallback to the scheme's default
  if (!targetColor) {
    targetColor = defaultAccents[scheme] || "teal";
  }

  // Apply only if changed to avoid infinite loops with MutationObserver
  if (currentAccent !== targetColor) {
    body.setAttribute("data-md-color-accent", targetColor);
  }
}

// Apply on initial load
setVerticalAccent();

// Re-apply on navigation (for mkdocs-material instant loading)
if (window.document$) {
  window.document$.subscribe(function() {
    setVerticalAccent();
  });
}

// Watch for theme changes or external accent updates
const observer = new MutationObserver(function(mutations) {
  // We don't need to check mutation type, just run the logic
  // The logic inside setVerticalAccent handles the "no-change" case
  setVerticalAccent();
});

observer.observe(document.querySelector("body"), {
  attributes: true,
  attributeFilter: ["data-md-color-scheme", "data-md-color-accent"]
});

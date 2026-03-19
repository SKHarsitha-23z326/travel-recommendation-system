/**
 * script.js — WanderMatch Frontend JavaScript
 * Handles:
 *   - Form loading spinner
 *   - Intersection Observer for scroll-triggered fade-in (results cards)
 *   - Chart.js accuracy bar chart (results + about pages)
 */

// ─────────────────────────────────────────────
// FORM SUBMIT — Show loading spinner
// ─────────────────────────────────────────────
const prefForm = document.getElementById("pref-form");
if (prefForm) {
  prefForm.addEventListener("submit", function () {
    const btnText   = document.querySelector(".btn-text");
    const btnLoader = document.getElementById("btn-loader");
    if (btnText && btnLoader) {
      btnText.style.display   = "none";
      btnLoader.style.display = "inline-flex";
    }
  });
}

// ─────────────────────────────────────────────
// INTERSECTION OBSERVER — Trigger fade-in on cards
// ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", function () {
  // Re-observe any .fade-in elements that are not yet visible
  const fadeEls = document.querySelectorAll(".fade-in");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            // The CSS animation already kicks in via animation-delay.
            // We just ensure visibility for elements that loaded off-screen.
            entry.target.style.opacity = "";
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 }
    );
    fadeEls.forEach((el) => observer.observe(el));
  } else {
    // Fallback for old browsers
    fadeEls.forEach((el) => (el.style.opacity = "1"));
  }
});

// ─────────────────────────────────────────────
// CHART.JS — Model Accuracy Bar Chart
// Called by results.html and about.html after
// the accuracy data is injected via Jinja2.
// ─────────────────────────────────────────────
function initAccuracyChart(canvasId, data) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  // Shorter display labels
  const shortLabels = data.labels.map((l) =>
    l.replace("Random Forest", "Random\nForest")
     .replace("Decision Tree", "Decision\nTree")
     .replace("Logistic Regression", "Log.\nRegression")
  );

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: data.labels,
      datasets: [
        {
          label: "Accuracy (%)",
          data: data.accuracy,
          backgroundColor: [
            "rgba(245,166,35,0.85)",
            "rgba(130,167,255,0.75)",
            "rgba(72,199,142,0.75)",
            "rgba(255,130,120,0.75)",
          ],
          borderColor: [
            "rgba(245,166,35,1)",
            "rgba(100,140,255,1)",
            "rgba(50,180,120,1)",
            "rgba(235,90,80,1)",
          ],
          borderWidth: 2,
          borderRadius: 8,
          borderSkipped: false,
        },
        {
          label: "Precision (%)",
          data: data.precision,
          backgroundColor: [
            "rgba(245,166,35,0.3)",
            "rgba(130,167,255,0.3)",
            "rgba(72,199,142,0.3)",
            "rgba(255,130,120,0.3)",
          ],
          borderColor: [
            "rgba(245,166,35,0.6)",
            "rgba(100,140,255,0.6)",
            "rgba(50,180,120,0.6)",
            "rgba(235,90,80,0.6)",
          ],
          borderWidth: 2,
          borderRadius: 8,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: "rgba(255,255,255,0.75)",
            font: { family: "Poppins", size: 11 },
          },
        },
        tooltip: {
          backgroundColor: "rgba(10,35,66,0.92)",
          titleColor: "#f5a623",
          bodyColor: "rgba(255,255,255,0.85)",
          padding: 12,
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: "rgba(255,255,255,0.65)",
            font: { family: "Poppins", size: 10 },
            maxRotation: 0,
          },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
        y: {
          min: 0,
          max: 100,
          ticks: {
            color: "rgba(255,255,255,0.55)",
            font: { family: "Poppins", size: 10 },
            callback: (v) => v + "%",
          },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
      },
      animation: {
        duration: 1000,
        easing: "easeOutQuart",
      },
    },
  });
}

/**
 * Main JavaScript for Inventory Optimization Dashboard
 * Handles interactive elements, form validation, and chart initialization
 */

document.addEventListener("DOMContentLoaded", function () {
  // Initialize tooltips
  const tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });

  // File upload handling
  setupFileUploads();

  // Initialize charts if we're on the results page
  if (document.getElementById("forecastChart")) {
    initializeCharts();
  }
});

/**
 * Sets up file upload functionality with drag and drop support
 */
function setupFileUploads() {
  const fileInputs = document.querySelectorAll(".file-input");

  fileInputs.forEach((input) => {
    const fileInfo = document.getElementById(`${input.id}_info`);
    const uploadArea = input.closest(".upload-area");

    // Update file info when a file is selected
    input.addEventListener("change", function (e) {
      if (this.files && this.files.length > 0) {
        fileInfo.textContent = this.files[0].name;
        fileInfo.classList.add("text-primary");
        uploadArea.classList.add("border-primary");
      } else {
        fileInfo.textContent = "No file selected";
        fileInfo.classList.remove("text-primary");
        uploadArea.classList.remove("border-primary");
      }
    });

    // Highlight drop area when dragging files over
    ["dragenter", "dragover"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, highlight, false);
    });

    // Remove highlight when dragging leaves
    ["dragleave", "drop"].forEach((eventName) => {
      uploadArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    uploadArea.addEventListener("drop", handleDrop, false);
  });

  // Form validation
  const form = document.getElementById("uploadForm");
  if (form) {
    form.addEventListener("submit", function (e) {
      const requiredInputs = form.querySelectorAll("input[required]");
      let isValid = true;

      requiredInputs.forEach((input) => {
        if (!input.files || input.files.length === 0) {
          isValid = false;
          const uploadArea = input.closest(".upload-area");
          uploadArea.classList.add("border-danger");
          setTimeout(() => uploadArea.classList.remove("border-danger"), 2000);
        }
      });

      if (!isValid) {
        e.preventDefault();
        showAlert("Please upload all required files", "danger");
      } else {
        // Show loading state
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML =
          '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Processing...';
      }
    });
  }
}

/**
 * Handles file drop event
 * @param {Event} e - The drop event
 */
function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;
  const input = this.querySelector('input[type="file"]');

  if (files.length > 0) {
    input.files = files;
    const event = new Event("change");
    input.dispatchEvent(event);
  }
}

/**
 * Highlights the drop zone when dragging files over it
 * @param {Event} e - The drag event
 */
function highlight(e) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.add("border-primary", "bg-light");
}

/**
 * Removes highlight from drop zone
 * @param {Event} e - The drag event
 */
function unhighlight(e) {
  e.preventDefault();
  e.stopPropagation();
  this.classList.remove("border-primary", "bg-light");
}

/**
 * Initializes charts on the results page
 */
function initializeCharts() {
  // This would be populated with actual chart initialization code
  // For now, it's a placeholder for future implementation
  console.log("Initializing charts...");

  // Example of initializing a chart (would be replaced with actual data)
  const ctx = document.getElementById("forecastChart");
  if (ctx) {
    new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Forecast",
            data: [],
            borderColor: "#0d6efd",
            tension: 0.1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
          },
        },
      },
    });
  }
}

/**
 * Shows a temporary alert message
 * @param {string} message - The message to display
 * @param {string} type - The alert type (e.g., 'success', 'danger', 'warning')
 */
function showAlert(message, type = "info") {
  const alertDiv = document.createElement("div");
  alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
  alertDiv.role = "alert";
  alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

  const container = document.querySelector(".container");
  if (container) {
    container.insertBefore(alertDiv, container.firstChild);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      const alert = bootstrap.Alert.getOrCreateInstance(alertDiv);
      alert.close();
    }, 5000);
  }
}

/**
 * Toggles the visibility of an element
 * @param {string} elementId - The ID of the element to toggle
 * @param {boolean} show - Whether to show or hide the element
 */
function toggleElement(elementId, show) {
  const element = document.getElementById(elementId);
  if (element) {
    element.style.display = show ? "block" : "none";
  }
}

/**
 * Formats a number with commas as thousand separators
 * @param {number} num - The number to format
 * @returns {string} The formatted number
 */
function formatNumber(num) {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Export functions for use in other scripts if needed
window.InventoryDashboard = {
  showAlert,
  toggleElement,
  formatNumber,
};

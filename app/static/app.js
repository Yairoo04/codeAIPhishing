const tabResults = {
  url: { resultHTML: '', features: null },
  image: { resultHTML: '', features: null },
  file: { resultHTML: '', features: null },
  email: { resultHTML: '', features: null }
};

document.querySelectorAll('button[id^="tab-"]').forEach((tab) => {
  tab.addEventListener("click", () => {
    document
      .querySelectorAll('button[id^="tab-"]')
      .forEach((t) => t.classList.remove("tab-active"));
    tab.classList.add("tab-active");

    document
      .querySelectorAll(".tab-content")
      .forEach((content) => content.classList.add("hidden"));
    const sectionId = tab.id.replace("tab-", "") + "-section";
    document.getElementById(sectionId).classList.remove("hidden");

    const tabType = tab.id.replace("tab-", "");
    const savedResult = tabResults[tabType];
    if (savedResult.resultHTML) {
      document.getElementById("resultContent").innerHTML = savedResult.resultHTML;
      document.getElementById("resultSection").classList.remove("hidden");
      if (savedResult.features) {
        displayFeatures(savedResult.features);
      } else {
        document.getElementById("featuresSection").classList.add("hidden");
      }
    } else {
      document.getElementById("resultSection").classList.add("hidden");
    }
  });
});

function showLoading() {
  const resultContent = document.getElementById("resultContent");
  resultContent.innerHTML = `
    <div class="flex justify-center" id="loadingSpinner">
      <div class="loading-spinner"></div>
    </div>`;
  document.getElementById("resultSection").classList.remove("hidden");
}

function hideLoading() {
  const spinner = document.getElementById("loadingSpinner");
  if (spinner) {
    spinner.remove();
  }
}

// Kiểm tra URL
function checkURL() {
  const urlInput = document.getElementById("urlInput");
  let url = urlInput.value.trim();

  if (!url) {
    alert("Vui lòng nhập URL!");
    return;
  }
  if (!/^https?:\/\//i.test(url)) {
    url = "https://" + url;
  }
  const urlPattern = /^(https?:\/\/)([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(\/\S*)?$/;
  if (!urlPattern.test(url)) {
    alert("Vui lòng nhập URL hợp lệ (ví dụ: https://example.com)");
    return;
  }

  showLoading();
  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  })
    .then((response) => response.json())
    .then((data) => {
      hideLoading();
      if (data.error) {
        document.getElementById("resultContent").innerHTML = `<p class="text-red-600">Lỗi: ${data.error}</p>`;
        tabResults.url = { resultHTML: `<p class="text-red-600">Lỗi: ${data.error}</p>`, features: null };
        document.getElementById("resultSection").classList.remove("hidden");
        document.getElementById("featuresSection").classList.add("hidden");
        return;
      }
      let resultHTML = `
        <p><strong class="text-gray-700">URL:</strong> <span class="result-highlight">${url}</span></p>
        <p><strong class="text-gray-700">Dự đoán:</strong> 
          <span class="${data.result === "Phishing" ? "text-red-600" : "text-green-600"} font-semibold">${data.result}</span>
        </p>
        <p><strong class="text-gray-700">Xác suất Phishing:</strong> 
          <span class="${data.result === "Phishing" ? "text-red-600" : "text-green-600"}">${(data.rf_confidence * 100).toFixed(2)}%</span>
        </p>
      `;
      tabResults.url = { resultHTML, features: data.features };
      document.getElementById("resultContent").innerHTML = resultHTML;
      document.getElementById("resultSection").classList.remove("hidden");
      if (data.features) {
        displayFeatures(data.features);
      } else {
        document.getElementById("featuresSection").classList.add("hidden");
      }
    })
    .catch((error) => {
      hideLoading();
      document.getElementById("resultContent").innerHTML = `<p class="text-red-600">Lỗi: ${error.message}</p>`;
      tabResults.url = { resultHTML: `<p class="text-red-600">Lỗi: ${error.message}</p>`, features: null };
      document.getElementById("resultSection").classList.remove("hidden");
      document.getElementById("featuresSection").classList.add("hidden");
    });
}

// Tải lên và kiểm tra tệp (image, file, email)
function uploadFile(type) {
  const input = document.getElementById(`${type}Input`).files[0];
  if (!input) {
    alert("Vui lòng chọn tệp!");
    return;
  }

  let previewHTML = "";
  if (type === "image" && input.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = function (e) {
      previewHTML = `
        <div class="mb-4">
          <p class="text-gray-700 font-semibold mb-2">Ảnh đã tải lên:</p>
          <img src="${e.target.result}" alt="Ảnh tải lên" class="max-w-xs rounded-lg border shadow">
        </div>
      `;
      if (tabResults[type].resultHTML) {
        tabResults[type].resultHTML = previewHTML + tabResults[type].resultHTML;
        document.getElementById("resultContent").innerHTML = tabResults[type].resultHTML;
      }
    };
    reader.readAsDataURL(input);
  }

  showLoading();
  const formData = new FormData();
  formData.append("file", input);

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      hideLoading();
      if (data.error) {
        document.getElementById("resultContent").innerHTML = `<p class="text-red-600">Lỗi: ${data.error}</p>`;
        tabResults[type] = { resultHTML: `<p class="text-red-600">Lỗi: ${data.error}</p>`, features: null };
        document.getElementById("resultSection").classList.remove("hidden");
        document.getElementById("featuresSection").classList.add("hidden");
        return;
      }

      let resultHTML = previewHTML;
      if (data.qr_results) {
        resultHTML += '<p><strong class="text-gray-700">QR Codes Detected:</strong></p>';
        data.qr_results.forEach((qr, index) => {
          resultHTML += `
            <div class="feature-item p-3 rounded-lg mb-4">
              <p><strong>QR #${index + 1}:</strong> ${qr.qr_url}</p>
              <p><strong>Xác suất Phishing:</strong> 
                <span class="${qr.result === "Phishing" ? "text-red-600" : "text-green-600"}">${(qr.rf_confidence * 100).toFixed(2)}%</span>
              </p>
              <p><strong>Kết quả:</strong> 
                <span class="${qr.result === "Phishing" ? "text-red-600" : "text-green-600"}">${qr.result}</span>
              </p>
              <div class="features-list mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
                ${Object.entries(qr.features)
                  .map(([key, value]) => `
                    <div class="bg-gray-100 p-2 rounded">
                      <span class="font-medium">${key}:</span> ${value}
                    </div>
                  `)
                  .join("")}
              </div>
            </div>
          `;
        });
      } else {
        resultHTML += `
          <p><strong class="text-gray-700">Tệp:</strong> <span class="result-highlight">${input.name}</span></p>
          <p><strong class="text-gray-700">Dự đoán:</strong> 
            <span class="${data.result === "Phishing" ? "text-red-600" : "text-green-600"} font-semibold">${data.result}</span>
          </p>
          <p><strong class="text-gray-700">Xác suất:</strong> 
            <span class="${data.result === "Phishing" ? "text-red-600" : "text-green-600"}">${(type === "image" ? data.cnn_confidence : data.rf_confidence * 100).toFixed(2)}%</span>
          </p>
        `;
        if (data.image_url) {
          resultHTML += `
            <div class="mt-4">
              <img src="${data.image_url}" alt="Ảnh đã tải lên" class="max-w-full h-auto rounded-lg shadow-md">
            </div>
          `;
        }
      }

      tabResults[type] = { resultHTML, features: data.features && !data.qr_results ? data.features : null };
      document.getElementById("resultContent").innerHTML = resultHTML;
      document.getElementById("resultSection").classList.remove("hidden");

      if (data.features && !data.qr_results) {
        displayFeatures(data.features);
      } else {
        document.getElementById("featuresSection").classList.add("hidden");
      }
    })
    .catch((error) => {
      hideLoading();
      document.getElementById("resultContent").innerHTML = `<p class="text-red-600">Lỗi: ${error.message}</p>`;
      tabResults[type] = { resultHTML: `<p class="text-red-600">Lỗi: ${error.message}</p>`, features: null };
      document.getElementById("resultSection").classList.remove("hidden");
      document.getElementById("featuresSection").classList.add("hidden");
    });
}

function displayFeatures(features) {
  document.getElementById("featuresSection").classList.remove("hidden");
  const featuresList = document.getElementById("featuresList");
  featuresList.innerHTML = "";
  for (const [key, value] of Object.entries(features)) {
    featuresList.innerHTML += `
      <div class="feature-item p-3 rounded-lg">
        <span class="font-medium text-gray-700">${key}:</span> 
        <span class="text-gray-900">${value}</span>
      </div>
    `;
  }
}
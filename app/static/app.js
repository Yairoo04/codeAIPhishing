function showLoading(target) {
  document.getElementById(target).innerHTML = `<p style="color: blue;">Đang kiểm tra...</p>`;
}

function checkURL() {
  const urlInput = document.getElementById("urlInput");
  let url = urlInput.value.trim();

  if (!url) {
    alert("Please enter a URL!");
    return;
  }

  if (!/^https?:\/\//i.test(url)) {
    url = 'https://' + url;
  }

  const urlPattern = /^(https?:\/\/)([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(\/\S*)?$/;

  if (!urlPattern.test(url)) {
    alert("Please enter a valid URL (e.g. https://example.com)");
    return;
  }

  showLoading("urlResult");

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    if (data.error) {
      document.getElementById("urlResult").innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
      return;
    }

    document.getElementById("urlResult").innerHTML = `
      <p><strong>URL:</strong> ${data.url}</p>
      <p><strong>Random Forest Confidence:</strong> ${data.rf_confidence}</p>
      
      <p><strong>Final Decision:</strong> 
        <span style="color: ${data.result === 'Phishing' ? 'red' : 'green'}; font-weight: bold;">
          ${data.result}
        </span>
      </p>
    `;
  })
  .catch(error => {
    console.error("Error:", error);
    document.getElementById("urlResult").innerHTML = `<p style="color: red;">Failed to check URL: ${error.message}</p>`;
  });
}

function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];

  if (!file) {
    alert("Please select a file!");
    return;
  }

  showLoading("fileResult");

  const formData = new FormData();
  formData.append("file", file);

  if (file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = function (e) {
      document.getElementById("preview").innerHTML = `
        <p><strong>Uploaded Image:</strong></p>
        <img src="${e.target.result}" alt="Uploaded Image" style="max-width: 300px; border: 2px solid #ddd; padding: 5px; border-radius: 8px;">
      `;
    };
    reader.readAsDataURL(file);
  } else {
    document.getElementById("preview").innerHTML = `
      <p><strong>Uploaded File:</strong> ${file.name} (${formatFileSize(file.size)})</p>
    `;
  }

  fetch("/predict", {
    method: "POST",
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      document.getElementById("fileResult").innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
      return;
    }
    
    if (data.qr_results) {
      let qrContent = `<p><strong>QR Codes Detected:</strong></p>`;
      data.qr_results.forEach((qr, index) => {
        qrContent += `
          <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; border-radius: 8px;">
            <p><strong>QR #${index + 1}:</strong> ${qr.qr_url}</p>
            <p><strong>Random Forest Confidence:</strong> ${qr.rf_confidence}</p>
            <p><strong>Final Decision:</strong> 
              <span style="color: ${qr.result === 'Phishing' ? 'red' : 'green'}">${qr.result}</span>
            </p>
          </div>
        `;
      });
      document.getElementById("fileResult").innerHTML = qrContent;
    } else {
      const fileType = getFileType(file);
      let resultHTML = `<p><strong>File Type:</strong> ${fileType}</p>`;

      if (fileType === "Image") {
        // Chỉ hiển thị CNN confidence cho ảnh
        resultHTML += `
          <p><strong>CNN Confidence:</strong> ${data.cnn_confidence}</p>
          <p><strong>Final Decision:</strong> 
            <span style="color: ${data.result === 'Phishing' ? 'red' : 'green'}">${data.result}</span>
          </p>
        `;
      } else {
        // Trường hợp Email hoặc PDF
        resultHTML += `
          <p><strong>Random Forest Confidence:</strong> ${data.rf_confidence}</p>
          <p><strong>Final Decision:</strong> 
            <span style="color: ${data.result === 'Phishing' ? 'red' : 'green'}">${data.result}</span>
          </p>
        `;
      }

      document.getElementById("fileResult").innerHTML = resultHTML;
    }
  })
  .catch(error => {
    console.error("Error:", error);
    document.getElementById("fileResult").innerHTML = `<p style="color: red;">Failed to process file: ${error.message}</p>`;
  });
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " bytes";
  else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + " KB";
  else return (bytes / 1048576).toFixed(2) + " MB";
}

function getFileType(file) {
  const extension = file.name.split('.').pop().toLowerCase();
  
  if (file.type.startsWith("image/")) return "Image";
  else if (extension === "pdf") return "PDF";
  else if (extension === "eml") return "Email";
  else return "Unknown";
}
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("preview");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const previewSize = document.getElementById("previewSize");
  const scanBtn = document.getElementById("scanBtn");
  const showVideoBtn = document.getElementById("showVideoBtn");
  const videoContainer = document.getElementById("videoContainer");

  function handleFileSelected(file) {
    if (!file) {
      scanBtn.disabled = true;
      preview.classList.remove("visible");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewName.textContent = file.name;
      previewSize.textContent = formatBytes(file.size);
      preview.classList.add("visible");
    };
    reader.readAsDataURL(file);
    scanBtn.disabled = false;
  }

  fileInput.addEventListener("change", () => {
    handleFileSelected(fileInput.files[0]);
  });

  ["dragover", "dragenter"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(file);
    }
  });

  showVideoBtn.addEventListener("click", () => {
    videoContainer.classList.toggle("visible");
  });

  const enrollFileInput = document.getElementById("enrollFileInput");
  const enrollFileCount = document.getElementById("enrollFileCount");
  if (enrollFileInput) {
    enrollFileInput.addEventListener("change", () => {
      const n = enrollFileInput.files.length;
      enrollFileCount.textContent = n
        ? `${n} photo${n === 1 ? "" : "s"} selected`
        : "png / jpg · 3-10 photos";
    });
  }
});

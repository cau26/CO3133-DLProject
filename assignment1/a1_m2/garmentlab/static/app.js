"use strict";

const el = (id) => document.getElementById(id);
const state = {models: null, file: null, objectURL: null, pending: false, mode: "reference"};
const classLabels = {"T-shirt/top": "T-shirt / top", "Trouser": "Trouser", "Pullover": "Pullover", "Dress": "Dress", "Coat": "Coat", "Sandal": "Sandal", "Shirt": "Shirt", "Sneaker": "Sneaker", "Bag": "Bag", "Ankle boot": "Ankle boot"};

function showMessage(text) {
  el("message").textContent = text || "";
  el("message").hidden = !text;
}

function ready() {
  return Boolean(state.models && state.models[state.mode] && state.models[state.mode].ready);
}

function updateControls() {
  const reference = state.mode === "reference";
  el("sample-section").hidden = !reference;
  el("crop-section").hidden = reference;
  el("upload-hint").textContent = reference ? "PNG 28×28 thang xám · tối đa 5 MiB" : "PNG hoặc JPEG · một trang phục trong vùng crop · tối đa 5 MiB";
  el("mode-note").textContent = reference
    ? "Mô hình tham chiếu dành cho Fashion-MNIST; khả năng nhận diện ảnh chụp thật chưa được xác nhận."
    : ready() ? "Chọn một trang phục rõ ràng. Phạm vi nhãn và tiền xử lý theo phiên bản mô hình đang nạp."
      : "Nhánh ảnh quần áo thật chưa có mô hình được nạp. Cần hoàn thành dữ liệu, training và đánh giá riêng trước khi sử dụng.";
  el("predict-button").disabled = state.pending || !state.file || !ready();
  el("predict-button").firstChild.textContent = state.pending ? "Đang xử lý… " : !ready() ? "Mô hình chưa sẵn sàng " : "Phân tích ảnh ";
  el("image-file").disabled = state.pending;
  document.querySelectorAll("[name=mode],.sample").forEach((node) => { node.disabled = state.pending; });
}

function clearPrediction() {
  el("prediction").hidden = true;
  el("empty-state").hidden = false;
  el("result-mode").textContent = "Chưa có dự đoán";
}

function setFile(file) {
  if (state.objectURL) URL.revokeObjectURL(state.objectURL);
  state.objectURL = null;
  state.file = null;
  showMessage("");
  clearPrediction();
  if (file && file.size > 5 * 1024 * 1024) {
    showMessage("Ảnh vượt giới hạn 5 MiB. Vui lòng chọn tệp nhỏ hơn.");
    file = null;
  }
  if (file) {
    state.file = file;
    state.objectURL = URL.createObjectURL(file);
    el("preview").src = state.objectURL;
    el("file-name").textContent = file.name;
    el("file-info").textContent = `${(file.size / 1024).toFixed(1)} KiB`;
  } else {
    el("preview").removeAttribute("src");
    el("image-file").value = "";
  }
  el("preview-row").hidden = !file;
  updateControls();
}

function cropValue() {
  if (state.mode === "reference") return null;
  const values = ["crop-left", "crop-top", "crop-right", "crop-bottom"].map((id) => el(id).value.trim());
  if (values.some((value) => value === "" || !/^\d+$/.test(value))) {
    throw new Error("Chế độ ảnh thật cần đủ bốn tọa độ pixel nguyên cho vùng chứa một trang phục.");
  }
  const crop = values.map(Number);
  if (crop[2] <= crop[0] || crop[3] <= crop[1]) throw new Error("Right cần lớn hơn Left; Bottom cần lớn hơn Top.");
  return JSON.stringify(crop);
}

function render(payload) {
  const result = payload.result;
  if (!result || !result.predicted_class || !Array.isArray(result.probabilities)) throw new Error("Dịch vụ trả về kết quả không đúng định dạng.");
  const prediction = result.predicted_class;
  const probability = Number(prediction.probability);
  if (!Number.isFinite(probability)) throw new Error("Xác suất trả về không hợp lệ.");
  el("predicted-label").textContent = classLabels[prediction.label] || prediction.label;
  el("predicted-probability").textContent = `${(probability * 100).toFixed(1)}%`;
  el("confidence-note").textContent = result.calibrated
    ? result.review_policy === "review_all" ? "Phiên bản này chưa đủ bằng chứng để tự động nhận dự đoán. Kết quả cần người kiểm tra."
      : result.review_required ? "Dự đoán này cần được kiểm tra lại theo ngưỡng của phiên bản mô hình." : "Xác suất đã áp dụng bước calibration của phiên bản này. Vẫn cần xem phạm vi đánh giá."
    : "Đây là xác suất softmax chưa được calibration. Giá trị cao không bảo đảm dự đoán đúng; cần kiểm tra lại kết quả.";
  el("probability-label").textContent = result.calibrated ? "Đã calibration" : "Softmax chưa calibration";
  el("probability-bars").replaceChildren();
  [...result.probabilities].sort((a, b) => b.probability - a.probability).forEach((item) => {
    const value = Number(item.probability);
    if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error("Phân bố xác suất không hợp lệ.");
    const row = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = classLabels[item.label] || item.label;
    const track = document.createElement("span");
    track.className = "track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.width = `${value * 100}%`;
    track.append(fill);
    const score = document.createElement("span");
    score.className = "prob-value";
    score.textContent = `${(value * 100).toFixed(1)}%`;
    row.append(label, track, score);
    el("probability-bars").append(row);
  });
  el("model-version").textContent = result.model_version || "Chưa cung cấp";
  const input = result.input || {};
  el("input-shape").textContent = [input.width && input.height ? `${input.width}×${input.height}` : "", input.mode || "", input.crop ? `crop ${JSON.stringify(input.crop)}` : ""].filter(Boolean).join(" · ") || "Chưa cung cấp";
  el("preprocessing-version").textContent = result.preprocessing && result.preprocessing.version || "Chưa cung cấp";
  const total = result.timing_ms && Number(result.timing_ms.total);
  el("request-time").textContent = Number.isFinite(total) ? `${total.toFixed(2)} ms` : "Chưa cung cấp";
  el("request-id").textContent = payload.request_id || "Chưa cung cấp";
  el("result-mode").textContent = state.mode === "reference" ? "Tham chiếu Fashion-MNIST" : "Ảnh quần áo thật";
  el("empty-state").hidden = true;
  el("prediction").hidden = false;
}

async function loadModels() {
  try {
    const response = await fetch("/v1/models");
    if (!response.ok) throw new Error("Dịch vụ chưa phản hồi.");
    state.models = await response.json();
    if (state.models.garment.ready) {
      state.mode = "garment";
      document.querySelector('[name="mode"][value="garment"]').checked = true;
    }
    el("reference-status").textContent = state.models.reference.ready ? "Đã nạp" : "Chưa nạp";
    el("garment-status").textContent = state.models.garment.ready ? "Đã nạp" : "Chưa sẵn sàng";
    el("service-status").textContent = state.models.garment.ready ? "Dịch vụ trên máy đang hoạt động" : state.models.reference.ready ? "Đang hoạt động · chỉ mô hình tham chiếu" : "Dịch vụ hoạt động · chưa nạp mô hình";
    el("status-dot").classList.toggle("ready", state.models.reference.ready || state.models.garment.ready);
  } catch (error) {
    el("service-status").textContent = "Không kết nối được dịch vụ";
    showMessage(error.message);
  }
  updateControls();
}

el("image-file").addEventListener("change", (event) => setFile(event.target.files[0] || null));
el("clear-image").addEventListener("click", () => setFile(null));
document.querySelectorAll("[name=mode]").forEach((radio) => radio.addEventListener("change", () => {
  state.mode = radio.value;
  setFile(null);
  ["crop-left", "crop-top", "crop-right", "crop-bottom"].forEach((id) => { el(id).value = ""; });
}));
document.querySelectorAll("[data-sample]").forEach((button) => button.addEventListener("click", async () => {
  try {
    const response = await fetch(`/static/samples/${button.dataset.sample}`);
    if (!response.ok) throw new Error("Không tải được ảnh mẫu.");
    setFile(new File([await response.blob()], button.dataset.sample, {type: "image/png"}));
  } catch (error) { showMessage(error.message); }
}));
el("predict-button").addEventListener("click", async () => {
  if (state.pending || !state.file || !ready()) return;
  const form = new FormData();
  form.append("file", state.file);
  try {
    const crop = cropValue();
    if (crop) form.append("crop", crop);
  } catch (error) { showMessage(error.message); return; }
  state.pending = true;
  showMessage("");
  clearPrediction();
  updateControls();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(state.mode === "reference" ? "/v1/reference/predict" : "/v1/predict", {method: "POST", body: form, signal: controller.signal});
    const payload = await response.json();
    if (!response.ok) {
      const message = payload.error && payload.error.message || "Không thể xử lý ảnh.";
      throw new Error(`${message}${payload.request_id ? ` (Request ${payload.request_id})` : ""}`);
    }
    render(payload);
  } catch (error) {
    clearPrediction();
    showMessage(error.name === "AbortError" ? "Lần xử lý vượt 30 giây. Hãy kiểm tra dịch vụ và thử lại." : error.message);
  } finally {
    clearTimeout(timeout);
    state.pending = false;
    updateControls();
  }
});
loadModels();

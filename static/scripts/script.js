let selectedFile = null;
let fileId = null;
let isSending = false;
let isUploading = false;
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const fileNameEl = document.getElementById("fileName");
const fileIdEl = document.getElementById("fileId");
const bar = document.getElementById("bar");
const statusText = document.getElementById("statusText");
const kbBadge = document.getElementById("kbBadge");
const chatBody = document.getElementById("chatBody");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("sendBtn");
const resetBtn = document.getElementById("resetBtn");
const clearChatBtn = document.getElementById("clearChatBtn");

function setStatus(t) {
  statusText.textContent = t;
}

function scrollChat() {
  chatBody.scrollTop = chatBody.scrollHeight;
}

function renderMath() {
  if (window.renderMathInElement) {
    try {
      renderMathInElement(chatBody, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
        strict: false,
        trust: true,
        fleqn: false,
        macros: {
          "\\RR": "\\mathbb{R}",
          "\\NN": "\\mathbb{N}",
          "\\ZZ": "\\mathbb{Z}",
          "\\QQ": "\\mathbb{Q}",
        }
      });
    } catch (e) {
      console.error("Math rendering error:", e);
    }
  }
}

function escapeHTML(str) {
  return (str || "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[m]));
}

function setBubbleContent(bubbleEl, text, who = "bot") {
  const msg = text || "";
  if (who === "bot") {
    if (window.marked) {
      // Configure marked to not escape HTML entities in code blocks
      const renderer = new marked.Renderer();
      const originalCode = renderer.code;
      renderer.code = function(code, language) {
        return originalCode.call(this, code, language);
      };
      
      bubbleEl.innerHTML = marked.parse(msg, {
        renderer: renderer,
        breaks: true,
        gfm: true
      });
    } else {
      bubbleEl.textContent = msg;
    }
    // Add a small delay to ensure DOM is updated before rendering math
    setTimeout(() => renderMath(), 10);
  } else {
    bubbleEl.innerHTML = escapeHTML(msg);
  }
  scrollChat();
}

function addBubble(text, who = "bot") {
  const b = document.createElement("div");
  b.className = "bubble " + who;
  setBubbleContent(b, text, who);
  chatBody.appendChild(b);
  scrollChat();
  return b;
}

async function safeJson(res) {
  const raw = await res.text();
  let data = null;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    console.error("JSON parse error:", e);
    console.error("Raw response:", raw);
    return {
      ok: false,
      status: res.status,
      message:
        "Server returned non-JSON response.\n\n```\n" +
        raw.slice(0, 800) +
        (raw.length > 800 ? "\n...(truncated)" : "") +
        "\n```",
    };
  }
  return {
    ok: res.ok,
    status: res.status,
    ...data,
  };
}

if (dropzone) {
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  });
}

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    setFile(e.target.files[0]);
  }
});

function setFile(f) {
  selectedFile = f;
  fileNameEl.textContent = f.name;
  bar.style.width = "0%";
  setStatus("File selected");
}

uploadBtn.addEventListener("click", async () => {
  if (isUploading) return; 
  if (!selectedFile) {
    setStatus("Choose a file first");
    return;
  }
  isUploading = true;
  uploadBtn.disabled = true;
  setStatus("Uploading...");
  bar.style.width = "8%";
  const formData = new FormData();
  formData.append("file", selectedFile);
  const infoBubble = addBubble("Uploading file…", "bot");
  
  try {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);
    
    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable) {
        const pct = Math.min(90, Math.round((evt.loaded / evt.total) * 90));
        bar.style.width = pct + "%";
      }
    };
    
    xhr.onload = () => {
      try {
        if (xhr.status >= 200 && xhr.status < 300) {
          let data;
          try {
            data = JSON.parse(xhr.responseText);
          } catch (e) {
            console.error("Upload response JSON parse error:", e);
            console.error("Raw response:", xhr.responseText);
            setBubbleContent(
              infoBubble,
              "Upload failed (non-JSON response):\n\n```\n" + 
              xhr.responseText.slice(0, 800) + 
              (xhr.responseText.length > 800 ? "\n...(truncated)" : "") +
              "\n```",
              "bot"
            );
            setStatus("Upload failed");
            bar.style.width = "0%";
            return;
          }
          
          // Validate that we received the expected fields
          if (!data.file_id) {
            console.error("Missing file_id in response:", data);
            setBubbleContent(
              infoBubble,
              "Upload failed: Server response missing file_id",
              "bot"
            );
            setStatus("Upload failed");
            bar.style.width = "0%";
            return;
          }
          
          fileId = data.file_id;
          fileIdEl.textContent = "file_id: " + fileId;
          kbBadge.textContent = "KB: loaded";
          bar.style.width = "100%";
          setStatus("Upload done");
          setBubbleContent(
            infoBubble, 
            data.message || "Uploaded & indexed. Now ask your question.", 
            "bot"
          );
        } else {
          setStatus("Upload failed");
          bar.style.width = "0%";
          let errorMsg = "Upload failed";
          try {
            const errorData = JSON.parse(xhr.responseText);
            errorMsg = errorData.message || errorData.error || xhr.responseText;
          } catch (e) {
            errorMsg = xhr.responseText;
          }
          setBubbleContent(infoBubble, "Upload failed:\n\n" + errorMsg, "bot");
        }
      } finally {
        isUploading = false;
        uploadBtn.disabled = false;
      }
    };
    
    xhr.onerror = () => {
      setStatus("Upload error");
      bar.style.width = "0%";
      setBubbleContent(infoBubble, "Upload error. Check console/logs.", "bot");
      isUploading = false;
      uploadBtn.disabled = false;
    };
    
    xhr.send(formData);
  } catch (err) {
    console.error("Upload exception:", err);
    setStatus("Upload error");
    bar.style.width = "0%";
    setBubbleContent(infoBubble, "Upload error:\n\n" + String(err), "bot");
    isUploading = false;
    uploadBtn.disabled = false;
  }
});

async function sendMessage() {
  if (isSending) return; 
  const userText = (promptEl.value || "").trim();
  if (!userText) {
    setStatus("Type something");
    return;
  }
  isSending = true;
  sendBtn.disabled = true;
  addBubble(userText, "user");
  promptEl.value = "";
  setStatus("Thinking...");
  const loading = addBubble("Processing…", "bot");
  
  try {
    const payload = {
      user_input: userText,
      file_id: fileId
    };
    
    console.log("Sending request with payload:", payload);
    
    const res = await fetch("/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    const data = await safeJson(res);
    
    console.log("Received response:", data);
    
    if (!data.ok) {
      setBubbleContent(
        loading,
        `**Error (${data.status})**\n\n${data.message || data.error || "Unknown server error"}`,
        "bot"
      );
      setStatus("Error");
      return;
    }
    
    // Validate response has message field
    if (!data.message && !data.response) {
      console.error("Invalid response format:", data);
      setBubbleContent(
        loading,
        "**Error**: Server returned invalid response format",
        "bot"
      );
      setStatus("Error");
      return;
    }
    
    setBubbleContent(loading, data.message || data.response || "No response", "bot");
    setStatus("Ready");
  } catch (e) {
    console.error("Send message error:", e);
    setBubbleContent(loading, "**Network Error**\n\n" + String(e), "bot");
    setStatus("Error");
  } finally {
    isSending = false;
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

resetBtn.addEventListener("click", async () => {
  setStatus("Resetting...");
  const b = addBubble("Resetting knowledge base…", "bot");
  try {
    const res = await fetch("/reset", { method: "POST" });
    const data = await safeJson(res);
    
    if (!data.ok) {
      setBubbleContent(
        b,
        `**Reset failed (${data.status})**\n\n${data.message || data.error || "Unknown error"}`,
        "bot"
      );
      setStatus("Error");
      return;
    }
    
    setBubbleContent(b, data.message || "KB reset", "bot");
    fileId = null;
    fileIdEl.textContent = "file_id: —";
    kbBadge.textContent = "KB: empty";
    bar.style.width = "0%";
    setStatus("Ready");
  } catch (e) {
    console.error("Reset error:", e);
    setBubbleContent(b, "**Reset failed**\n\n" + String(e), "bot");
    setStatus("Error");
  }
});

clearChatBtn.addEventListener("click", () => {
  chatBody.innerHTML = "";
  addBubble("Chat cleared.", "bot");
  setStatus("Ready");
});

window.addEventListener("load", () => {
  renderMath();
});

const kbToggleBtn = document.getElementById("kbToggleBtn");
const kbPanel = document.getElementById("kbPanel");
if (kbToggleBtn && kbPanel) {
  kbToggleBtn.addEventListener("click", () => {
    kbPanel.classList.toggle("open");
  });
}

/**
 * FallGuard AI - Central Dashboard Frontend Controller
 * Handles WebSocket streaming, Chart.js telemetry waveforms, REST API polling, and fall alerts.
 */

class FallGuardApp {
  constructor() {
    this.ws = null;
    this.reconnectTimer = null;
    this.activeTab = "tab-dashboard";
    this.currentSpeed = 1.0;
    this.selectedFallId = null;

    // Rolling waveform data buffers (last 50 data points)
    this.maxPoints = 50;
    this.chartLabels = Array(this.maxPoints).fill("");
    this.accXData = Array(this.maxPoints).fill(0);
    this.accYData = Array(this.maxPoints).fill(0);
    this.accZData = Array(this.maxPoints).fill(1.0);
    this.accMagData = Array(this.maxPoints).fill(1.0);
    this.gyroXData = Array(this.maxPoints).fill(0);
    this.gyroYData = Array(this.maxPoints).fill(0);
    this.gyroZData = Array(this.maxPoints).fill(0);

    // Chart instances
    this.miniAccChart = null;
    this.liveAccChart = null;
    this.liveGyroChart = null;

    // Bearer tokens are kept for this browser tab's session. Migrate tokens
    // created by older versions without forcing users to paste them again.
    this.token = sessionStorage.getItem("token") || localStorage.getItem("token");
    if (this.token && !sessionStorage.getItem("token")) {
      sessionStorage.setItem("token", this.token);
      localStorage.removeItem("token");
    }
    this.init();
  }

  async init() {
    if (!this.token) {
      window.location.href = "/login";
      return;
    }
    const profileResponse = await this.authFetch("/api/auth/me");
    if (!profileResponse || !profileResponse.ok) return;
    const profile = await profileResponse.json();
    sessionStorage.setItem("user", JSON.stringify(profile));
    this.renderUserProfile(profile);
    this.bindEvents();
    this.initCharts();
    this.connectWebSocket();
    this.loadDashboardSummary();
    this.loadModelAnalytics();
    this.loadFallIncidents();
    this.loadActivityHistory();

    // Auto-refresh summary data every 4 seconds
    setInterval(() => {
      if (this.activeTab === "tab-dashboard") this.loadDashboardSummary();
      if (this.activeTab === "tab-falls") this.loadFallIncidents();
    }, 4000);
  }

  renderUserProfile(authenticatedUser = null) {
    try {
      const stored = authenticatedUser ? JSON.stringify(authenticatedUser) : sessionStorage.getItem("user");
      if (stored) {
        const user = JSON.parse(stored);
        if (user.name) {
          const nameEl = document.getElementById("profile-name");
          if (nameEl) nameEl.textContent = user.name;
          const initials = user.name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();
          const avatarEl = document.getElementById("user-avatar");
          if (avatarEl) avatarEl.textContent = initials || "U";
        }
        if (user.role) {
          const roleEl = document.getElementById("profile-role");
          if (roleEl) roleEl.textContent = `${user.role} View`;
        }
      }
    } catch (_) {}
  }

  async authFetch(url, options = {}) {
    if (!this.token) {
      this.redirectToLogin("expired");
      return null;
    }
    const headers = {
      ...(options.headers || {}),
      "Authorization": `Bearer ${this.token}`
    };
    try {
      const res = await fetch(url, { ...options, headers });
      if (res.status === 401) {
        this.redirectToLogin("expired");
        return null;
      }
      return res;
    } catch (err) {
      console.error(`Network or fetch error for ${url}:`, err);
      return null;
    }
  }

  redirectToLogin(reason = "expired") {
    sessionStorage.removeItem("token");
    sessionStorage.removeItem("user");
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    window.location.replace(`/login?reason=${encodeURIComponent(reason)}`);
  }

  // --- UI Event Binding ---
  bindEvents() {
    // Logout button
    document.getElementById("btn-logout")?.addEventListener("click", () => {
      this.redirectToLogin("logout");
    });
    // Navigation tabs
    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const tabId = btn.getAttribute("data-tab");
        this.switchTab(tabId);
      });
    });

    document.getElementById("btn-goto-live")?.addEventListener("click", () => {
      this.switchTab("tab-live");
    });

    // Theme toggle
    document.getElementById("theme-toggle")?.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "dark";
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
    });

    // Quick Simulation buttons
    document.getElementById("btn-quick-sim-start")?.addEventListener("click", () => this.controlSim("start"));
    document.getElementById("btn-quick-sim-pause")?.addEventListener("click", () => this.controlSim("pause"));
    document.getElementById("btn-quick-sim-fall")?.addEventListener("click", () => this.controlSim("trigger_fall"));

    // Full Simulation Panel buttons
    document.getElementById("btn-sim-start")?.addEventListener("click", () => this.controlSim("start"));
    document.getElementById("btn-sim-pause")?.addEventListener("click", () => this.controlSim("pause"));
    document.getElementById("btn-sim-stop")?.addEventListener("click", () => this.controlSim("stop"));
    document.getElementById("btn-sim-manual-fall")?.addEventListener("click", () => this.controlSim("trigger_fall"));

    // Speed buttons
    document.querySelectorAll(".btn-speed").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-speed").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        this.currentSpeed = parseFloat(btn.getAttribute("data-speed"));
        this.controlSim("start"); // restart with new speed
      });
    });

    // Banner actions
    document.getElementById("btn-banner-view")?.addEventListener("click", () => {
      this.switchTab("tab-falls");
    });
    document.getElementById("btn-banner-ack")?.addEventListener("click", () => {
      if (this.currentActiveFallId) {
        this.openAcknowledgeModal(this.currentActiveFallId, "Recent Fall", "95%");
      }
    });

    // Modal buttons
    document.getElementById("btn-close-modal")?.addEventListener("click", () => this.closeAcknowledgeModal());
    document.getElementById("btn-cancel-modal")?.addEventListener("click", () => this.closeAcknowledgeModal());
    document.getElementById("btn-confirm-ack")?.addEventListener("click", () => this.submitAcknowledge());

    // Filter changes
    document.getElementById("filter-fall-status")?.addEventListener("change", () => this.loadFallIncidents());
    document.getElementById("btn-refresh-falls")?.addEventListener("click", () => this.loadFallIncidents());
    document.getElementById("filter-act-name")?.addEventListener("change", () => this.loadActivityHistory());
    document.getElementById("filter-act-risk")?.addEventListener("change", () => this.loadActivityHistory());
    document.getElementById("btn-refresh-history")?.addEventListener("click", () => this.loadActivityHistory());
  }

  switchTab(tabId) {
    this.activeTab = tabId;
    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".tab-content").forEach(content => {
      content.classList.toggle("active", content.id === tabId);
    });

    // Update heading
    const titles = {
      "tab-dashboard": ["Monitoring Dashboard", "Real-time wearable sensor stream and posture classification."],
      "tab-live": ["6-Axis Sensor Monitor", "High-frequency accelerometer and gyroscope kinematic waveforms."],
      "tab-falls": ["Fall Incidents & Safety Log", "Detected impact events, triage status, and caregiver notes."],
      "tab-history": ["Activity Audit Log", "Classified posture sliding window history."],
      "tab-analytics": ["Model Analytics & Explainability", "5-model benchmark evaluation, feature importance, and safety metrics."],
      "tab-settings": ["Hardware & IoT Integration", "ESP32 + MPU6050 firmware configuration and system thresholds."]
    };

    if (titles[tabId]) {
      document.getElementById("page-title").textContent = titles[tabId][0];
      document.getElementById("page-desc").textContent = titles[tabId][1];
    }
  }

  // --- WebSocket Connection ---
  connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/monitor`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("[WS] Connected to FallGuard telemetry server.");
      // Keep the dashboard JWT out of browser history, proxy logs and URL logs.
      this.ws.send(JSON.stringify({ type: "auth", token: this.token }));
      const pill = document.getElementById("ws-status-pill");
      const text = document.getElementById("ws-status-text");
      pill.className = "connection-status-pill connected";
      text.textContent = "WS Connected";
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleWebSocketMessage(msg);
      } catch (err) {
        console.error("Error parsing WS packet:", err);
      }
    };

    this.ws.onclose = () => {
      console.warn("[WS] Disconnected. Reconnecting in 3s...");
      const pill = document.getElementById("ws-status-pill");
      const text = document.getElementById("ws-status-text");
      pill.className = "connection-status-pill disconnected";
      text.textContent = "Disconnected (Retrying)";

      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = setTimeout(() => this.connectWebSocket(), 3000);
    };
  }

  handleWebSocketMessage(msg) {
    if (msg.type === "SENSOR_TICK") {
      this.updateWaveforms(msg.data);
    } else if (msg.type === "ACTIVITY_UPDATE") {
      this.updateActivityState(msg.data);
    } else if (msg.type === "FALL_PENDING_CONFIRMATION") {
      this.handlePendingConfirmation(msg.event);
    } else if (msg.type === "FALL_CONFIRMATION_RESOLVED") {
      this.handleConfirmationResolved(msg.data);
    } else if (msg.type === "FALL_ALERT") {
      this.handleFallAlert(msg.event);
    } else if (msg.type === "DEVICE_STATUS_UPDATE") {
      this.updateDeviceStatus(msg.data);
    }
  }

  // --- Telemetry & Chart Updates ---
  updateWaveforms(data) {
    const timeLabel = data.timestamp || "";
    this.chartLabels.shift();
    this.chartLabels.push(timeLabel);

    this.accXData.shift(); this.accXData.push(data.acc_x);
    this.accYData.shift(); this.accYData.push(data.acc_y);
    this.accZData.shift(); this.accZData.push(data.acc_z);
    this.accMagData.shift(); this.accMagData.push(data.acc_magnitude);

    this.gyroXData.shift(); this.gyroXData.push(data.gyro_x);
    this.gyroYData.shift(); this.gyroYData.push(data.gyro_y);
    this.gyroZData.shift(); this.gyroZData.push(data.gyro_z);

    // Update Live Peak Acc
    const elPeak = document.getElementById("live-peak-acc");
    if (elPeak) elPeak.textContent = `${data.acc_magnitude.toFixed(2)} g`;

    // Re-render active charts
    if (this.activeTab === "tab-dashboard" && this.miniAccChart) {
      this.miniAccChart.update("none");
    } else if (this.activeTab === "tab-live") {
      if (this.liveAccChart) this.liveAccChart.update("none");
      if (this.liveGyroChart) this.liveGyroChart.update("none");
    }
  }

  updateActivityState(data) {
    const act = data.activity;
    const conf = (data.confidence * 100).toFixed(1) + "%";
    const risk = data.risk_level;

    // Overview Tab
    const elCurAct = document.getElementById("stat-current-activity");
    const elCurConf = document.getElementById("stat-current-conf");
    const elCurRisk = document.getElementById("stat-current-risk");

    if (elCurAct) elCurAct.textContent = act;
    if (elCurConf) elCurConf.textContent = conf;
    if (elCurRisk) {
      elCurRisk.textContent = `${risk} RISK`;
      elCurRisk.className = `stat-pill-sm pill-${risk.toLowerCase()}`;
    }

    // Live Tab
    const elLiveAct = document.getElementById("live-act-name");
    const elLiveConf = document.getElementById("live-act-conf");
    const elLiveRisk = document.getElementById("live-act-risk");

    if (elLiveAct) elLiveAct.textContent = act;
    if (elLiveConf) elLiveConf.textContent = conf;
    if (elLiveRisk) {
      elLiveRisk.textContent = `${risk} RISK`;
      elLiveRisk.className = `stat-pill-sm pill-${risk.toLowerCase()}`;
    }

    // Update Probability Bars
    if (data.probabilities) {
      this.renderProbabilityBars(data.probabilities);
    }
  }

  renderProbabilityBars(probs) {
    const container = document.getElementById("prob-bars-container");
    if (!container) return;

    let html = "";
    for (const [cls, val] of Object.entries(probs)) {
      const pct = (val * 100).toFixed(1);
      const isFall = cls === "FALL";
      html += `
        <div class="prob-bar-item">
          <div class="prob-header">
            <span>${cls}</span>
            <span>${pct}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill ${isFall ? 'fall-fill' : ''}" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    }
    container.innerHTML = html;
  }

  handleFallAlert(event) {
    this.currentActiveFallId = event.id;

    // Show Global Emergency Banner
    const banner = document.getElementById("global-alert-banner");
    const msg = document.getElementById("banner-alert-msg");
    if (banner && msg) {
      msg.textContent = `${event.message} (Device: ${event.device_id || 'Room 104'})`;
      banner.style.display = "block";
    }

    // Update Sidebar Badge
    const badge = document.getElementById("sidebar-alert-badge");
    if (badge) {
      badge.style.display = "inline-block";
      const cur = parseInt(badge.textContent || "0") + 1;
      badge.textContent = cur;
    }

    // Refresh falls table if active
    this.loadFallIncidents();
    this.loadDashboardSummary();
  }

  handlePendingConfirmation(event) {
    this.currentActiveFallId = event.id || event.event_id;
    const banner = document.getElementById("global-alert-banner");
    const msg = document.getElementById("banner-alert-msg");
    if (banner && msg) {
      msg.innerHTML = `⚠️ <b>POTENTIAL FALL DETECTED</b> on device <code>${event.device_id}</code>. Smartphone user verification in progress (${event.timeout_seconds || 15}s timeout)...`;
      banner.style.display = "block";
      banner.style.backgroundColor = "rgba(245, 158, 11, 0.95)"; // Warning amber
    }
  }

  handleConfirmationResolved(data) {
    const banner = document.getElementById("global-alert-banner");
    const msg = document.getElementById("banner-alert-msg");
    if (!banner || !msg) return;

    if (data.status === "CANCELLED_BY_USER") {
      msg.innerHTML = `✅ <b>SAFE CONFIRMATION</b>: User on <code>${data.device_id}</code> pressed "I'm OK". False alarm cancelled.`;
      banner.style.backgroundColor = "rgba(16, 185, 129, 0.95)"; // Green
      setTimeout(() => {
        banner.style.display = "none";
      }, 5000);
    } else if (data.status === "CONFIRMED") {
      msg.innerHTML = `🚨 <b>CONFIRMED FALL INCIDENT</b> on device <code>${data.device_id}</code>! Emergency escalation dispatched.`;
      banner.style.backgroundColor = "rgba(239, 68, 68, 0.95)"; // Red
    }
    this.loadFallIncidents();
    this.loadDashboardSummary();
  }

  updateDeviceStatus(data) {
    const textEl = document.getElementById("phone-status-text");
    if (!textEl) return;
    if (data.connected) {
      textEl.textContent = `Online (${data.device_id})`;
      textEl.style.color = "#10B981"; // Green
    } else {
      textEl.textContent = `Disconnected`;
      textEl.style.color = "#94A3B8"; // Grey
    }
  }

  // --- Chart.js Initializers ---
  initCharts() {
    const commonChartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      elements: { line: { tension: 0.3, borderWidth: 2 }, point: { radius: 0 } },
      scales: {
        x: { display: false },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.06)" },
          ticks: { color: "#94a3b8", font: { size: 10 } }
        }
      },
      plugins: {
        legend: { labels: { color: "#94a3b8", boxWidth: 12, font: { size: 11 } } }
      }
    };

    // 1. Mini Acceleration Preview Chart
    const ctxMini = document.getElementById("mini-acc-chart")?.getContext("2d");
    if (ctxMini) {
      this.miniAccChart = new Chart(ctxMini, {
        type: "line",
        data: {
          labels: this.chartLabels,
          datasets: [
            { label: "Acc X", data: this.accXData, borderColor: "#3b82f6" },
            { label: "Acc Y", data: this.accYData, borderColor: "#10b981" },
            { label: "Acc Z", data: this.accZData, borderColor: "#8b5cf6" },
            { label: "Magnitude (g)", data: this.accMagData, borderColor: "#ef4444", borderWidth: 2.5 }
          ]
        },
        options: commonChartOptions
      });
    }

    // 2. Full Live Accelerometer Chart
    const ctxLiveAcc = document.getElementById("live-acc-chart")?.getContext("2d");
    if (ctxLiveAcc) {
      this.liveAccChart = new Chart(ctxLiveAcc, {
        type: "line",
        data: {
          labels: this.chartLabels,
          datasets: [
            { label: "Acc X (g)", data: this.accXData, borderColor: "#3b82f6" },
            { label: "Acc Y (g)", data: this.accYData, borderColor: "#10b981" },
            { label: "Acc Z (g)", data: this.accZData, borderColor: "#8b5cf6" },
            { label: "Magnitude (g)", data: this.accMagData, borderColor: "#ef4444", borderWidth: 2.5 }
          ]
        },
        options: {
          ...commonChartOptions,
          scales: {
            ...commonChartOptions.scales,
            y: { ...commonChartOptions.scales.y, min: -4.0, max: 6.0 }
          }
        }
      });
    }

    // 3. Full Live Gyroscope Chart
    const ctxLiveGyro = document.getElementById("live-gyro-chart")?.getContext("2d");
    if (ctxLiveGyro) {
      this.liveGyroChart = new Chart(ctxLiveGyro, {
        type: "line",
        data: {
          labels: this.chartLabels,
          datasets: [
            { label: "Gyro X (rad/s)", data: this.gyroXData, borderColor: "#06b6d4" },
            { label: "Gyro Y (rad/s)", data: this.gyroYData, borderColor: "#f59e0b" },
            { label: "Gyro Z (rad/s)", data: this.gyroZData, borderColor: "#ec4899" }
          ]
        },
        options: {
          ...commonChartOptions,
          scales: {
            ...commonChartOptions.scales,
            y: { ...commonChartOptions.scales.y, min: -6.0, max: 6.0 }
          }
        }
      });
    }
  }

  // --- REST API Calls ---
  async loadDashboardSummary() {
    try {
      const res = await this.authFetch("/api/dashboard/summary");
      if (!res || !res.ok) return;
      const data = await res.json();

      document.getElementById("stat-falls-today").textContent = data.falls_today_count;
      document.getElementById("stat-total-falls").textContent = data.total_falls_count;
      document.getElementById("stat-unack-badge").textContent = `${data.unacknowledged_falls_count} Pending`;

      const sideBadge = document.getElementById("sidebar-alert-badge");
      if (sideBadge) {
        if (data.unacknowledged_falls_count > 0) {
          sideBadge.style.display = "inline-block";
          sideBadge.textContent = data.unacknowledged_falls_count;
        } else {
          sideBadge.style.display = "none";
        }
      }

      // Render Active Alerts Feed
      const alertFeed = document.getElementById("dashboard-alerts-feed");
      if (alertFeed) {
        if (data.active_alerts && data.active_alerts.length > 0) {
          alertFeed.innerHTML = data.active_alerts.map(a => `
            <div class="alert-feed-item">
              <div class="alert-item-time">${new Date(a.created_at).toLocaleTimeString()}</div>
              <div class="alert-item-text">${a.message}</div>
              <button class="btn btn-sm btn-primary" onclick="window.app.openAcknowledgeModal(${a.fall_event_id}, '${new Date(a.created_at).toLocaleTimeString()}', '96%')">Acknowledge</button>
            </div>
          `).join("");
        } else {
          alertFeed.innerHTML = `<div class="empty-feed">No active unacknowledged fall alerts. System nominal.</div>`;
        }
      }

      // Render Recent Telemetry Table
      const recentTbody = document.getElementById("dashboard-recent-table-body");
      if (recentTbody && data.recent_activities) {
        if (data.recent_activities.length > 0) {
          recentTbody.innerHTML = data.recent_activities.map(r => `
            <tr>
              <td>${new Date(r.timestamp).toLocaleTimeString()}</td>
              <td><code>${r.device_id}</code></td>
              <td><b>${r.activity}</b></td>
              <td>${(r.confidence * 100).toFixed(1)}%</td>
              <td><span class="stat-pill-sm pill-${r.risk_level.toLowerCase()}">${r.risk_level}</span></td>
              <td><span class="text-${r.is_fall ? 'danger' : 'success'}">${r.is_fall ? 'FALL DETECTED' : 'NORMAL'}</span></td>
            </tr>
          `).join("");
        }
      }
    } catch (err) {
      console.error("Error loading dashboard summary:", err);
    }
  }

  async loadFallIncidents() {
    try {
      const statusFilter = document.getElementById("filter-fall-status")?.value || "";
      const url = statusFilter ? `/api/falls?status=${statusFilter}` : `/api/falls`;
      const res = await this.authFetch(url);
      if (!res || !res.ok) return;
      const falls = await res.json();

      const tbody = document.getElementById("falls-table-body");
      if (!tbody) return;

      if (falls.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-center">No fall incidents matching filter.</td></tr>`;
        return;
      }

      tbody.innerHTML = falls.map(f => `
        <tr>
          <td><b>#${f.id}</b></td>
          <td>${new Date(f.timestamp).toLocaleString()}</td>
          <td><code>${f.device_id}</code></td>
          <td>${(f.confidence * 100).toFixed(1)}%</td>
          <td>${f.acc_peak ? f.acc_peak.toFixed(2) + ' g' : '--'}</td>
          <td><span class="stat-pill-sm pill-${f.risk_level.toLowerCase()}">${f.risk_level}</span></td>
          <td><span class="stat-pill-sm ${f.acknowledged ? 'pill-ack' : 'pill-unack'}">${f.status}</span></td>
          <td><span class="notes-text">${f.notes || 'None'}</span></td>
          <td>
            ${!f.acknowledged ? `
              <button class="btn btn-sm btn-primary" onclick="window.app.openAcknowledgeModal(${f.id}, '${new Date(f.timestamp).toLocaleTimeString()}', '${(f.confidence*100).toFixed(1)}%')">Acknowledge</button>
            ` : `<span class="text-success">✓ Signed by ${f.acknowledged_by || 'Staff'}</span>`}
          </td>
        </tr>
      `).join("");
    } catch (err) {
      console.error("Error loading fall incidents:", err);
    }
  }

  async loadActivityHistory() {
    try {
      const actFilter = document.getElementById("filter-act-name")?.value || "";
      const riskFilter = document.getElementById("filter-act-risk")?.value || "";
      let query = [];
      if (actFilter) query.push(`activity=${actFilter}`);
      if (riskFilter) query.push(`risk_level=${riskFilter}`);
      const qs = query.length > 0 ? `?${query.join("&")}` : "";

      const res = await this.authFetch(`/api/activities/history${qs}`);
      if (!res || !res.ok) return;
      const records = await res.json();

      const tbody = document.getElementById("history-table-body");
      if (!tbody) return;

      if (records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center">No activity history matching criteria.</td></tr>`;
        return;
      }

      tbody.innerHTML = records.map(r => `
        <tr>
          <td>#${r.id}</td>
          <td>${new Date(r.timestamp).toLocaleTimeString()}</td>
          <td><code>${r.device_id}</code></td>
          <td><b>${r.activity}</b></td>
          <td>${(r.confidence * 100).toFixed(1)}%</td>
          <td><span class="stat-pill-sm pill-${r.risk_level.toLowerCase()}">${r.risk_level}</span></td>
          <td>${r.model_version || '1.0.0'}</td>
        </tr>
      `).join("");
    } catch (err) {
      console.error("Error loading activity history:", err);
    }
  }

  async loadModelAnalytics() {
    try {
      const [resInfo, resComp] = await Promise.all([
        this.authFetch("/api/model/info"),
        this.authFetch("/api/model/comparisons")
      ]);

      if (resInfo && resInfo.ok) {
        const info = await resInfo.json();
        document.getElementById("analytics-model-name").textContent = info.model_name;
        document.getElementById("stat-model-name").textContent = info.model_name;

        if (info.metrics) {
          const fallRec = (info.metrics.fall_recall * 100).toFixed(1) + "%";
          const macroF1 = (info.metrics.macro_f1 * 100).toFixed(1) + "%";
          document.getElementById("analytics-fall-recall").textContent = fallRec;
          document.getElementById("stat-fall-recall").textContent = fallRec;
          document.getElementById("analytics-macro-f1").textContent = macroF1;
        }
        document.getElementById("analytics-feat-count").textContent = info.features_count;

        // Render Feature Importance Top 10
        const featContainer = document.getElementById("feat-importance-container");
        if (featContainer && info.feature_importance_top10) {
          const maxVal = Math.max(...info.feature_importance_top10.map(f => f.importance), 0.01);
          featContainer.innerHTML = info.feature_importance_top10.map(f => {
            const widthPct = ((f.importance / maxVal) * 100).toFixed(1);
            return `
              <div class="feat-item">
                <span class="feat-name">${f.feature}</span>
                <div class="feat-bar-wrap">
                  <div class="feat-bar-fill" style="width: ${widthPct}%"></div>
                </div>
                <span class="feat-score">${f.importance.toFixed(4)}</span>
              </div>
            `;
          }).join("");
        }
      }

      if (resComp && resComp.ok) {
        const comps = await resComp.json();
        const tbody = document.getElementById("models-benchmark-tbody");
        if (tbody && comps.length > 0) {
          tbody.innerHTML = comps.map(c => `
            <tr>
              <td><b>${c.Model}</b></td>
              <td>${(c.Accuracy * 100).toFixed(2)}%</td>
              <td>${(c["Macro F1"] * 100).toFixed(2)}%</td>
              <td><b class="text-success">${(c["Fall Recall"] * 100).toFixed(2)}%</b></td>
              <td>${(c["Fall Precision"] * 100).toFixed(2)}%</td>
              <td>${(c["Fall FPR"] * 100).toFixed(2)}%</td>
              <td>${(c["Fall FNR"] * 100).toFixed(2)}%</td>
              <td><b class="highlight">${(c["Composite Score"] * 100).toFixed(2)}%</b></td>
            </tr>
          `).join("");
        }
      }
    } catch (err) {
      console.error("Error loading model analytics:", err);
    }
  }

  // --- Simulation Control ---
  async controlSim(action) {
    try {
      const scenario = document.getElementById("sim-scenario-select")?.value || "mixed_activities_with_fall";
      const res = await this.authFetch("/api/simulation/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: action,
          playback_speed: this.currentSpeed,
          scenario: scenario
        })
      });
      if (res && res.ok) {
        const status = await res.json();
        const badge = document.getElementById("sim-badge-status");
        if (badge) {
          badge.textContent = status.is_running ? `Streaming (${status.playback_speed}x)` : "Simulator Idle";
          badge.className = status.is_running ? "badge-status-running active" : "badge-status-running";
        }
      }
    } catch (err) {
      console.error("Error controlling simulation:", err);
    }
  }

  // --- Modal Handling ---
  openAcknowledgeModal(fallId, time, conf) {
    this.selectedFallId = fallId;
    document.getElementById("modal-fall-id").textContent = fallId;
    document.getElementById("modal-fall-time").textContent = time;
    document.getElementById("modal-fall-conf").textContent = conf;
    document.getElementById("modal-ack-notes").value = "";
    document.getElementById("ack-modal").style.display = "flex";
  }

  closeAcknowledgeModal() {
    document.getElementById("ack-modal").style.display = "none";
    this.selectedFallId = null;
  }

  async submitAcknowledge() {
    if (!this.selectedFallId) return;
    const staff = document.getElementById("modal-ack-staff").value || "Caregiver";
    const notes = document.getElementById("modal-ack-notes").value || "";

    try {
      const res = await this.authFetch(`/api/falls/${this.selectedFallId}/acknowledge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ acknowledged_by: staff, notes: notes })
      });
      if (res && res.ok) {
        this.closeAcknowledgeModal();
        document.getElementById("global-alert-banner").style.display = "none";
        this.loadFallIncidents();
        this.loadDashboardSummary();
      }
    } catch (err) {
      console.error("Error submitting acknowledgment:", err);
    }
  }
}

// Instantiate globally on page load
window.addEventListener("DOMContentLoaded", () => {
  window.app = new FallGuardApp();
});

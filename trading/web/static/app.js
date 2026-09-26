// Dashboard behaviour. Loaded as a file (the CSP forbids inline scripts);
// everything that comes from the server is written with textContent, never innerHTML.
"use strict";

(function () {
  // ------------------------------------------------------------------ live updates
  const TRADING_TYPES = new Set(["orders", "fills", "signals", "rejected", "engine"]);
  let socket = null;
  let retry = 1000;

  function setStatus(on) {
    document.querySelectorAll("[data-ws-status]").forEach((el) => {
      el.classList.toggle("on", on);
      el.textContent = on ? "live" : "offline";
    });
  }

  function describe(event) {
    const d = event.data || {};
    switch (event.type) {
      case "orders":
        return `${d.side} ${d.qty} ${d.symbol} ${d.order_type} ${d.status}` +
          (d.status_message ? ` (${d.status_message})` : "");
      case "fills":
        return `${d.side} ${d.qty} ${d.symbol} @ ${d.price}`;
      case "signals":
        return `${d.strategy_id} ${d.symbol} score ${Number(d.score).toFixed(3)}`;
      case "rejected":
        return `${d.rule}: ${d.reason}`;
      case "alerts":
        return `${d.level} ${d.source}: ${d.message}`;
      case "control":
        return `${d.command} by ${d.issued_by}${d.reason ? `: ${d.reason}` : ""}`;
      case "engine":
        return d.state + (d.error ? `: ${d.error}` : d.reason ? `: ${d.reason}` : "");
      case "kill":
        return `kill switch engaged by ${d.by}: ${d.reason}`;
      case "resume":
        return `kill switch released by ${d.by}`;
      default:
        return JSON.stringify(d).slice(0, 160);
    }
  }

  function addToFeed(event) {
    document.querySelectorAll("[data-feed]").forEach((feed) => {
      if (event.mode && feed.dataset.feed !== event.mode) return;
      const li = document.createElement("li");
      if (event.type === "alerts" || event.type === "kill") li.className = "alert";
      const t = document.createElement("span");
      t.className = "t";
      t.textContent = (event.ts || "").slice(11, 19);
      const k = document.createElement("span");
      k.className = "k";
      k.textContent = event.type;
      const text = document.createElement("span");
      text.textContent = describe(event);
      li.append(t, k, text);
      feed.prepend(li);
      while (feed.children.length > 200) feed.lastElementChild.remove();
    });
  }

  function fire(name) {
    if (window.htmx) window.htmx.trigger(document.body, name);
  }

  function connect() {
    if (!document.body.hasAttribute("data-signed-in")) return;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws/live`);
    socket.addEventListener("open", () => { retry = 1000; setStatus(true); });
    socket.addEventListener("message", (msg) => {
      let event;
      try { event = JSON.parse(msg.data); } catch { return; }
      if (event.type === "hello") return;
      addToFeed(event);
      if (event.type === "kill" || event.type === "resume") fire("kill-changed");
      if (TRADING_TYPES.has(event.type)) fire("trading-update");
    });
    socket.addEventListener("close", (e) => {
      setStatus(false);
      if (e.code === 4401) return; // signed out: do not hammer the server
      setTimeout(connect, retry);
      retry = Math.min(retry * 2, 30000);
    });
  }

  // ------------------------------------------------------------------ charts
  function initChart(fig) {
    let data;
    try { data = JSON.parse(fig.dataset.points); } catch { return; }
    const plot = fig.querySelector(".chart-plot");
    const svg = fig.querySelector("svg");
    const tip = fig.querySelector(".chart-tip");
    const cross = svg && svg.querySelector(".crosshair");
    const dot = svg && svg.querySelector(".hover-dot");
    if (!svg || !tip || !cross || !dot || !data.x || !data.x.length) return;
    let index = -1;

    function nearest(x) {
      let lo = 0, hi = data.x.length - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (data.x[mid] < x) lo = mid; else hi = mid;
      }
      return Math.abs(data.x[lo] - x) <= Math.abs(data.x[hi] - x) ? lo : hi;
    }

    function show(i) {
      index = Math.max(0, Math.min(data.x.length - 1, i));
      const x = data.x[index], y = data.y[index];
      cross.setAttribute("x1", x);
      cross.setAttribute("x2", x);
      cross.setAttribute("visibility", "visible");
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", y);
      dot.setAttribute("visibility", "visible");
      tip.textContent = "";
      const when = document.createElement("span");
      when.textContent = data.t[index];
      const value = document.createElement("span");
      value.className = "v";
      value.textContent = data.v[index];
      tip.append(when, value);
      tip.hidden = false;
      const scale = svg.getBoundingClientRect().width / data.w;
      const px = x * scale;
      const half = tip.offsetWidth / 2;
      tip.style.left = `${Math.max(half, Math.min(plot.clientWidth - half, px))}px`;
      const above = y * scale - tip.offsetHeight - 12;
      tip.style.top = `${above >= 0 ? above : y * scale + 12}px`; // flip below near the top
    }

    function hide() {
      index = -1;
      tip.hidden = true;
      cross.setAttribute("visibility", "hidden");
      dot.setAttribute("visibility", "hidden");
    }

    plot.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      show(nearest(((e.clientX - rect.left) / rect.width) * data.w));
    });
    plot.addEventListener("pointerleave", hide);
    window.addEventListener("resize", hide); // its position was computed for the old width
    plot.addEventListener("blur", hide);
    plot.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 10 : 1;
      const last = data.x.length - 1;
      const moves = {
        ArrowRight: index < 0 ? last : index + step,
        ArrowLeft: index < 0 ? last : index - step,
        Home: 0,
        End: last,
      };
      if (e.key in moves) { e.preventDefault(); show(moves[e.key]); }
      else if (e.key === "Escape") hide();
    });
  }

  // ------------------------------------------------------------------ builder rows
  document.addEventListener("click", (e) => {
    const button = e.target.closest("[data-remove-row]");
    if (!button) return;
    const row = button.closest(".cond-row");
    const form = button.closest("form");
    const rows = row && row.parentElement;
    if (!row) return;
    if (rows && rows.children.length === 1) {
      // keep one (empty) row so the group can still be filled in
      row.querySelector("select").selectedIndex = 0;
      row.querySelector("input").value = "";
    } else {
      row.remove();
    }
    if (form) form.dispatchEvent(new Event("change", { bubbles: true }));
  });

  // ------------------------------------------------------------------ errors from htmx
  function toast(text) {
    let el = document.querySelector(".toast");
    if (!el) {
      el = document.createElement("div");
      el.className = "toast";
      el.setAttribute("role", "alert");
      document.body.append(el);
    }
    el.textContent = text;
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.remove(), 6000);
  }

  document.addEventListener("htmx:responseError", (e) => {
    const xhr = e.detail.xhr;
    let detail = `${xhr.status} ${xhr.statusText}`;
    try {
      const body = JSON.parse(xhr.responseText);
      if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    toast(detail);
  });
  document.addEventListener("htmx:sendError", () => toast("The server did not answer. Is it running?"));

  function init(root) {
    root.querySelectorAll(".chart[data-points]").forEach((fig) => {
      if (fig.dataset.ready) return;
      fig.dataset.ready = "1";
      initChart(fig);
    });
  }

  document.addEventListener("DOMContentLoaded", () => { init(document); connect(); });
  document.addEventListener("htmx:afterSettle", (e) => init(e.target));
})();

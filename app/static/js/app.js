/* TerraLens front-end behaviour (vanilla JS, no dependencies). */
(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  function fmtSize(n) {
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    if (n < 1073741824) return (n / 1048576).toFixed(1) + " MB";
    return (n / 1073741824).toFixed(2) + " GB";
  }

  /* ---------- loading overlay + confirm on forms ---------- */
  function showLoading(text) {
    var el = $("#loading");
    if (!el) return;
    $("#loading-text").textContent = text || "Working...";
    el.classList.add("on");
  }
  $$("form[data-confirm]").forEach(function (f) {
    f.addEventListener("submit", function (e) { if (!window.confirm(f.getAttribute("data-confirm"))) e.preventDefault(); });
  });
  $$("form[data-loading]").forEach(function (f) {
    f.addEventListener("submit", function (e) {
      if (e.defaultPrevented) return;
      if (!f.checkValidity()) return;
      showLoading(f.getAttribute("data-loading"));
    });
  });

  /* ---------- upload page ---------- */
  var upForm = $("#upload-form");
  if (upForm) {
    var MAX_MB = 200, tagReset = {};
    ["t1", "t2"].forEach(function (tag) {
      var input = $("#" + tag + "_file"), drop = $('[data-drop="' + tag + '"]');
      var nameEl = $("#" + tag + "_name"), metaEl = $("#" + tag + "_meta"), thumb = $("#" + tag + "_thumb"), status = $("#" + tag + "_status");
      var thumbDefault = thumb.innerHTML, metaDefault = metaEl.textContent, nameDefault = nameEl.innerHTML;

      function reset() {
        input.value = ""; thumb.innerHTML = thumbDefault; metaEl.textContent = metaDefault;
        nameEl.innerHTML = nameDefault; status.style.display = "none";
      }
      function show(file) {
        if (!file) { reset(); return; }
        nameEl.textContent = file.name;
        metaEl.textContent = fmtSize(file.size);
        status.style.display = "inline-flex";
        var big = file.size > MAX_MB * 1048576;
        status.className = "pill " + (big ? "red" : "ok");
        status.textContent = big ? "Too large for the upload limit" : "Ready to upload";
        var ext = (file.name.split(".").pop() || "").toLowerCase();
        if (["png", "jpg", "jpeg"].indexOf(ext) >= 0) {
          var r = new FileReader();
          r.onload = function (ev) { thumb.innerHTML = ""; var im = new Image(); im.src = ev.target.result; im.alt = ""; thumb.appendChild(im); };
          r.readAsDataURL(file);
        } else {
          thumb.innerHTML = '<span style="font-size:13px;color:var(--muted);padding:6px;text-align:center">' + ext.toUpperCase() + "<br>preview after upload</span>";
        }
      }
      input.addEventListener("change", function () { show(input.files[0]); });
      ["dragenter", "dragover"].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
      ["dragleave", "drop"].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
      drop.addEventListener("drop", function (e) {
        if (e.dataTransfer && e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(input.files[0]); }
      });
      tagReset[tag] = reset;
    });
    $("#clear-btn").addEventListener("click", function () {
      ["t1", "t2"].forEach(function (t) { if (tagReset[t]) tagReset[t](); });
      $("#name").value = "";
    });
    // sensible defaults when the dataset changes
    var src = $("#source"), layout = $("#layout"), res = $("#res");
    src.addEventListener("change", function () {
      if (src.value === "LISS-IV") { layout.value = "gr_nir"; res.value = "5.8"; }
      else { layout.value = "rgb"; res.value = "10"; }
    });
    upForm.addEventListener("submit", function (e) {
      var total = 0;
      ["t1", "t2"].forEach(function (t) { var f = $("#" + t + "_file").files[0]; if (f) total += f.size; });
      if (total > MAX_MB * 1048576) {
        e.preventDefault(); e.stopImmediatePropagation();
        window.alert("The two files together are larger than " + MAX_MB + " MB. Please crop or resample them first.");
      }
    }, true);
  }

  /* ---------- zoom / fullscreen ---------- */
  var zoomState = new WeakMap();
  function zoomBy(container, factor) {
    $$("[data-zoom]", container).forEach(function (z) {
      var s = (zoomState.get(z) || 1) * factor;
      s = Math.max(1, Math.min(6, s));
      zoomState.set(z, s);
      z.style.transform = "scale(" + s + ")";
    });
  }
  $$("[data-act]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var container = btn.closest(".cmp-panel") || btn.closest(".card");
      var act = btn.getAttribute("data-act");
      if (act === "zoom-in") zoomBy(container, 1.35);
      else if (act === "zoom-out") zoomBy(container, 1 / 1.35);
      else if (act === "full") {
        var v = $(".cmp-single", container) && getComputedStyle($(".cmp-single", container)).display !== "none"
          ? $(".cmp-single .viewer", container) : $(".viewer", container);
        if (v && v.requestFullscreen) v.requestFullscreen();
      }
    });
  });

  /* ---------- compare modes ---------- */
  var cmp = $("#cmp");
  if (cmp) {
    var topImg = $("#cmp-top"), slider = $("#cmp-slider");
    function setMode(mode) {
      cmp.className = "cmp-wrap mode-" + mode;
      $$("#modes button").forEach(function (b) { b.classList.toggle("on", b.getAttribute("data-mode") === mode); });
      if (mode === "swipe") { topImg.style.opacity = 1; applySlider(); }
      if (mode === "overlay") { topImg.style.opacity = slider.value / 100; topImg.style.clipPath = "none"; }
    }
    function applySlider() { cmp.style.setProperty("--cut", slider.value + "%"); }
    slider.addEventListener("input", function () {
      if (cmp.classList.contains("mode-swipe")) applySlider();
      else topImg.style.opacity = slider.value / 100;
    });
    $$("#modes button").forEach(function (b) { b.addEventListener("click", function () { setMode(b.getAttribute("data-mode")); }); });
  }

  /* ---------- detect page: show one layer at a time ---------- */
  var stack = $("#det-stack");
  if (stack) {
    $$("#det-checks input").forEach(function (cb) {
      cb.addEventListener("change", function () {
        var which = cb.getAttribute("data-show");
        if (!cb.checked) { cb.checked = true; return; }               // keep one selected
        $$("#det-checks input").forEach(function (o) { if (o !== cb) o.checked = false; });
        $$("img[data-layer]", stack).forEach(function (im) { im.style.display = im.getAttribute("data-layer") === which ? "" : "none"; });
      });
    });
  }

  /* ---------- classify page: toggle class layers ---------- */
  $$("[data-cls-toggle]").forEach(function (cb) {
    cb.addEventListener("change", function () {
      var im = $('img[data-cls="' + cb.getAttribute("data-cls-toggle") + '"]');
      if (im) im.style.display = cb.checked ? "" : "none";
    });
  });
})();

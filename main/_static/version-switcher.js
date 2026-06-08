(function () {
  if (typeof defined_VERSIONS === "undefined") return;

  var versions = defined_VERSIONS;
  var currentVersion = typeof defined_VERSION_MATCH !== "undefined" ? defined_VERSION_MATCH : "";

  var wrapper = document.createElement("div");
  wrapper.className = "version-switcher";

  var btn = document.createElement("button");
  btn.className = "version-switcher__btn";
  btn.setAttribute("aria-expanded", "false");
  btn.textContent = currentVersion || "Version";

  var list = document.createElement("ul");
  list.className = "version-switcher__list";
  list.hidden = true;

  versions.forEach(function (v) {
    var li = document.createElement("li");
    var a = document.createElement("a");
    a.href = v.url;
    a.textContent = v.name;
    if (v.version === currentVersion) {
      a.setAttribute("aria-current", "true");
      btn.textContent = v.name;
    }
    li.appendChild(a);
    list.appendChild(li);
  });

  btn.addEventListener("click", function () {
    var show = list.hidden;
    list.hidden = !show;
    btn.setAttribute("aria-expanded", String(show));
  });

  document.addEventListener("click", function (e) {
    if (!wrapper.contains(e.target)) {
      list.hidden = true;
      btn.setAttribute("aria-expanded", "false");
    }
  });

  wrapper.appendChild(btn);
  wrapper.appendChild(list);

  var target = document.querySelector(".sy-head-extra");
  if (target) {
    target.insertBefore(wrapper, target.firstChild);
  }
})();

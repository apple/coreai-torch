// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-Clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


// Break sidebar entries at `.` boundaries so long dotted FQNs wrap cleanly
// (e.g. `coreai_opt.quantization.QuantizerConfig.presets.w4` wraps between
// segments rather than mid-word). Inserts a zero-width space (U+200B) after
// each `.`, which the browser treats as a valid line-break opportunity when
// combined with `white-space: normal` in the sidebar CSS.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".wy-menu-vertical li a").forEach(function (link) {
    // Only touch the text nodes — don't mangle any HTML inside (e.g. icons).
    link.childNodes.forEach(function (node) {
      if (node.nodeType === Node.TEXT_NODE && node.textContent.indexOf(".") !== -1) {
        node.textContent = node.textContent.replace(/\./g, ".​");
      }
    });
  });
});

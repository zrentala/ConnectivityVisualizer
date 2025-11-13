(function() {
  // Simple draggable vertical splitter between #left-panel and #right-panel
  function once(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn);
    } else {
      fn();
    }
  }

  once(function() {
    const splitter = document.getElementById('splitter');
    const left = document.getElementById('left-panel');
    const container = splitter ? splitter.parentElement : null;
    if (!splitter || !left || !container) return;

    let dragging = false;

    splitter.addEventListener('mousedown', function(e) {
      dragging = true;
      document.body.style.cursor = 'col-resize';
      e.preventDefault();
    });

    window.addEventListener('mouseup', function() {
      if (dragging) {
        dragging = false;
        document.body.style.cursor = '';
      }
    });

    window.addEventListener('mousemove', function(e) {
      if (!dragging) return;
      const rect = container.getBoundingClientRect();
      let offset = e.clientX - rect.left;
      let pct = (offset / rect.width) * 100;
      pct = Math.max(10, Math.min(90, pct));
      left.style.flexBasis = pct + '%';
    });

    // Touch support
    splitter.addEventListener('touchstart', function(e) {
      dragging = true;
      e.preventDefault();
    });
    window.addEventListener('touchend', function() { dragging = false; });
    window.addEventListener('touchmove', function(e) {
      if (!dragging) return;
      const touch = e.touches[0];
      const rect = container.getBoundingClientRect();
      let offset = touch.clientX - rect.left;
      let pct = (offset / rect.width) * 100;
      pct = Math.max(10, Math.min(90, pct));
      left.style.flexBasis = pct + '%';
    });
  });
})();

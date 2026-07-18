/* === SAKF-IK Research Showcase — Runtime JS === */
/* Scroll spy, smooth nav, counter animations, back-to-top, mobile menu */

(function() {
  'use strict';

  // --- DOM refs ---
  const nav = document.getElementById('top-nav');
  const navLinks = document.getElementById('nav-links');
  const navToggle = document.getElementById('nav-toggle');
  const progressBar = document.getElementById('progress-bar');
  const backToTop = document.getElementById('back-to-top');
  const allNavAnchors = document.querySelectorAll('.nav-links a');
  const allSections = document.querySelectorAll('.section, .hero');

  // --- Scroll spy: highlight current nav item ---
  function updateActiveNav() {
    const scrollY = window.scrollY + 120;
    let currentId = '';
    allSections.forEach(function(section) {
      const top = section.offsetTop;
      const bottom = top + section.offsetHeight;
      if (scrollY >= top && scrollY < bottom) {
        currentId = section.id;
      }
    });

    allNavAnchors.forEach(function(a) {
      a.classList.remove('active');
      if (a.getAttribute('href') === '#' + currentId) {
        a.classList.add('active');
      }
    });

    // If scrolled past the last section, activate the last nav item
    if (!currentId && window.scrollY + window.innerHeight >= document.body.offsetHeight - 50) {
      allNavAnchors.forEach(function(a) { a.classList.remove('active'); });
      const lastAnchor = allNavAnchors[allNavAnchors.length - 1];
      if (lastAnchor) lastAnchor.classList.add('active');
    }
  }

  // --- Progress bar ---
  function updateProgressBar() {
    const scrollTop = window.scrollY;
    const docHeight = document.body.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    progressBar.style.width = Math.min(progress, 100) + '%';
  }

  // --- Back to top button ---
  function updateBackToTop() {
    if (window.scrollY > 600) {
      backToTop.classList.add('visible');
    } else {
      backToTop.classList.remove('visible');
    }
  }

  // --- Nav shadow on scroll ---
  function updateNavShadow() {
    if (window.scrollY > 10) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
  }

  // --- Reveal content elements on scroll (Intersection Observer) ---
  function setupRevealObserver() {
    var revealTargets = document.querySelectorAll(
      '.card, figure, .callout, .code-block, .data-table, ' +
      '.arch-layer, .mem-level, .gpu-box, .step, .stat-item, .formula'
    );

    if (!('IntersectionObserver' in window)) {
      // Fallback: show all elements immediately
      revealTargets.forEach(function(el) { el.classList.add('visible'); });
      return;
    }

    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.05, rootMargin: '0px 0px -30px 0px' });

    revealTargets.forEach(function(el) {
      observer.observe(el);
    });

    // Also make hero stat items visible immediately
    var heroStats = document.querySelectorAll('.hero .stat-item');
    heroStats.forEach(function(el) { el.classList.add('visible'); });
  }

  // --- Counter animation ---
  function animateCounter(el) {
    const target = parseFloat(el.getAttribute('data-to'));
    const decimals = parseInt(el.getAttribute('data-decimals') || '0');
    const duration = 1500;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1.0);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = target * eased;

      if (decimals > 0) {
        el.textContent = current.toFixed(decimals);
      } else if (target >= 1000) {
        el.textContent = Math.floor(current).toLocaleString();
      } else {
        el.textContent = Math.floor(current).toString();
      }

      if (progress < 1.0) {
        requestAnimationFrame(update);
      } else {
        // Final value
        if (decimals > 0) {
          el.textContent = target.toFixed(decimals);
        } else if (target >= 1000) {
          el.textContent = Math.floor(target).toLocaleString();
        } else {
          el.textContent = Math.floor(target).toString();
        }
      }
    }

    requestAnimationFrame(update);
  }

  function setupCounterObserver() {
    if (!('IntersectionObserver' in window)) {
      document.querySelectorAll('.counter').forEach(function(el) { animateCounter(el); });
      return;
    }

    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });

    document.querySelectorAll('.counter').forEach(function(el) {
      observer.observe(el);
    });
  }

  // --- Smooth scroll for nav links ---
  allNavAnchors.forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const targetId = this.getAttribute('href').substring(1);
      const target = document.getElementById(targetId);
      if (target) {
        const navHeight = nav.offsetHeight;
        const targetPos = target.offsetTop - navHeight - 20;
        window.scrollTo({ top: targetPos, behavior: 'smooth' });
      }
      // Close mobile menu
      navLinks.classList.remove('open');
    });
  });

  // --- Mobile nav toggle ---
  navToggle.addEventListener('click', function() {
    navLinks.classList.toggle('open');
  });

  // Close mobile nav when clicking outside
  document.addEventListener('click', function(e) {
    if (!nav.contains(e.target)) {
      navLinks.classList.remove('open');
    }
  });

  // --- Back to top click ---
  backToTop.addEventListener('click', function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // --- KaTeX auto-render (if available) ---
  function renderMath() {
    if (typeof renderMathInElement !== 'undefined') {
      try {
        renderMathInElement(document.body, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false }
          ],
          throwOnError: false
        });
      } catch (e) {
        console.warn('KaTeX render error:', e.message);
      }
    }
  }

  // --- Combined scroll handler (throttled) ---
  let ticking = false;
  function onScroll() {
    if (!ticking) {
      requestAnimationFrame(function() {
        updateActiveNav();
        updateProgressBar();
        updateBackToTop();
        updateNavShadow();
        ticking = false;
      });
      ticking = true;
    }
  }

  // --- Init ---
  function init() {
    setupRevealObserver();
    setupCounterObserver();
    updateActiveNav();
    updateProgressBar();
    updateNavShadow();

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', updateActiveNav);

    // Try to render math after fonts load
    if (document.readyState === 'complete') {
      renderMath();
    } else {
      window.addEventListener('load', renderMath);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

document.addEventListener('DOMContentLoaded', () => {
  const homeRoot = document.querySelector('.gn-home');
  if (!homeRoot) return;

  const canvas = homeRoot.querySelector('.gn-web-canvas');
  const pageBody = document.body;

  if (canvas) {
    const context = canvas.getContext('2d');
    const points = [];
    let width = 0;
    let height = 0;
    let animationFrameId;
    let pulse = 0;
    let isAnimating = false;

    const colorPalettes = {
      dark: {
        gradientA: 'rgba(51, 217, 178, 0.08)',
        gradientB: 'rgba(9, 21, 30, 0.16)',
        gradientC: 'rgba(4, 8, 14, 0.55)',
        lineRgb: '75, 230, 191',
        lineAlphaMax: 0.24,
        nodeRgb: '110, 245, 211',
        nodeBaseAlpha: 0.45,
        nodePulse: 0.08,
      },
      light: {
        gradientA: 'rgba(21, 125, 148, 0.08)',
        gradientB: 'rgba(44, 96, 121, 0.06)',
        gradientC: 'rgba(7, 42, 61, 0.10)',
        lineRgb: '20, 108, 136',
        lineAlphaMax: 0.26,
        nodeRgb: '16, 120, 156',
        nodeBaseAlpha: 0.50,
        nodePulse: 0.10,
      },
    };

    const isLightScheme = () => {
      const root = document.documentElement;
      const body = document.body;
      const scheme =
        root?.getAttribute('data-md-color-scheme') ||
        body?.getAttribute('data-md-color-scheme') ||
        '';
      return scheme.toLowerCase() === 'default';
    };

    let activePalette = isLightScheme() ? colorPalettes.light : colorPalettes.dark;

    const updateThemePalette = () => {
      activePalette = isLightScheme() ? colorPalettes.light : colorPalettes.dark;
    };

    const createPoint = () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      radius: 0.6 + Math.random() * 1.8,
    });

    const setCanvasSize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);

      const density = Math.max(45, Math.floor((width * height) / 28000));
      points.length = 0;
      for (let i = 0; i < density; i += 1) {
        points.push(createPoint());
      }
    };

    const draw = () => {
      isAnimating = true;
      pulse += 0.01;
      context.clearRect(0, 0, width, height);

      const gradient = context.createRadialGradient(
        width * 0.72,
        height * 0.18,
        0,
        width * 0.5,
        height * 0.5,
        Math.max(width, height)
      );
      gradient.addColorStop(0, activePalette.gradientA);
      gradient.addColorStop(0.5, activePalette.gradientB);
      gradient.addColorStop(1, activePalette.gradientC);
      context.fillStyle = gradient;
      context.fillRect(0, 0, width, height);

      const maxDistance = Math.min(165, Math.max(120, width * 0.12));

      for (let i = 0; i < points.length; i += 1) {
        const pointA = points[i];
        pointA.x += pointA.vx;
        pointA.y += pointA.vy;

        if (pointA.x < -20 || pointA.x > width + 20) pointA.vx *= -1;
        if (pointA.y < -20 || pointA.y > height + 20) pointA.vy *= -1;

        for (let j = i + 1; j < points.length; j += 1) {
          const pointB = points[j];
          const dx = pointA.x - pointB.x;
          const dy = pointA.y - pointB.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < maxDistance) {
            const alpha = (1 - distance / maxDistance) * activePalette.lineAlphaMax;
            context.strokeStyle = `rgba(${activePalette.lineRgb}, ${alpha})`;
            context.lineWidth = 0.6;
            context.beginPath();
            context.moveTo(pointA.x, pointA.y);
            context.lineTo(pointB.x, pointB.y);
            context.stroke();
          }
        }

        context.fillStyle = `rgba(${activePalette.nodeRgb}, ${activePalette.nodeBaseAlpha + (Math.sin(pulse + i) + 1) * activePalette.nodePulse})`;
        context.beginPath();
        context.arc(pointA.x, pointA.y, pointA.radius, 0, Math.PI * 2);
        context.fill();
      }

      animationFrameId = window.requestAnimationFrame(draw);
    };

    setCanvasSize();
    draw();

    window.addEventListener('resize', setCanvasSize);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        if (animationFrameId) {
          window.cancelAnimationFrame(animationFrameId);
        }
        isAnimating = false;
      } else {
        if (!isAnimating) {
          draw();
        }
      }
    });

    const themeObserver = new MutationObserver(() => {
      updateThemePalette();
    });

    if (document.documentElement) {
      themeObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['data-md-color-scheme'],
      });
    }

    if (document.body) {
      themeObserver.observe(document.body, {
        attributes: true,
        attributeFilter: ['data-md-color-scheme'],
      });
    }

    if (pageBody) {
      pageBody.classList.add('gn-home-active');
    }
  }

  const revealNodes = homeRoot.querySelectorAll('.gn-reveal');
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15, rootMargin: '0px 0px -10% 0px' }
  );

  revealNodes.forEach((node, index) => {
    node.style.transitionDelay = `${Math.min(index * 70, 280)}ms`;
    revealObserver.observe(node);
  });

  const hero = homeRoot.querySelector('.gn-hero-bg');
  if (!hero) return;

  const updateParallax = () => {
    const y = Math.min(window.scrollY * 0.18, 80);
    hero.style.transform = `translateY(${y}px)`;
  };

  updateParallax();
  window.addEventListener('scroll', updateParallax, { passive: true });
});

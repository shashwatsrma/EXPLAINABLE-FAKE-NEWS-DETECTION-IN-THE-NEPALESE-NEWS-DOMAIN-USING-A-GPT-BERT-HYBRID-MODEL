// =====================================================
// DOM ELEMENTS
// =====================================================
const newsInput = document.getElementById('newsInput');
const analyzeBtn = document.querySelector('.analyze-btn');
const clearBtn = document.querySelector('.clear-btn');
const charCount = document.querySelector('.char-count');
const optionBtns = document.querySelectorAll('.option-btn');
const navLinks = document.querySelectorAll('.nav-link');

// =====================================================
// SECTION SWITCHING
// =====================================================
function showSection(sectionName) {
    document.querySelectorAll('.content-section').forEach(s => s.style.display = 'none');
    const analyzerSection = document.getElementById('analyzer-section');
    const heroSection = document.getElementById('home-section');
    if (sectionName === 'home') {
        if (heroSection) heroSection.style.display = 'flex';
        if (analyzerSection) analyzerSection.style.display = 'flex';
    } else {
        if (heroSection) heroSection.style.display = 'none';
        if (analyzerSection) analyzerSection.style.display = 'none';
        const target = document.getElementById(`${sectionName}-section`);
        if (target) { target.style.display = 'block'; target.classList.add('fade-in'); }
    }
    navLinks.forEach(link => link.classList.toggle('active', link.getAttribute('data-section') === sectionName));
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
window.showSection = showSection;
navLinks.forEach(link => { link.addEventListener('click', e => { e.preventDefault(); showSection(link.getAttribute('data-section')); }); });

function scrollToAnalyzer() {
    showSection('home');
    setTimeout(() => { document.getElementById('analyzer-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 100);
}
window.scrollToAnalyzer = scrollToAnalyzer;

// =====================================================
// CHARACTER COUNTER
// =====================================================
if (newsInput && charCount) {
    newsInput.addEventListener('input', () => {
        const len = newsInput.value.length;
        charCount.textContent = `${len} / 5000 characters`;
        charCount.style.color = len > 4500 ? 'var(--fake-color)' : len > 3500 ? 'var(--warning-color)' : 'var(--neutral-500)';
    });
}

// =====================================================
// CLEAR BUTTON
// =====================================================
if (clearBtn && newsInput) {
    clearBtn.addEventListener('click', () => {
        newsInput.value = '';
        if (charCount) { charCount.textContent = '0 / 5000 characters'; charCount.style.color = 'var(--neutral-500)'; }
        document.getElementById('result-section').style.display = 'none';
        document.getElementById('loading-section').style.display = 'none';
    });
}

// =====================================================
// INPUT TYPE TOGGLE
// =====================================================
optionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        optionBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        if (newsInput) {
            newsInput.placeholder = btn.dataset.type === 'url'
                ? 'Enter news article URL here (e.g., https://example.com/news-article)'
                : 'नेपाली समाचार यहाँ पेस्ट गर्नुहोस् / Paste Nepali news article here...';
            newsInput.rows = btn.dataset.type === 'url' ? 2 : 8;
        }
    });
});

// =====================================================
// LOADING ANIMATION
// =====================================================
let loadingInterval = null;

function showLoading() {
    document.getElementById('result-section').style.display = 'none';
    const loading = document.getElementById('loading-section');
    loading.style.display = 'block';
    loading.scrollIntoView({ behavior: 'smooth', block: 'start' });

    const steps = ['step-preprocess', 'step-bert', 'step-gpt2', 'step-classify', 'step-lime'];
    steps.forEach(id => document.getElementById(id).classList.remove('active', 'done'));

    let i = 0;
    loadingInterval = setInterval(() => {
        if (i > 0) document.getElementById(steps[i - 1]).classList.replace('active', 'done');
        if (i < steps.length) { document.getElementById(steps[i]).classList.add('active'); i++; }
        else clearInterval(loadingInterval);
    }, 2500);
}

function hideLoading() {
    if (loadingInterval) clearInterval(loadingInterval);
    document.getElementById('loading-section').style.display = 'none';
}

// =====================================================
// ANALYZE BUTTON — API CALL WITH LIME
// =====================================================
if (analyzeBtn && newsInput) {
    analyzeBtn.addEventListener('click', async () => {
        const inputText = newsInput.value.trim();
        if (!inputText) { showNotification('Please enter news text to analyze', 'warning'); return; }
        if (inputText.length < 20) { showNotification('Please enter at least 20 characters for accurate analysis', 'warning'); return; }

        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="spinner"></span> Analyzing with LIME...';

        showLoading();

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: inputText, explain: true })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error');

            hideLoading();
            resetAnalyzeButton();
            showResults(data);
        } catch (error) {
            console.error("Analysis error:", error);
            hideLoading();
            resetAnalyzeButton();
            showNotification("Failed to connect to the server. Is the backend running?", "warning");
        }
    });
}

function resetAnalyzeButton() {
    analyzeBtn.classList.remove('loading');
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
            <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm1 11H9V9h2v4zm0-6H9V5h2v2z"/>
        </svg> Analyze News`;
}

// =====================================================
// SHOW RESULTS — Dynamic LIME-powered display
// =====================================================
function showResults(data) {
    const section = document.getElementById('result-section');
    const isReal = data.prediction === 'Real';
    const confidence = data.confidence;

    section.className = 'card result-card ' + (isReal ? 'result-real' : 'result-fake');

    // Header
    const icon = document.getElementById('result-icon');
    icon.className = 'result-icon ' + (isReal ? 'real-icon' : 'fake-icon');
    icon.innerHTML = isReal
        ? '<svg width="60" height="60" viewBox="0 0 60 60" fill="none"><circle cx="30" cy="30" r="28" stroke="currentColor" stroke-width="3"/><path d="M18 30L26 38L42 22" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        : '<svg width="60" height="60" viewBox="0 0 60 60" fill="none"><circle cx="30" cy="30" r="28" stroke="currentColor" stroke-width="3"/><path d="M20 20L40 40M40 20L20 40" stroke="currentColor" stroke-width="4" stroke-linecap="round"/></svg>';

    document.getElementById('result-title').textContent = isReal ? 'Verified as Authentic' : 'Potentially Misleading';
    const label = document.getElementById('result-label');
    label.textContent = isReal ? 'REAL NEWS' : 'FAKE NEWS';
    label.className = 'result-label ' + (isReal ? 'real' : 'fake');

    // Confidence
    document.getElementById('confidence-value').textContent = `${confidence}%`;
    const fill = document.getElementById('confidence-fill');
    fill.className = 'confidence-fill ' + (isReal ? 'real-fill' : 'fake-fill');
    fill.style.width = '0%';
    setTimeout(() => fill.style.width = `${confidence}%`, 300);

    const desc = document.getElementById('confidence-description');
    desc.textContent = isReal
        ? 'High confidence indicates strong alignment with verified reporting patterns'
        : 'High confidence in fake classification — proceed with caution';
    desc.className = 'confidence-description' + (isReal ? '' : ' warning');

    // Probabilities
    if (data.probabilities) {
        document.getElementById('prob-real').textContent = `${data.probabilities.real}%`;
        document.getElementById('prob-fake').textContent = `${data.probabilities.fake}%`;
        setTimeout(() => {
            document.getElementById('prob-real-bar').style.width = `${data.probabilities.real}%`;
            document.getElementById('prob-fake-bar').style.width = `${data.probabilities.fake}%`;
        }, 400);
    }

    // LIME words
    renderLimeWords('lime-fake-words', data.top_fake_words || [], 'fake');
    renderLimeWords('lime-real-words', data.top_real_words || [], 'real');
    document.getElementById('lime-fake-section').style.display = data.top_fake_words?.length ? 'block' : 'none';
    document.getElementById('lime-real-section').style.display = data.top_real_words?.length ? 'block' : 'none';

    // Warning
    document.getElementById('warning-box').style.display = isReal ? 'none' : 'flex';

    // Show
    section.style.display = 'block';
    section.classList.add('slide-up');
    setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

// =====================================================
// RENDER LIME WORD CHIPS
// =====================================================
function renderLimeWords(containerId, words, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    if (!words || words.length === 0) {
        container.innerHTML = '<p class="no-words">No significant words detected</p>';
        return;
    }

    const maxWeight = Math.max(...words.map(w => Math.abs(w[1])));

    words.forEach(([word, weight]) => {
        const chip = document.createElement('div');
        chip.className = `lime-chip lime-chip-${type}`;
        const intensity = Math.max(0.4, Math.abs(weight) / maxWeight);
        chip.style.opacity = intensity;
        const barWidth = Math.max(15, (Math.abs(weight) / maxWeight) * 100);

        chip.innerHTML = `
            <span class="lime-word">${word}</span>
            <div class="lime-bar-container">
                <div class="lime-bar lime-bar-${type}" style="width: ${barWidth}%"></div>
            </div>
            <span class="lime-weight">${(Math.abs(weight) * 100).toFixed(1)}%</span>
        `;
        container.appendChild(chip);
    });
}

// =====================================================
// ANALYZE ANOTHER
// =====================================================
document.addEventListener('click', (e) => {
    if (e.target.id === 'analyze-another-btn' ||
        (e.target.classList.contains('primary-btn-outline') && e.target.textContent.includes('Analyze Another'))) {
        document.querySelector('.input-card')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        if (newsInput) { newsInput.value = ''; }
        if (charCount) { charCount.textContent = '0 / 5000 characters'; charCount.style.color = 'var(--neutral-500)'; }
        document.getElementById('result-section').style.display = 'none';
    }
});

// =====================================================
// NOTIFICATION SYSTEM
// =====================================================
function showNotification(message, type = 'info') {
    const n = document.createElement('div');
    n.className = `notification notification-${type}`;
    n.innerHTML = `<div class="notification-content">
        <span class="notification-icon">${type === 'warning' ? '⚠️' : type === 'success' ? '✓' : 'ℹ'}</span>
        <span class="notification-message">${message}</span>
    </div>`;

    if (!document.getElementById('notification-styles')) {
        const s = document.createElement('style');
        s.id = 'notification-styles';
        s.textContent = `.notification{position:fixed;top:100px;right:20px;background:#fff;padding:1rem 1.5rem;border-radius:.75rem;box-shadow:0 10px 15px -3px rgba(0,0,0,.1);z-index:10000;animation:slideInRight .3s ease-out;max-width:400px;border-left:4px solid}.notification-info{border-color:#6366f1}.notification-warning{border-color:#f59e0b}.notification-success{border-color:#10b981}.notification-content{display:flex;align-items:center;gap:.75rem}.notification-icon{font-size:1.5rem}.notification-message{color:#27272a;font-weight:600}@keyframes slideInRight{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}`;
        document.head.appendChild(s);
    }
    document.body.appendChild(n);
    setTimeout(() => { n.style.animation = 'slideInRight .3s ease-out reverse'; setTimeout(() => n.remove(), 300); }, 4000);
}

// =====================================================
// KEYBOARD SHORTCUTS
// =====================================================
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && document.activeElement === newsInput) analyzeBtn?.click();
    if (e.key === 'Escape') {
        const active = Array.from(document.querySelectorAll('.content-section')).filter(s => s.style.display !== 'none');
        if (active.length > 0) showSection('home');
        else if (document.activeElement === newsInput) clearBtn?.click();
    }
});

// =====================================================
// TEXTAREA AUTO-RESIZE
// =====================================================
if (newsInput) {
    newsInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 400) + 'px';
    });
}

// =====================================================
// INITIALIZE
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('loading-section').style.display = 'none';
    document.querySelectorAll('.content-section').forEach(s => s.style.display = 'none');
    showSection('home');
    document.documentElement.style.scrollBehavior = 'smooth';
});

// =====================================================
// LAZY LOAD ANIMATIONS
// =====================================================
if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('fade-in'); obs.unobserve(e.target); } });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    document.querySelectorAll('.about-card, .timeline-item, .metric-card, .team-member').forEach(el => obs.observe(el));
}

// =====================================================
// BROWSER HISTORY
// =====================================================
window.addEventListener('popstate', (e) => showSection(e.state?.section || 'home'));
const _showSection = showSection;
window.showSection = function (name) { _showSection(name); history.pushState({ section: name }, '', `#${name}`); };

// =====================================================
// CONSOLE
// =====================================================
console.log('%c🚀 FactCheck Nepal — GBERT + LIME', 'color:#6366f1;font-size:20px;font-weight:bold');
console.log('%cBERT + GPT-2 Fusion | LIME Explainability', 'color:#10b981;font-size:14px');

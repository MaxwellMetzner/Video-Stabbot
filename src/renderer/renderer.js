// ============================================================
// Video Stabbot — Renderer
// ============================================================

// State
let system = null;
let filePath = null;
let videoInfo = null;
let outputPath = null;
let elapsedTimer = null;
let processingStart = null;
let scriptsPath = null;

// Views
const VIEWS = ['loading', 'error', 'select', 'quality', 'custom', 'subject', 'panorama', 'processing', 'complete'];

function showView(name) {
    VIEWS.forEach(v => {
        const el = document.getElementById(`view-${v}`);
        if (el) el.classList.toggle('active', v === name);
    });
}

// ============================================================
// Formatters
// ============================================================

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDuration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${m}:${String(s).padStart(2, '0')}`;
}

function formatTimeStat(seconds) {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(0);
    return `${m}m ${s}s`;
}

// ============================================================
// Toast
// ============================================================

let toastTimeout = null;

function showToast(message, duration = 4000) {
    const toast = document.getElementById('toast');
    document.getElementById('toast-message').textContent = message;
    if (toastTimeout) clearTimeout(toastTimeout);
    toast.classList.add('show');
    toastTimeout = setTimeout(() => toast.classList.remove('show'), duration);
}

// ============================================================
// Encoder Badge
// ============================================================

function updateBadges(encoder, hasPython) {
    const badge = document.getElementById('encoder-badge');
    const nameEl = document.getElementById('encoder-name');
    nameEl.textContent = encoder.name;
    badge.classList.remove('hidden');
    if (encoder.type === 'software') badge.classList.add('software');

    // Python availability affects Subject Lock and Panorama cards
    const subjectBtn = document.getElementById('btn-subject');
    const panoramaBtn = document.getElementById('btn-panorama');
    if (!hasPython) {
        subjectBtn.classList.add('disabled');
        subjectBtn.title = 'Requires Python 3.8+ with opencv-python and numpy';
        panoramaBtn.classList.add('disabled');
        panoramaBtn.title = 'Requires Python 3.8+ with opencv-python and numpy';
    }
}

// ============================================================
// File Handling
// ============================================================

const SUPPORTED_EXT = new Set([
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v',
]);

function isVideoFile(path) {
    const ext = '.' + path.split('.').pop().toLowerCase();
    return SUPPORTED_EXT.has(ext);
}

async function handleFile(path) {
    if (!isVideoFile(path)) {
        showToast('Unsupported file format. Please select a video file.');
        return;
    }
    filePath = path;
    try {
        videoInfo = await window.stabbot.getVideoInfo({
            ffprobe: system.ffprobe,
            filePath: filePath,
        });
        updateFileInfo();
        showView('quality');
    } catch (err) {
        showToast(`Could not read video: ${err.message}`);
    }
}

function updateFileInfo() {
    const name = filePath.split(/[\\/]/).pop();
    document.getElementById('file-name').textContent = name;

    const parts = [];
    if (videoInfo.width && videoInfo.height) parts.push(`${videoInfo.width}×${videoInfo.height}`);
    if (videoInfo.fps) parts.push(`${videoInfo.fps} fps`);
    if (videoInfo.duration) parts.push(formatDuration(videoInfo.duration));
    if (videoInfo.size) parts.push(formatSize(videoInfo.size));
    document.getElementById('file-meta').textContent = parts.join('  •  ');
}

// ============================================================
// Quality Selection (Vidstab modes)
// ============================================================

async function handleQualityChoice(mode, customSettings) {
    const savePath = await window.stabbot.selectSavePath(filePath);
    if (!savePath) return;

    outputPath = savePath;
    showView('processing');
    resetProcessingView();
    startElapsed();

    window.stabbot.onProgress(data => updateProgress(data));

    try {
        const opts = {
            ffmpeg: system.ffmpeg,
            input: filePath,
            output: outputPath,
            mode: mode,
            encoder: system.encoder,
            duration: videoInfo.duration,
        };
        if (mode === 'custom' && customSettings) {
            opts.customSettings = customSettings;
        }
        const result = await window.stabbot.startProcessing(opts);
        window.stabbot.removeProgressListener();
        stopElapsed();
        showComplete(result);
    } catch (err) {
        window.stabbot.removeProgressListener();
        stopElapsed();
        if (err.message === 'CANCELLED') {
            showView('quality');
        } else {
            showProcessingError(err.message);
        }
    }
}

// ============================================================
// Subject Lock Processing
// ============================================================

async function handleSubjectLock() {
    if (!system.hasPythonDeps) {
        showToast('Python + OpenCV required. Install: pip install opencv-python numpy');
        return;
    }

    const savePath = await window.stabbot.selectSavePath(filePath);
    if (!savePath) return;

    outputPath = savePath;
    const trackMode = getRadioValue('opt-track-mode');
    const smooth = document.getElementById('opt-subject-smooth').value;
    const scale = document.getElementById('opt-subject-scale').value;
    const resolution = document.getElementById('opt-subject-resolution').value;

    showView('processing');
    resetProcessingView('Tracking subject\u2026');
    startElapsed();
    window.stabbot.onProgress(data => updateProgress(data));

    try {
        const scriptFile = scriptsPath + (navigator.platform.includes('Win') ? '\\' : '/') + 'reframe.py';
        const args = [
            '--input', filePath,
            '--output', outputPath,
            '--mode', trackMode,
            '--smoothing', smooth,
            '--scale', scale,
            '--resolution', resolution,
            '--ffmpeg', system.ffmpeg,
        ];
        const result = await window.stabbot.runPythonScript({
            python: system.python,
            script: scriptFile,
            args: args,
            duration: videoInfo.duration,
        });
        window.stabbot.removeProgressListener();
        stopElapsed();
        showComplete(result);
    } catch (err) {
        window.stabbot.removeProgressListener();
        stopElapsed();
        if (err.message === 'CANCELLED') {
            showView('subject');
        } else {
            showProcessingError(err.message);
        }
    }
}

// ============================================================
// Panorama Processing
// ============================================================

async function handlePanorama() {
    if (!system.hasPythonDeps) {
        showToast('Python + OpenCV required. Install: pip install opencv-python numpy');
        return;
    }

    const outputType = getRadioValue('opt-pano-output');
    let savePath;
    if (outputType === 'image') {
        savePath = await window.stabbot.selectSaveImagePath(filePath);
    } else {
        savePath = await window.stabbot.selectSavePath(filePath);
    }
    if (!savePath) return;

    outputPath = savePath;
    const detector = document.getElementById('opt-pano-detector').value;
    const smooth = document.getElementById('opt-pano-smooth').value;
    const blend = document.getElementById('opt-pano-blend').value;

    showView('processing');
    resetProcessingView('Building panorama\u2026');
    startElapsed();
    window.stabbot.onProgress(data => updateProgress(data));

    try {
        const scriptFile = scriptsPath + (navigator.platform.includes('Win') ? '\\' : '/') + 'mosaic.py';
        const args = [
            '--input', filePath,
            '--output', outputPath,
            '--type', outputType,
            '--detector', detector,
            '--smoothing', smooth,
            '--blend', blend,
            '--ffmpeg', system.ffmpeg,
        ];
        const result = await window.stabbot.runPythonScript({
            python: system.python,
            script: scriptFile,
            args: args,
            duration: videoInfo.duration,
        });
        window.stabbot.removeProgressListener();
        stopElapsed();
        showComplete(result);
    } catch (err) {
        window.stabbot.removeProgressListener();
        stopElapsed();
        if (err.message === 'CANCELLED') {
            showView('panorama');
        } else {
            showProcessingError(err.message);
        }
    }
}

// ============================================================
// Progress
// ============================================================

function resetProcessingView(label) {
    document.getElementById('processing-content').classList.remove('hidden');
    document.getElementById('processing-error').classList.add('hidden');
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-percent').textContent = '0%';
    document.getElementById('progress-elapsed').textContent = '0:00';
    document.getElementById('phase-label').textContent = label || 'Analyzing motion\u2026';
}

function updateProgress(data) {
    const overall = Math.round(data.overall || 0);
    document.getElementById('progress-bar').style.width = `${overall}%`;
    document.getElementById('progress-percent').textContent = `${overall}%`;

    const labels = {
        detect: 'Analyzing motion\u2026',
        transform: 'Stabilizing video\u2026',
        tracking: 'Tracking subject\u2026',
        reframing: 'Reframing video\u2026',
        features: 'Extracting features\u2026',
        homography: 'Computing transforms\u2026',
        compositing: 'Compositing canvas\u2026',
        encoding: 'Encoding output\u2026',
        processing: 'Processing\u2026',
    };
    if (data.phase && labels[data.phase]) {
        document.getElementById('phase-label').textContent = labels[data.phase];
    }
}

function startElapsed() {
    processingStart = Date.now();
    const el = document.getElementById('progress-elapsed');
    elapsedTimer = setInterval(() => {
        const sec = (Date.now() - processingStart) / 1000;
        el.textContent = formatDuration(sec);
    }, 1000);
}

function stopElapsed() {
    if (elapsedTimer) clearInterval(elapsedTimer);
    elapsedTimer = null;
}

function showProcessingError(message) {
    document.getElementById('processing-content').classList.add('hidden');
    document.getElementById('processing-error').classList.remove('hidden');
    document.getElementById('error-detail').textContent = message;
}

// ============================================================
// Complete
// ============================================================

function showComplete(result) {
    document.getElementById('stat-detect').textContent = formatTimeStat(result.detectTime || 0);
    document.getElementById('stat-transform').textContent = formatTimeStat(result.transformTime || 0);
    document.getElementById('stat-total').textContent = formatTimeStat(result.totalTime || 0);
    document.getElementById('stat-original-size').textContent = formatSize(videoInfo.size);
    document.getElementById('stat-output-size').textContent = formatSize(result.outputSize || 0);
    showView('complete');
}

// ============================================================
// Custom Settings Helpers
// ============================================================

function getRadioValue(groupId) {
    const group = document.getElementById(groupId);
    const active = group.querySelector('.radio-option.active');
    return active ? active.dataset.value : null;
}

function initRadioGroup(groupId) {
    const group = document.getElementById(groupId);
    if (!group) return;
    const options = group.querySelectorAll('.radio-option');
    options.forEach(opt => {
        opt.addEventListener('click', () => {
            options.forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
        });
    });
}

function initSlider(sliderId, valueId, formatter) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(valueId);
    if (!slider || !display) return;
    const update = () => {
        display.textContent = formatter ? formatter(slider.value) : slider.value;
    };
    slider.addEventListener('input', update);
    update();
}

function initToggle(toggleId, labelId) {
    const toggle = document.getElementById(toggleId);
    const label = document.getElementById(labelId);
    if (!toggle || !label) return;
    toggle.addEventListener('change', () => {
        label.textContent = toggle.checked ? 'On' : 'Off';
    });
}

function gatherCustomSettings() {
    return {
        crop: getRadioValue('opt-border'),
        smoothing: parseInt(document.getElementById('opt-smoothing').value),
        shakiness: parseInt(document.getElementById('opt-shakiness').value),
        accuracy: parseInt(document.getElementById('opt-accuracy').value),
        optzoom: parseInt(getRadioValue('opt-optzoom')),
        zoom: parseInt(document.getElementById('opt-zoom').value),
        interpol: document.getElementById('opt-interpol').value,
        encoding: document.getElementById('opt-encoding').value,
        tripod: document.getElementById('opt-tripod').checked,
        maxshift: parseInt(document.getElementById('opt-maxshift').value),
        maxangle: parseFloat(document.getElementById('opt-maxangle').value),
        stepsize: parseInt(document.getElementById('opt-stepsize').value),
        mincontrast: parseFloat(document.getElementById('opt-mincontrast').value),
    };
}

// ============================================================
// Reset
// ============================================================

function resetToFileSelect() {
    filePath = null;
    videoInfo = null;
    outputPath = null;
    window.stabbot.removeProgressListener();
    stopElapsed();
    showView('select');
}

// ============================================================
// Event Listeners
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    // Prevent default drag on entire document
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', e => e.preventDefault());

    // Drop zone
    const dropZone = document.getElementById('drop-zone');

    dropZone.addEventListener('click', async () => {
        const selected = await window.stabbot.selectFile();
        if (selected) await handleFile(selected);
    });

    dropZone.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            const selected = await window.stabbot.selectFile();
            if (selected) await handleFile(selected);
        }
    });

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', async e => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.path) {
            await handleFile(file.path);
        }
    });

    // Quality cards
    document.getElementById('btn-quick').addEventListener('click', () => handleQualityChoice('quick'));
    document.getElementById('btn-hq').addEventListener('click', () => handleQualityChoice('highQuality'));
    document.getElementById('btn-custom').addEventListener('click', () => showView('custom'));

    // Advanced mode cards
    document.getElementById('btn-subject').addEventListener('click', () => {
        if (!system.hasPythonDeps) {
            showToast('Requires Python 3.8+ with: pip install opencv-python numpy');
            return;
        }
        showView('subject');
    });
    document.getElementById('btn-panorama').addEventListener('click', () => {
        if (!system.hasPythonDeps) {
            showToast('Requires Python 3.8+ with: pip install opencv-python numpy');
            return;
        }
        showView('panorama');
    });

    // Custom settings
    initRadioGroup('opt-border');
    initRadioGroup('opt-optzoom');
    initSlider('opt-smoothing', 'val-smoothing');
    initSlider('opt-shakiness', 'val-shakiness');
    initSlider('opt-accuracy', 'val-accuracy');
    initSlider('opt-zoom', 'val-zoom', v => `${v}%`);
    initSlider('opt-maxshift', 'val-maxshift', v => v === '0' ? 'unlimited' : `${v} px`);
    initSlider('opt-maxangle', 'val-maxangle', v => v === '0' ? 'unlimited' : `${v}\u00b0`);
    initSlider('opt-stepsize', 'val-stepsize');
    initSlider('opt-mincontrast', 'val-mincontrast');
    initToggle('opt-tripod', 'lbl-tripod');

    document.getElementById('btn-custom-back').addEventListener('click', () => showView('quality'));
    document.getElementById('btn-custom-start').addEventListener('click', () => {
        const settings = gatherCustomSettings();
        handleQualityChoice('custom', settings);
    });

    // Subject lock settings
    initRadioGroup('opt-track-mode');
    initSlider('opt-subject-smooth', 'val-subject-smooth');
    initSlider('opt-subject-scale', 'val-subject-scale', v => `${v}%`);
    document.getElementById('btn-subject-back').addEventListener('click', () => showView('quality'));
    document.getElementById('btn-subject-start').addEventListener('click', handleSubjectLock);

    // Panorama settings
    initRadioGroup('opt-pano-output');
    initSlider('opt-pano-smooth', 'val-pano-smooth');
    document.getElementById('btn-pano-back').addEventListener('click', () => showView('quality'));
    document.getElementById('btn-pano-start').addEventListener('click', handlePanorama);

    // Back button
    document.getElementById('btn-back').addEventListener('click', resetToFileSelect);

    // Cancel processing
    document.getElementById('btn-cancel').addEventListener('click', async () => {
        await window.stabbot.cancelProcessing();
    });

    // Error actions
    document.getElementById('btn-error-back').addEventListener('click', () => {
        showView('quality');
    });

    document.getElementById('btn-copy-error').addEventListener('click', () => {
        const text = document.getElementById('error-detail').textContent;
        navigator.clipboard.writeText(text).then(() => {
            const btn = document.getElementById('btn-copy-error');
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy Error'; }, 2000);
        });
    });

    // Complete actions
    document.getElementById('btn-show-folder').addEventListener('click', () => {
        if (outputPath) window.stabbot.showInFolder(outputPath);
    });

    document.getElementById('btn-another').addEventListener('click', resetToFileSelect);

    // Init
    init();
});

// ============================================================
// Init
// ============================================================

async function init() {
    showView('loading');
    try {
        system = await window.stabbot.detectSystem();
        scriptsPath = await window.stabbot.getScriptsPath();
        if (!system.ok) {
            document.getElementById('error-message').textContent = system.error;
            showView('error');
            return;
        }
        updateBadges(system.encoder, system.hasPythonDeps);
        showView('select');
    } catch (err) {
        document.getElementById('error-message').textContent = `Startup error: ${err.message}`;
        showView('error');
    }
}

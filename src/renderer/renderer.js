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

// Views
const VIEWS = ['loading', 'error', 'select', 'quality', 'custom', 'opencv', 'raft', 'processing', 'complete'];

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

function updateBadges(system) {
    const badge = document.getElementById('encoder-badge');
    const nameEl = document.getElementById('encoder-name');
    nameEl.textContent = system.encoder.name;
    badge.classList.remove('hidden');
    if (system.encoder.type === 'software') badge.classList.add('software');

    // Show/hide advanced mode cards based on dependencies
    const opencvBtn = document.getElementById('btn-opencv');
    const raftBtn = document.getElementById('btn-raft');

    // OpenCV requires scipy
    if (!system.hasScipy) {
        opencvBtn.classList.add('disabled');
        opencvBtn.title = 'Requires Python + SciPy\nInstall with: pip install scipy';
        opencvBtn.style.pointerEvents = 'none';
        opencvBtn.style.opacity = '0.5';
    } else {
        opencvBtn.classList.remove('disabled');
        opencvBtn.title = '';
        opencvBtn.style.pointerEvents = 'auto';
        opencvBtn.style.opacity = '1';
    }

    // RAFT requires PyTorch + torchvision + scipy + cv2 + numpy
    if (!system.raftReady) {
        raftBtn.classList.add('disabled');
        const reasons = system.raftMissing && system.raftMissing.length > 0
            ? `Missing: ${system.raftMissing.join(', ')}\nInstall with: pip install torch torchvision opencv-python numpy scipy`
            : 'Requires Python + PyTorch + torchvision + SciPy';
        raftBtn.title = reasons;
        raftBtn.style.pointerEvents = 'none';
        raftBtn.style.opacity = '0.5';
        raftBtn.style.filter = 'grayscale(100%)';
    } else {
        raftBtn.classList.remove('disabled');
        raftBtn.title = '';
        raftBtn.style.pointerEvents = 'auto';
        raftBtn.style.opacity = '1';
        raftBtn.style.filter = 'none';
    }
}

// ============================================================
// RAFT Deep Learning Processing
// ============================================================

async function handleRAFT() {
    const settings = {
        raftModel: getRadioValue('opt-raft-model'),
        maxIterations: parseInt(document.getElementById('val-raft-iterations').value),
        smoothingMethod: document.getElementById('opt-raft-smoothing-method').value,
        smoothingStrength: parseInt(document.getElementById('val-raft-smoothing').value),
        cropPercent: parseInt(document.getElementById('val-raft-crop').value),
        resolution: document.getElementById('opt-raft-resolution').value,
    };

    const savePath = await window.stabbot.selectSavePath(filePath);
    if (!savePath) return;

    outputPath = savePath;
    showView('processing');
    resetProcessingView('Loading RAFT model\u2026');
    startElapsed();

    window.stabbot.onProgress(data => updateProgress(data));

    try {
        const result = await window.stabbot.runPythonScript({
            scriptName: 'raft_dense_motion.py',
            args: [
                '--input', filePath,
                '--output', outputPath,
                '--ffmpeg', system.ffmpeg,
                '--raft-model', settings.raftModel,
                '--max-iterations', settings.maxIterations.toString(),
                '--smoothing-method', settings.smoothingMethod,
                '--smoothing-strength', settings.smoothingStrength.toString(),
                '--crop-percent', settings.cropPercent.toString(),
                '--resolution', settings.resolution,
            ],
            duration: videoInfo.duration,
        });

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
// OpenCV Feature Tracking Processing
// ============================================================

async function handleOpenCV() {
    const settings = {
        detector: getRadioValue('opt-opencv-detector'),
        maxFeatures: parseInt(document.getElementById('val-opencv-features').value),
        transformType: getRadioValue('opt-opencv-transform'),
        smoothingMethod: document.getElementById('opt-opencv-smoothing-method').value,
        smoothingStrength: parseInt(document.getElementById('val-opencv-smoothing').value),
        cropPercent: parseInt(document.getElementById('val-opencv-crop').value),
        resolution: document.getElementById('opt-opencv-resolution').value,
    };

    const savePath = await window.stabbot.selectSavePath(filePath);
    if (!savePath) return;

    outputPath = savePath;
    showView('processing');
    resetProcessingView('Detecting features\u2026');
    startElapsed();

    window.stabbot.onProgress(data => updateProgress(data));

    try {
        const result = await window.stabbot.runPythonScript({
            scriptName: 'opencv_feature_tracking.py',
            args: [
                '--input', filePath,
                '--output', outputPath,
                '--ffmpeg', system.ffmpeg,
                '--detector', settings.detector,
                '--max-features', settings.maxFeatures.toString(),
                '--transform-type', settings.transformType,
                '--smoothing-method', settings.smoothingMethod,
                '--smoothing-strength', settings.smoothingStrength.toString(),
                '--crop-percent', settings.cropPercent.toString(),
                '--resolution', settings.resolution,
            ],
            duration: videoInfo.duration,
        });
        window.stabbot.removeProgressListener();
        stopElapsed();
        showComplete(result);
    } catch (err) {
        window.stabbot.removeProgressListener();
        stopElapsed();
        if (err.message === 'CANCELLED') {
            showView('opencv');
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
        features: 'Detecting features\u2026',
        trajectory: 'Smoothing trajectory\u2026',
        loading: 'Loading AI model\u2026',
        flow: 'Estimating optical flow\u2026',
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
    const valueInput = document.getElementById(valueId);
    if (!slider || !valueInput) return;

    // Check if valueInput is a number input or span
    const isNumberInput = valueInput.tagName === 'INPUT';

    // Slider -> Value display/input
    const updateFromSlider = () => {
        const value = slider.value;
        if (isNumberInput) {
            valueInput.value = formatter ? formatter(value) : value;
        } else {
            valueInput.textContent = formatter ? formatter(value) : value;
        }
    };

    // Input -> Slider (only for number inputs)
    if (isNumberInput) {
        valueInput.addEventListener('input', () => {
            let val = parseFloat(valueInput.value);
            const min = parseFloat(valueInput.min);
            const max = parseFloat(valueInput.max);

            if (!isNaN(val)) {
                // Clamp value to min/max
                val = Math.max(min, Math.min(max, val));
                slider.value = val;
                valueInput.value = val;
            }
        });

        // Also sync on blur to clean up invalid input
        valueInput.addEventListener('blur', () => {
            let val = parseFloat(valueInput.value);
            const min = parseFloat(valueInput.min);
            const max = parseFloat(valueInput.max);

            if (isNaN(val)) {
                val = parseFloat(slider.value);
            } else {
                val = Math.max(min, Math.min(max, val));
            }
            valueInput.value = val;
            slider.value = val;
        });
    }

    slider.addEventListener('input', updateFromSlider);
    updateFromSlider();
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
        smoothing: parseInt(document.getElementById('val-smoothing').value),
        shakiness: parseInt(document.getElementById('val-shakiness').value),
        accuracy: parseInt(document.getElementById('val-accuracy').value),
        optzoom: parseInt(getRadioValue('opt-optzoom')),
        zoom: parseInt(document.getElementById('val-zoom').value),
        interpol: document.getElementById('opt-interpol').value,
        encoding: document.getElementById('opt-encoding').value,
        tripod: document.getElementById('opt-tripod').checked,
        relative: document.getElementById('opt-relative').checked,
        zoomspeed: parseFloat(document.getElementById('val-zoomspeed').value),
        maxshift: parseInt(document.getElementById('val-maxshift').value),
        maxangle: parseFloat(document.getElementById('val-maxangle').value),
        stepsize: parseInt(document.getElementById('val-stepsize').value),
        mincontrast: parseFloat(document.getElementById('val-mincontrast').value),
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
    document.getElementById('btn-hq').addEventListener('click', () => handleQualityChoice('highQuality'));
    document.getElementById('btn-custom').addEventListener('click', () => showView('custom'));
    document.getElementById('btn-opencv').addEventListener('click', () => showView('opencv'));

    // Custom settings
    initRadioGroup('opt-border');
    initRadioGroup('opt-optzoom');
    initSlider('opt-smoothing', 'val-smoothing');
    initSlider('opt-shakiness', 'val-shakiness');
    initSlider('opt-accuracy', 'val-accuracy');
    initSlider('opt-zoom', 'val-zoom');
    initSlider('opt-zoomspeed', 'val-zoomspeed');
    initSlider('opt-maxshift', 'val-maxshift');
    initSlider('opt-maxangle', 'val-maxangle');
    initSlider('opt-stepsize', 'val-stepsize');
    initSlider('opt-mincontrast', 'val-mincontrast');
    initToggle('opt-tripod', 'lbl-tripod');
    initToggle('opt-relative', 'lbl-relative');

    document.getElementById('btn-custom-back').addEventListener('click', () => showView('quality'));
    document.getElementById('btn-custom-start').addEventListener('click', () => {
        const settings = gatherCustomSettings();
        handleQualityChoice('custom', settings);
    });

    // OpenCV settings
    initRadioGroup('opt-opencv-detector');
    initRadioGroup('opt-opencv-transform');
    initSlider('opt-opencv-features', 'val-opencv-features');
    initSlider('opt-opencv-smoothing', 'val-opencv-smoothing');
    initSlider('opt-opencv-crop', 'val-opencv-crop');
    document.getElementById('btn-opencv-back').addEventListener('click', () => showView('quality'));
    document.getElementById('btn-opencv-start').addEventListener('click', handleOpenCV);

    // RAFT settings
    initRadioGroup('opt-raft-model');
    initSlider('opt-raft-iterations', 'val-raft-iterations');
    initSlider('opt-raft-smoothing', 'val-raft-smoothing');
    initSlider('opt-raft-crop', 'val-raft-crop');
    document.getElementById('btn-raft').addEventListener('click', () => {
        if (!document.getElementById('btn-raft').classList.contains('disabled')) {
            showView('raft');
        }
    });
    document.getElementById('btn-raft-back').addEventListener('click', () => showView('quality'));
    document.getElementById('btn-raft-start').addEventListener('click', handleRAFT);

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

    // Listen for step-by-step loading status from main process
    const stepOrder = ['ffmpeg', 'encoder', 'python', 'packages'];
    const progressMap = { ffmpeg: 25, encoder: 50, python: 75, packages: 100 };

    window.stabbot.onLoadingStatus(({ step, status, message }) => {
        // Update the status text
        const statusEl = document.getElementById('loading-status');
        if (statusEl) statusEl.textContent = message;

        // Update the step indicator
        const stepEl = document.getElementById(`step-${step}`);
        if (stepEl) {
            stepEl.className = `loading-step ${status}`;
        }

        // Update progress bar
        if (status === 'done' || status === 'skipped' || status === 'error') {
            const fill = document.getElementById('loading-progress-fill');
            if (fill) fill.style.width = `${progressMap[step] || 0}%`;
        } else if (status === 'active') {
            // Show partial progress for active step
            const idx = stepOrder.indexOf(step);
            const prev = idx > 0 ? progressMap[stepOrder[idx - 1]] : 0;
            const fill = document.getElementById('loading-progress-fill');
            if (fill) fill.style.width = `${prev + 5}%`;
        }
    });

    try {
        system = await window.stabbot.detectSystem();
        window.stabbot.removeLoadingStatusListener();
        if (!system.ok) {
            document.getElementById('error-message').textContent = system.error;
            showView('error');
            return;
        }
        updateBadges(system);
        showView('select');
    } catch (err) {
        window.stabbot.removeLoadingStatusListener();
        document.getElementById('error-message').textContent = `Startup error: ${err.message}`;
        showView('error');
    }
}

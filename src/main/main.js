const { app, BrowserWindow, ipcMain, dialog, shell, Menu } = require('electron');
const { spawn, execSync, execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Set to true to enable verbose Python script logging
const DEBUG = false;

let mainWindow;
let currentProcess = null;
let cancelled = false;

// ============================================================
// System Detection
// ============================================================

function findExecutable(name) {
    try {
        const cmd = process.platform === 'win32' ? 'where' : 'which';
        const result = execSync(`${cmd} ${name}`, {
            encoding: 'utf8',
            timeout: 5000,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        return result.trim().split(/\r?\n/)[0].trim();
    } catch {
        return null;
    }
}

function findPython() {
    // Try to find all Python installations (where/which returns multiple results)
    const candidateNames = process.platform === 'win32'
        ? ['py', 'python3', 'python']
        : ['python3', 'python'];

    for (const name of candidateNames) {
        try {
            const cmd = process.platform === 'win32' ? 'where' : 'which';
            const result = execSync(`${cmd} ${name}`, {
                encoding: 'utf8',
                timeout: 5000,
                stdio: ['pipe', 'pipe', 'pipe'],
            });

            // Get all paths (split by newline)
            const paths = result.trim().split(/\r?\n/).map(p => p.trim()).filter(Boolean);

            // Try each path, validating WindowsApps aliases instead of blindly skipping them
            for (const p of paths) {
                try {
                    // For 'py' launcher, get the actual Python path
                    let pythonPath = p;
                    if (name === 'py') {
                        // Use py launcher to get actual Python executable path
                        const pyPath = execFileSync(p, ['-c', 'import sys; print(sys.executable)'], {
                            encoding: 'utf8',
                            timeout: 5000,
                            stdio: ['pipe', 'pipe', 'pipe'],
                        }).trim();
                        if (pyPath) {
                            pythonPath = pyPath;
                        } else {
                            continue;
                        }
                    } else if (p.includes('WindowsApps') || p.includes('Microsoft\\WindowsApps')) {
                        // WindowsApps paths may be valid MSIX execution aliases or
                        // fake Store redirects. Resolve to the real executable path.
                        try {
                            const resolved = execFileSync(p, ['-c', 'import sys; print(sys.executable)'], {
                                encoding: 'utf8',
                                timeout: 5000,
                                stdio: ['pipe', 'pipe', 'pipe'],
                            }).trim();
                            if (resolved) {
                                pythonPath = resolved;
                            } else {
                                continue;
                            }
                        } catch {
                            // Not a real Python — skip this WindowsApps path
                            continue;
                        }
                    }

                    // Test if this Python works and check version
                    const ver = execFileSync(pythonPath, ['--version'], {
                        encoding: 'utf8',
                        timeout: 5000,
                        stdio: ['pipe', 'pipe', 'pipe'],
                    });
                    const match = (ver || '').match(/(\d+)\.(\d+)/);
                    if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 8) {
                        return pythonPath;
                    }
                } catch {
                    // This Python path doesn't work, try next
                    continue;
                }
            }
        } catch {
            // Command failed, try next name
            continue;
        }
    }

    // On Windows, also try common installation directories
    if (process.platform === 'win32') {
        const commonPaths = [
            path.join(os.homedir(), 'AppData', 'Local', 'Programs', 'Python'),
            'C:\\Python311',
            'C:\\Python310',
            'C:\\Python39',
            'C:\\Python38',
        ];

        for (const dir of commonPaths) {
            if (!fs.existsSync(dir)) continue;

            try {
                const pythonExe = path.join(dir, 'python.exe');
                if (fs.existsSync(pythonExe)) {
                    const ver = execFileSync(pythonExe, ['--version'], {
                        encoding: 'utf8',
                        timeout: 5000,
                        stdio: ['pipe', 'pipe', 'pipe'],
                    });
                    const match = (ver || '').match(/(\d+)\.(\d+)/);
                    if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 8) {
                        return pythonExe;
                    }
                }

                // Also check subdirectories like Python311, Python310, etc.
                const subdirs = fs.readdirSync(dir);
                for (const subdir of subdirs) {
                    const pythonExe = path.join(dir, subdir, 'python.exe');
                    if (fs.existsSync(pythonExe)) {
                        try {
                            const ver = execFileSync(pythonExe, ['--version'], {
                                encoding: 'utf8',
                                timeout: 5000,
                                stdio: ['pipe', 'pipe', 'pipe'],
                            });
                            const match = (ver || '').match(/(\d+)\.(\d+)/);
                            if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 8) {
                                return pythonExe;
                            }
                        } catch {
                            continue;
                        }
                    }
                }
            } catch {
                continue;
            }
        }
    }

    return null;
}

function checkPythonDeps(pythonPath) {
    try {
        execFileSync(pythonPath, ['-c', 'import cv2; import numpy'], {
            encoding: 'utf8',
            timeout: 10000,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        return true;
    } catch {
        return false;
    }
}

function checkPythonPackage(pythonPath, packageName) {
    try {
        execFileSync(pythonPath, ['-c', `import ${packageName}`], {
            encoding: 'utf8',
            timeout: 10000,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        return true;
    } catch {
        return false;
    }
}

function checkVidstabSupport(ffmpegPath) {
    try {
        const result = execFileSync(ffmpegPath, ['-filters'], {
            encoding: 'utf8',
            timeout: 10000,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        return result.includes('vidstabdetect') && result.includes('vidstabtransform');
    } catch {
        return false;
    }
}

function detectEncoder(ffmpegPath) {
    const fallback = { name: 'CPU (libx264)', id: 'libx264', type: 'software' };
    try {
        const result = execFileSync(ffmpegPath, ['-encoders'], {
            encoding: 'utf8',
            timeout: 10000,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        const candidates = [
            { name: 'NVIDIA NVENC', id: 'h264_nvenc', type: 'nvenc' },
            { name: 'Intel Quick Sync', id: 'h264_qsv', type: 'qsv' },
            { name: 'AMD AMF', id: 'h264_amf', type: 'amf' },
            { name: 'Apple VideoToolbox', id: 'h264_videotoolbox', type: 'videotoolbox' },
        ];
        for (const enc of candidates) {
            if (result.includes(enc.id)) return enc;
        }
    } catch {}
    return fallback;
}

function getVideoInfo(ffprobePath, filePath) {
    return new Promise((resolve, reject) => {
        const proc = spawn(ffprobePath, [
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            filePath,
        ]);
        let stdout = '';
        proc.stdout.on('data', d => (stdout += d));
        proc.on('error', err => reject(new Error(`ffprobe failed: ${err.message}`)));
        proc.on('close', code => {
            if (code !== 0) return reject(new Error('Could not read video info'));
            try {
                const data = JSON.parse(stdout);
                const vs = data.streams?.find(s => s.codec_type === 'video');
                if (!vs) return reject(new Error('No video stream found'));
                const fpsMatch = (vs.r_frame_rate || '30/1').match(/(\d+)\/(\d+)/);
                const fps = fpsMatch
                    ? parseInt(fpsMatch[1]) / parseInt(fpsMatch[2])
                    : 30;
                resolve({
                    duration: parseFloat(data.format?.duration || vs.duration || '0'),
                    width: vs.width || 0,
                    height: vs.height || 0,
                    fps: Math.round(fps * 100) / 100,
                    codec: vs.codec_name || 'unknown',
                    size: parseInt(data.format?.size || '0'),
                });
            } catch (e) {
                reject(new Error(`Failed to parse video info: ${e.message}`));
            }
        });
    });
}

// ============================================================
// Presets
// ============================================================

const PRESETS = {
    highQuality: {
        shakiness: 10,
        accuracy: 15,
        smoothing: 30,
        interpol: 'bicubic',
        optzoom: 1,
        zoom: 0,
        crop: 'black',
        encoding: {
            software: ['-preset', 'slow', '-crf', '18'],
            nvenc: ['-preset', 'p7', '-rc', 'vbr', '-b:v', '8M', '-maxrate', '12M', '-bufsize', '16M'],
            qsv: ['-preset', 'veryslow', '-global_quality', '18'],
            amf: ['-usage', 'transcoding', '-rc', 'vbr_peak', '-b:v', '8M'],
            videotoolbox: ['-b:v', '8M'],
        },
    },
};

// ============================================================
// Custom Preset Builder
// ============================================================

const ENCODING_PRESETS = {
    high: {
        software: ['-preset', 'slow', '-crf', '18'],
        nvenc: ['-preset', 'p7', '-rc', 'vbr', '-b:v', '8M', '-maxrate', '12M', '-bufsize', '16M'],
        qsv: ['-preset', 'veryslow', '-global_quality', '18'],
        amf: ['-usage', 'transcoding', '-rc', 'vbr_peak', '-b:v', '8M'],
        videotoolbox: ['-b:v', '8M'],
    },
    balanced: {
        software: ['-preset', 'medium', '-crf', '23'],
        nvenc: ['-preset', 'p4', '-rc', 'vbr', '-b:v', '5M', '-maxrate', '8M', '-bufsize', '10M'],
        qsv: ['-preset', 'fast', '-global_quality', '23'],
        amf: ['-usage', 'transcoding', '-rc', 'vbr_latency', '-b:v', '5M'],
        videotoolbox: ['-b:v', '5M'],
    },
    compressed: {
        software: ['-preset', 'fast', '-crf', '28'],
        nvenc: ['-preset', 'p1', '-rc', 'vbr', '-b:v', '3M', '-maxrate', '5M', '-bufsize', '6M'],
        qsv: ['-preset', 'fast', '-global_quality', '28'],
        amf: ['-usage', 'transcoding', '-rc', 'vbr_latency', '-b:v', '3M'],
        videotoolbox: ['-b:v', '3M'],
    },
};

function buildCustomPreset(settings, encoder) {
    const encQuality = settings.encoding || 'balanced';
    const encPreset = ENCODING_PRESETS[encQuality] || ENCODING_PRESETS.balanced;
    return {
        shakiness: clamp(settings.shakiness ?? 8, 1, 10),
        accuracy: clamp(settings.accuracy ?? 15, 1, 15),
        smoothing: clamp(settings.smoothing ?? 30, 1, 300),
        interpol: ['linear', 'bilinear', 'bicubic'].includes(settings.interpol) ? settings.interpol : 'bicubic',
        optzoom: clamp(settings.optzoom ?? 1, 0, 2),
        zoom: clamp(settings.zoom ?? 0, -50, 50),
        zoomspeed: clamp(settings.zoomspeed ?? 0.25, 0, 5),
        crop: settings.crop === 'keep' ? 'keep' : 'black',
        tripod: settings.tripod ? 1 : 0,
        relative: settings.relative ? 1 : 0,
        maxshift: clamp(settings.maxshift ?? 0, 0, 500),
        maxangle: settings.maxangle ?? 0,
        stepsize: clamp(settings.stepsize ?? 6, 1, 32),
        mincontrast: clamp(settings.mincontrast ?? 0.25, 0, 1),
        encoding: encPreset,
    };
}

function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
}

// ============================================================
// Processing
// ============================================================

function parseTime(str) {
    const m = str.match(/(\d+):(\d+):(\d+\.?\d*)/);
    return m ? parseInt(m[1]) * 3600 + parseInt(m[2]) * 60 + parseFloat(m[3]) : 0;
}

function send(event, channel, data) {
    try {
        event.sender.send(channel, data);
    } catch {}
}

function cleanup(dir) {
    try {
        fs.rmSync(dir, { recursive: true, force: true });
    } catch {}
}

function stabilize(ffmpegPath, input, output, mode, encoder, duration, event, customSettings) {
    return new Promise((resolve, reject) => {
        cancelled = false;
        let preset;
        if (mode === 'custom' && customSettings) {
            preset = buildCustomPreset(customSettings, encoder);
        } else {
            preset = PRESETS[mode];
        }
        if (!preset) return reject(new Error(`Unknown mode: ${mode}`));

        const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'stabbot-'));
        // Use just the filename to avoid Windows drive-letter colon issues in FFmpeg filters;
        // FFmpeg is spawned with cwd=tempDir so it finds the file directly.
        const trfFile = 'transforms.trf';
        const overallStart = Date.now();

        // ---- Phase 1: Motion Detection ----
        let detectFilter = `vidstabdetect=shakiness=${preset.shakiness}:accuracy=${preset.accuracy}:stepsize=${preset.stepsize || 6}:mincontrast=${preset.mincontrast ?? 0.25}:result=${trfFile}`;
        if (preset.tripod) detectFilter += ':tripod=1';
        const detectArgs = [
            '-y', '-i', input,
            '-vf', detectFilter,
            '-f', 'null', '-',
        ];

        send(event, 'progress', { phase: 'detect', percent: 0, overall: 0 });

        const p1 = spawn(ffmpegPath, detectArgs, { cwd: tempDir });
        currentProcess = p1;
        const detectStart = Date.now();
        let p1Stderr = '';

        p1.stderr.on('data', data => {
            const chunk = data.toString();
            p1Stderr += chunk;
            const tm = chunk.match(/time=(\d+:\d+:\d+\.?\d*)/);
            if (tm && duration > 0) {
                const pct = Math.min((parseTime(tm[1]) / duration) * 100, 100);
                send(event, 'progress', { phase: 'detect', percent: pct, overall: pct * 0.4 });
            }
        });

        p1.on('error', err => {
            cleanup(tempDir);
            reject(new Error(`Failed to start FFmpeg: ${err.message}`));
        });

        p1.on('close', code => {
            const detectTime = (Date.now() - detectStart) / 1000;

            if (cancelled) {
                cleanup(tempDir);
                return reject(new Error('CANCELLED'));
            }
            if (code !== 0) {
                cleanup(tempDir);
                const lastLines = p1Stderr.trim().split('\n').slice(-8).join('\n');
                return reject(new Error(`Motion detection failed (exit code ${code})\n\nFFmpeg output:\n${lastLines}\n\nCommand:\nffmpeg ${detectArgs.join(' ')}`));
            }

            // ---- Phase 2: Stabilization Transform ----
            const encFlags = preset.encoding[encoder.type] || preset.encoding.software;
            let filter =
                `vidstabtransform=smoothing=${preset.smoothing}` +
                `:crop=${preset.crop}` +
                `:zoom=${preset.zoom}` +
                `:optzoom=${preset.optzoom}` +
                `:interpol=${preset.interpol}` +
                `:input=${trfFile}`;
            if (preset.tripod) filter += ':tripod=1';
            if (preset.relative) filter += ':relative=1';
            if (preset.zoomspeed !== undefined && preset.zoomspeed > 0) filter += `:zoomspeed=${preset.zoomspeed}`;
            if (preset.maxshift > 0) filter += `:maxshift=${preset.maxshift}`;
            if (preset.maxangle > 0) filter += `:maxangle=${(preset.maxangle * Math.PI / 180).toFixed(4)}`;

            const transformArgs = [
                '-y', '-i', input,
                '-vf', filter,
                '-c:v', encoder.id,
                ...encFlags,
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k',
                '-threads', '0',
                output,
            ];

            send(event, 'progress', { phase: 'transform', percent: 0, overall: 40 });

            const p2 = spawn(ffmpegPath, transformArgs, { cwd: tempDir });
            currentProcess = p2;
            const transformStart = Date.now();
            let p2Stderr = '';

            p2.stderr.on('data', data => {
                const chunk = data.toString();
                p2Stderr += chunk;
                const tm = chunk.match(/time=(\d+:\d+:\d+\.?\d*)/);
                if (tm && duration > 0) {
                    const pct = Math.min((parseTime(tm[1]) / duration) * 100, 100);
                    send(event, 'progress', {
                        phase: 'transform',
                        percent: pct,
                        overall: 40 + pct * 0.6,
                    });
                }
            });

            p2.on('error', err => {
                cleanup(tempDir);
                reject(new Error(`Failed to start FFmpeg: ${err.message}`));
            });

            p2.on('close', code2 => {
                currentProcess = null;
                cleanup(tempDir);
                const transformTime = (Date.now() - transformStart) / 1000;
                const totalTime = (Date.now() - overallStart) / 1000;

                if (cancelled) return reject(new Error('CANCELLED'));
                if (code2 !== 0) {
                    const lastLines = p2Stderr.trim().split('\n').slice(-8).join('\n');
                    return reject(new Error(`Stabilization failed (exit code ${code2})\n\nFFmpeg output:\n${lastLines}\n\nCommand:\nffmpeg ${transformArgs.join(' ')}`));
                }

                let outputSize = 0;
                try { outputSize = fs.statSync(output).size; } catch {}
                resolve({ detectTime, transformTime, totalTime, outputSize });
            });
        });
    });
}

// ============================================================
// Window
// ============================================================

function createWindow() {
    if (process.platform !== 'darwin') {
        Menu.setApplicationMenu(null);
    }

    mainWindow = new BrowserWindow({
        width: 820,
        height: 760,
        minWidth: 680,
        minHeight: 600,
        backgroundColor: '#0c0c14',
        autoHideMenuBar: true,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
        show: false,
        title: 'Video Stabbot',
    });

    mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
    mainWindow.once('ready-to-show', () => mainWindow.show());
}

// ============================================================
// IPC Handlers
// ============================================================

ipcMain.handle('detect-system', async () => {
    const ffmpeg = findExecutable('ffmpeg');
    const ffprobe = findExecutable('ffprobe');
    if (!ffmpeg || !ffprobe) {
        return {
            ok: false,
            error:
                'FFmpeg not found in your system PATH.\n' +
                'Please install FFmpeg from https://ffmpeg.org/download.html ' +
                'and make sure both ffmpeg and ffprobe are available in PATH.',
        };
    }
    const vidstab = checkVidstabSupport(ffmpeg);
    if (!vidstab) {
        return {
            ok: false,
            error:
                'Your FFmpeg build does not include vidstab filters.\n' +
                'Please install a version of FFmpeg compiled with libvidstab support.',
        };
    }
    const encoder = detectEncoder(ffmpeg);
    const python = findPython();
    const hasPythonDeps = python ? checkPythonDeps(python) : false;
    const hasScipy = python ? checkPythonPackage(python, 'scipy') : false;
    const hasTorch = python ? checkPythonPackage(python, 'torch') : false;
    const hasTorchvision = python ? checkPythonPackage(python, 'torchvision') : false;

    // Build detailed RAFT status
    const raftMissing = [];
    if (!python) raftMissing.push('Python 3.8+');
    if (python && !hasPythonDeps) raftMissing.push('opencv-python, numpy');
    if (python && !hasScipy) raftMissing.push('scipy');
    if (python && !hasTorch) raftMissing.push('torch');
    if (python && !hasTorchvision) raftMissing.push('torchvision');

    return {
        ok: true,
        ffmpeg,
        ffprobe,
        encoder,
        python,
        hasPythonDeps,
        hasScipy,
        hasTorch,
        hasTorchvision,
        raftReady: raftMissing.length === 0,
        raftMissing,
    };
});

ipcMain.handle('get-video-info', async (_e, { ffprobe, filePath }) => {
    return getVideoInfo(ffprobe, filePath);
});

ipcMain.handle('select-file', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Video File',
        filters: [
            { name: 'Video Files', extensions: ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v'] },
            { name: 'All Files', extensions: ['*'] },
        ],
        properties: ['openFile'],
    });
    return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('select-save-path', async (_e, defaultPath) => {
    const parsed = path.parse(defaultPath);
    const result = await dialog.showSaveDialog(mainWindow, {
        title: 'Save Stabilized Video',
        defaultPath: path.join(parsed.dir, `${parsed.name}_stabilized${parsed.ext}`),
        filters: [
            { name: 'MP4 Video', extensions: ['mp4'] },
            { name: 'AVI Video', extensions: ['avi'] },
            { name: 'MOV Video', extensions: ['mov'] },
            { name: 'All Files', extensions: ['*'] },
        ],
    });
    return result.canceled ? null : result.filePath;
});

ipcMain.handle('start-processing', async (event, { ffmpeg, input, output, mode, encoder, duration, customSettings }) => {
    return stabilize(ffmpeg, input, output, mode, encoder, duration, event, customSettings);
});

ipcMain.handle('cancel-processing', async () => {
    cancelled = true;
    if (currentProcess) {
        currentProcess.kill();
        currentProcess = null;
    }
    return true;
});

ipcMain.handle('show-in-folder', async (_e, filePath) => {
    shell.showItemInFolder(filePath);
});

ipcMain.handle('run-python-script', async (event, { scriptName, args, duration }) => {
    const PATHS = {
        python: findPython(),
    };
    if (!PATHS.python) {
        throw new Error('Python 3.8+ not found. Please install Python to use advanced modes.');
    }
    const scriptPath = path.join(__dirname, '..', '..', 'scripts', scriptName);
    if (!fs.existsSync(scriptPath)) {
        throw new Error(`Script not found: ${scriptName}`);
    }
    return runPythonScript(PATHS.python, scriptPath, args, duration, event);
});

// ============================================================
// Python Processing
// ============================================================

function runPythonScript(pythonPath, scriptPath, args, duration, event) {
    return new Promise((resolve, reject) => {
        cancelled = false;
        const overallStart = Date.now();

        const proc = spawn(pythonPath, [scriptPath, ...args], {
            cwd: path.dirname(scriptPath),
        });
        currentProcess = proc;
        let stderr = '';

        proc.stdout.on('data', data => {
            const lines = data.toString().split('\n').filter(l => l.trim());
            for (const line of lines) {
                try {
                    const msg = JSON.parse(line);
                    if (msg.ok) {
                        // Final result line — store for close handler
                        proc._resultJSON = msg;
                        if (DEBUG) console.log('[python] ✓ Script completed successfully');
                    } else if (msg.phase !== undefined || msg.progress !== undefined) {
                        if (DEBUG) {
                            const pct = (msg.progress || 0).toFixed(1);
                            const detail = msg.message || '';
                            console.log(`[python] [${msg.phase || 'processing'}] ${pct}% ${detail}`);
                        }
                        send(event, 'progress', {
                            phase: msg.phase || 'processing',
                            percent: msg.progress || 0,
                            overall: msg.progress || 0,
                        });
                    } else if (msg.error) {
                        if (DEBUG) console.error(`[python] ERROR: ${msg.error}`);
                    } else {
                        if (DEBUG) console.log(`[python] ${line}`);
                    }
                } catch {
                    if (DEBUG) console.log(`[python] ${line}`);
                }
            }
        });

        proc.stderr.on('data', data => {
            const chunk = data.toString();
            stderr += chunk;
            if (DEBUG) {
                const lines = chunk.split('\n').filter(l => l.trim());
                for (const line of lines) {
                    console.warn(`[python:stderr] ${line}`);
                }
            }
        });

        proc.on('error', err => {
            currentProcess = null;
            let errorMsg = `Failed to start Python: ${err.message}`;

            // Provide helpful context for common errors
            if (err.code === 'ENOENT') {
                errorMsg =
                    'Python installation not found or not working.\n\n' +
                    'Common causes:\n' +
                    '• Windows Store Python stub detected (not a real installation)\n' +
                    '• Python not installed or not in system PATH\n\n' +
                    'Solution:\n' +
                    '1. Install Python from python.org (not Microsoft Store)\n' +
                    '2. During installation, check "Add Python to PATH"\n' +
                    '3. Restart this application after installing Python\n\n' +
                    `Detected path: ${pythonPath}`;
            }

            reject(new Error(errorMsg));
        });

        proc.on('close', code => {
            currentProcess = null;
            const totalTime = (Date.now() - overallStart) / 1000;

            if (cancelled) return reject(new Error('CANCELLED'));
            if (code !== 0) {
                const errMsg = stderr.trim().split('\n').pop() || `Process exited with code ${code}`;
                return reject(new Error(errMsg));
            }

            let outputSize = 0;
            if (proc._resultJSON && proc._resultJSON.outputSize) {
                outputSize = proc._resultJSON.outputSize;
            } else {
                const outputIdx = args.indexOf('--output');
                if (outputIdx !== -1 && args[outputIdx + 1]) {
                    try { outputSize = fs.statSync(args[outputIdx + 1]).size; } catch {}
                }
            }

            resolve({ totalTime, outputSize, detectTime: 0, transformTime: totalTime });
        });
    });
}

// ============================================================
// App Lifecycle
// ============================================================

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (currentProcess) {
        try { currentProcess.kill(); } catch {}
    }
    app.quit();
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

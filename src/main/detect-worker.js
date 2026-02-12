/**
 * detect-worker.js — Runs system detection in a Worker thread
 * so the main process event loop stays responsive during startup.
 */
const { parentPort } = require('worker_threads');
const { execSync, execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// ============================================================
// Helpers (copied from main — these use sync APIs which is fine
// inside a worker because they don't block the main thread)
// ============================================================

function sendStatus(step, status, message) {
    parentPort.postMessage({ type: 'status', step, status, message });
}

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

            const paths = result.trim().split(/\r?\n/).map(p => p.trim()).filter(Boolean);

            for (const p of paths) {
                try {
                    let pythonPath = p;
                    if (name === 'py') {
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
                            continue;
                        }
                    }

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
                    continue;
                }
            }
        } catch {
            continue;
        }
    }

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

// ============================================================
// Run detection
// ============================================================

(function run() {
    // Step 1: FFmpeg
    sendStatus('ffmpeg', 'active', 'Looking for FFmpeg\u2026');
    const ffmpeg = findExecutable('ffmpeg');
    const ffprobe = findExecutable('ffprobe');
    if (!ffmpeg || !ffprobe) {
        sendStatus('ffmpeg', 'error', 'FFmpeg not found');
        parentPort.postMessage({
            type: 'result',
            data: {
                ok: false,
                error:
                    'FFmpeg not found in your system PATH.\n' +
                    'Please install FFmpeg from https://ffmpeg.org/download.html ' +
                    'and make sure both ffmpeg and ffprobe are available in PATH.',
            },
        });
        return;
    }
    const vidstab = checkVidstabSupport(ffmpeg);
    if (!vidstab) {
        sendStatus('ffmpeg', 'error', 'vidstab not found');
        parentPort.postMessage({
            type: 'result',
            data: {
                ok: false,
                error:
                    'Your FFmpeg build does not include vidstab filters.\n' +
                    'Please install a version of FFmpeg compiled with libvidstab support.',
            },
        });
        return;
    }
    sendStatus('ffmpeg', 'done', 'FFmpeg ready');

    // Step 2: Encoder detection
    sendStatus('encoder', 'active', 'Detecting hardware encoder\u2026');
    const encoder = detectEncoder(ffmpeg);
    sendStatus('encoder', 'done', `Using ${encoder.name}`);

    // Step 3: Python
    sendStatus('python', 'active', 'Searching for Python\u2026');
    const python = findPython();
    sendStatus('python', python ? 'done' : 'skipped', python ? 'Python found' : 'Python not found (optional)');

    // Step 4: Python packages
    sendStatus('packages', 'active', 'Checking Python packages\u2026');
    const hasPythonDeps = python ? checkPythonDeps(python) : false;
    const hasScipy = python ? checkPythonPackage(python, 'scipy') : false;
    const hasTorch = python ? checkPythonPackage(python, 'torch') : false;
    const hasTorchvision = python ? checkPythonPackage(python, 'torchvision') : false;
    sendStatus('packages', 'done', 'Package check complete');

    // Build RAFT status
    const raftMissing = [];
    if (!python) raftMissing.push('Python 3.8+');
    if (python && !hasPythonDeps) raftMissing.push('opencv-python, numpy');
    if (python && !hasScipy) raftMissing.push('scipy');
    if (python && !hasTorch) raftMissing.push('torch');
    if (python && !hasTorchvision) raftMissing.push('torchvision');

    parentPort.postMessage({
        type: 'result',
        data: {
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
        },
    });
})();

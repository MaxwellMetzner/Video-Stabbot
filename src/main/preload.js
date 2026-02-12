const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('stabbot', {
    detectSystem: () => ipcRenderer.invoke('detect-system'),
    getVideoInfo: (data) => ipcRenderer.invoke('get-video-info', data),
    selectFile: () => ipcRenderer.invoke('select-file'),
    selectSavePath: (defaultPath) => ipcRenderer.invoke('select-save-path', defaultPath),
    startProcessing: (opts) => ipcRenderer.invoke('start-processing', opts),
    cancelProcessing: () => ipcRenderer.invoke('cancel-processing'),
    showInFolder: (filePath) => ipcRenderer.invoke('show-in-folder', filePath),
    runPythonScript: (opts) => ipcRenderer.invoke('run-python-script', opts),
    onProgress: (callback) => {
        ipcRenderer.on('progress', (_event, data) => callback(data));
    },
    removeProgressListener: () => {
        ipcRenderer.removeAllListeners('progress');
    },
});

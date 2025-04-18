const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  checkFileExists: (path) => ipcRenderer.invoke('check-file-exists', path),
  openFileDialog: (options) => ipcRenderer.invoke('open-file-dialog', options),
  getFilePath: (fileObject) => ipcRenderer.invoke('get-file-path', fileObject)
});
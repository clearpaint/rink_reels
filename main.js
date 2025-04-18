const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');

// Configure app for better file handling
app.allowRendererProcessReuse = true;

// Security note: These switches reduce security - only use in development
if (process.env.NODE_ENV === 'development') {
  app.commandLine.appendSwitch('disable-web-security');
  app.commandLine.appendSwitch('allow-file-access-from-files');
  app.commandLine.appendSwitch('disable-site-isolation-trials');
}

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: false,  // Disabled for security
      contextIsolation: true,  // Enabled for security
      enableRemoteModule: false,  // Disabled for security
      sandbox: true,  // Enabled for security
      preload: path.join(__dirname, 'preload.js'),
      additionalArguments: process.platform === 'linux' ?
        ['--no-sandbox', '--disable-setuid-sandbox'] : []
    }
  });

  // Load the index.html file
  mainWindow.loadFile(path.join(__dirname, 'index.html'));
mainWindow.webContents.openDevTools({ mode: 'detach' });
  // Open DevTools automatically in development
  if (process.env.NODE_ENV === 'development') {

  }

  // Handle permissions
  mainWindow.webContents.session.setPermissionRequestHandler(
    (webContents, permission, callback) => {
      // Only allow these specific permissions
      const allowedPermissions = ['fullscreen', 'pointerLock'];
      callback(allowedPermissions.includes(permission));
    }
  );

  // Window event listeners
  mainWindow.on('closed', () => (mainWindow = null));
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDesc) => {
    console.error('Failed to load:', errorDesc);
  });
}

// App event listeners
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers
ipcMain.handle('check-file-exists', (event, filePath) => {
  try {
    const fs = require('fs');
    const path = require('path');
    const normalizedPath = path.normalize(filePath);
    return fs.existsSync(normalizedPath);
  } catch (error) {
    console.error('File existence check failed:', error);
    return false;
  }
});

ipcMain.handle('open-file-dialog', async (event, options) => {
  try {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      filters: [{ name: 'Videos', extensions: ['mp4', 'avi', 'mov', 'mkv'] }],
      ...options
    });
    return result.canceled ? null : result.filePaths[0];
  } catch (error) {
    console.error('File dialog error:', error);
    return null;
  }
});

// Optional: Add handler for getting file path from file object
ipcMain.handle('get-file-path', (event, fileObject) => {
  return fileObject.path || null;
});
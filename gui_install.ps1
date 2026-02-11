<#
.SYNOPSIS
    GUI Installer for ComfyUI Models using WinForms.
    Reads install_manifest.json and lets user select models to download.
    Updated with BITS transfer for responsiveness and cancellation support.
#>

param (
    [string]$ManifestPath = "install_manifest.json",
    [string]$DefaultModelRoot = "..\..\models"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# --- Global State ---
$script:cancelRequested = $false
$script:isDownloading = $false

# --- Helper Functions ---

function Get-ModelRoot {
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Select ComfyUI Model Root"
    $form.Size = New-Object System.Drawing.Size(500, 150)
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text = "Please confirm the path to your 'models' directory:"
    $lbl.Location = New-Object System.Drawing.Point(10, 10)
    $lbl.AutoSize = $true
    $form.Controls.Add($lbl)

    $txt = New-Object System.Windows.Forms.TextBox
    $txt.Text = (Resolve-Path $DefaultModelRoot -ErrorAction SilentlyContinue).Path
    if (-not $txt.Text) { $txt.Text = $DefaultModelRoot }

    $txt.Location = New-Object System.Drawing.Point(10, 35)
    $txt.Size = New-Object System.Drawing.Size(380, 20)
    $form.Controls.Add($txt)

    $btnBrowse = New-Object System.Windows.Forms.Button
    $btnBrowse.Text = "Browse..."
    $btnBrowse.Location = New-Object System.Drawing.Point(400, 34)
    $btnBrowse.Size = New-Object System.Drawing.Size(75, 23)
    $btnBrowse.Add_Click({
        $fbd = New-Object System.Windows.Forms.FolderBrowserDialog
        $fbd.SelectedPath = $txt.Text
        if ($fbd.ShowDialog() -eq "OK") {
            $txt.Text = $fbd.SelectedPath
        }
    })
    $form.Controls.Add($btnBrowse)

    $btnOK = New-Object System.Windows.Forms.Button
    $btnOK.Text = "OK"
    $btnOK.DialogResult = "OK"
    $btnOK.Location = New-Object System.Drawing.Point(200, 80)
    $form.Controls.Add($btnOK)

    $form.AcceptButton = $btnOK
    
    if ($form.ShowDialog() -eq "OK") {
        return $txt.Text
    }
    return $null
}

# --- Main Logic ---

if (-not (Test-Path $ManifestPath)) {
    [System.Windows.Forms.MessageBox]::Show("Manifest file not found: $ManifestPath", "Error", "OK", "Error")
    exit
}

try {
    $json = Get-Content -Raw $ManifestPath | ConvertFrom-Json
    $items = $json.downloads
} catch {
    [System.Windows.Forms.MessageBox]::Show("Error reading manifest JSON.", "Error", "OK", "Error")
    exit
}

$rootPath = Get-ModelRoot
if (-not $rootPath) { exit }

# Create UI
$mainForm = New-Object System.Windows.Forms.Form
$mainForm.Text = "ComfyUI Cluster - Model Installer"
$mainForm.Size = New-Object System.Drawing.Size(800, 600)
$mainForm.StartPosition = "CenterScreen"

# Safety: Handle closing while downloading
$mainForm.Add_FormClosing({
    if ($script:isDownloading) {
        $result = [System.Windows.Forms.MessageBox]::Show("Downloads are in progress. Cancel operation and exit?", "Confirm Exit", [System.Windows.Forms.MessageBoxButtons]::YesNo, [System.Windows.Forms.MessageBoxIcon]::Warning)
        if ($result -eq [System.Windows.Forms.DialogResult]::Yes) {
            $script:cancelRequested = $true
            # Give loops a moment to see the flag
        } else {
            $_.Cancel = $true
        }
    }
})

$split = New-Object System.Windows.Forms.SplitContainer
$split.Dock = "Fill"
$split.Orientation = "Horizontal"
$split.SplitterDistance = 400
$mainForm.Controls.Add($split)

# Top: CheckBox List
$chkList = New-Object System.Windows.Forms.CheckedListBox
$chkList.Dock = "Fill"
$chkList.CheckOnClick = $true
$chkList.Font = New-Object System.Drawing.Font("Consolas", 10)
$split.Panel1.Controls.Add($chkList)

# Bottom: Log
$progressPanel = New-Object System.Windows.Forms.Panel
$progressPanel.Dock = "Bottom"
$progressPanel.Height = 50
$split.Panel2.Controls.Add($progressPanel)

$lblProgress = New-Object System.Windows.Forms.Label
$lblProgress.Text = "Ready."
$lblProgress.AutoSize = $true
$lblProgress.Location = New-Object System.Drawing.Point(5, 5)
$lblProgress.Font = New-Object System.Drawing.Font("Segoe UI", 9)
$progressPanel.Controls.Add($lblProgress)

$progressBar = New-Object System.Windows.Forms.ProgressBar
$progressBar.Location = New-Object System.Drawing.Point(5, 25)
$progressBar.Width = 750
$progressBar.Height = 20
$progressBar.Anchor = "Left, Right, Top"
$progressPanel.Controls.Add($progressBar)

$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ScrollBars = "Vertical"
$logBox.ReadOnly = $true
$logBox.Dock = "Fill"
$logBox.BackColor = "Black"
$logBox.ForeColor = "Lime"
$logBox.Text = "Ready. Select models to install.`r`n"
$split.Panel2.Controls.Add($logBox)

# Populate List
$globalItems = @()
foreach ($item in $items) {
    if (-not $item.filename) { continue }
    $subdir = if ($item.subdir) { $item.subdir } else { "checkpoints" }
    $targetPath = Join-Path $rootPath $subdir
    $fullPath = Join-Path $targetPath $item.filename
    
    $exists = Test-Path $fullPath
    $prefix = if ($exists) { "[INSTALLED]" } else { "[MISSING]  " }
    
    $display = "{0} {1} ({2})" -f $prefix, $item.filename, $item.subdir
    
    $idx = $chkList.Items.Add($display)
    
    # Store custom object for later retrieval
    $obj = New-Object PSObject -Property @{
        URL = $item.url;
        FullPath = $fullPath;
        Filename = $item.filename;
        DisplayIndex = $idx;
        Exists = $exists
    }
    $globalItems += $obj
}

# --- Install Action ---

$panelButtons = New-Object System.Windows.Forms.FlowLayoutPanel
$panelButtons.Dock = "Bottom"
$panelButtons.Height = 40
$split.Panel1.Controls.Add($panelButtons)

$btnInstall = New-Object System.Windows.Forms.Button
$btnInstall.Text = "Install Selected"
$btnInstall.Width = 120
$btnInstall.Height = 30
$panelButtons.Controls.Add($btnInstall)

$btnCancel = New-Object System.Windows.Forms.Button
$btnCancel.Text = "Cancel"
$btnCancel.Width = 80
$btnCancel.Height = 30
$btnCancel.Enabled = $false
$panelButtons.Controls.Add($btnCancel)

$btnSelectMissing = New-Object System.Windows.Forms.Button
$btnSelectMissing.Text = "Select All Missing"
$btnSelectMissing.Width = 120
$btnSelectMissing.Height = 30
$btnSelectMissing.Add_Click({
    for($i=0; $i -lt $chkList.Items.Count; $i++) {
        $obj = $globalItems[$i]
        if (-not $obj.Exists) {
            $chkList.SetItemChecked($i, $true)
        }
    }
})
$panelButtons.Controls.Add($btnSelectMissing)

$btnExit = New-Object System.Windows.Forms.Button
$btnExit.Text = "Exit"
$btnExit.Width = 80
$btnExit.Height = 30
$btnExit.Add_Click({ $mainForm.Close() })
$panelButtons.Controls.Add($btnExit)

# Event Handlers

$btnCancel.Add_Click({
    if ($script:isDownloading) {
        $script:cancelRequested = $true
        $logBox.AppendText("Cancellation Requested...`r`n")
        $btnCancel.Enabled = $false # Prevent double click
    }
})

$btnInstall.Add_Click({
    $filesToDownload = @()
    foreach ($idx in $chkList.CheckedIndices) {
        $filesToDownload += $globalItems[$idx]
    }

    if ($filesToDownload.Count -eq 0) {
        [System.Windows.Forms.MessageBox]::Show("No items selected.")
        return
    }

    $logBox.AppendText("Starting Download of $($filesToDownload.Count) items...`r`n")
    
    # Lock UI
    $script:isDownloading = $true
    $script:cancelRequested = $false
    $btnInstall.Enabled = $false
    $btnCancel.Enabled = $true
    $btnExit.Enabled = $false
    $btnSelectMissing.Enabled = $false
    
    foreach ($f in $filesToDownload) {
        if ($script:cancelRequested) { break }
        
        $progressBar.Value = 0
        $lblProgress.Text = "Initializing..."

        $url = $f.URL
        # Ensure we are using the absolute path or resolved path stored
        $dest = $f.FullPath
        $dir = Split-Path $dest
        
        if (-not (Test-Path $dir)) {
             $logBox.AppendText("Creating directory: $dir`r`n")
             New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }

        if (Test-Path $dest) {
            $existingSize = (Get-Item $dest).Length
            # Increase threshold to 512KB to catch HTML error pages (often 100-300KB)
            if ($existingSize -gt 524288) {
                 $logBox.AppendText("Skipping (Exists, $existingSize bytes): $($f.Filename)`r`n")
                 continue
            }
            $logBox.AppendText("File exists but small ($existingSize bytes). Overwriting: $($f.Filename)`r`n")
        }

        if (-not $url -or $url.Trim() -eq "") {
            $logBox.AppendText("ERROR: No URL for $($f.Filename)`r`n")
            continue
        }

        # --- Auto-Resolve Civitai Model Paths ---
        # Checks if URL is a generic model page (e.g. civitai.com/models/12345) and resolves to download URL via API
        if ($url -match 'https?://civitai\.com/models/(\d+)') {
            $civitaiId = $matches[1]
            if ($url -notmatch '/api/download/') {
                $logBox.AppendText("Resolving Civitai Model Link (ID: $civitaiId)...`r`n")
                $mainForm.Refresh()
                try {
                    $apiUri = "https://civitai.com/api/v1/models/$civitaiId"
                    $meta = Invoke-RestMethod -Uri $apiUri -ErrorAction Stop
                    
                    # Attempt to find the correct file
                    if ($meta.modelVersions -and $meta.modelVersions.Count -gt 0) {
                        # Default to the first file of the first (latest) version
                        $ver = $meta.modelVersions[0]
                        if ($ver.files -and $ver.files.Count -gt 0) {
                             $resolvedUrl = $ver.files[0].downloadUrl
                             if ($resolvedUrl) {
                                 $url = $resolvedUrl
                                 $logBox.AppendText("  -> Resolved: $url`r`n")
                             }
                        }
                    }
                } catch {
                    $logBox.AppendText("WARNING: Failed to resolve Civitai API for ID $civitaiId. Using original URL.`r`n")
                }
            }
        }
        # ----------------------------------------

        $logBox.AppendText("Downloading: $($f.Filename)`r`n  To: $dest `r`n")
        $mainForm.Refresh()
        
        $downloadSuccess = $false

        # ----------------------------------------
        
        $isCivitai = $url -match "civitai.com"
        # BITS often fails with Civitai (403 Forbidden) due to missing User-Agent/Headers.
        # We will skip BITS for Civitai unless we can configure it, favoring the fallback which we can patch.
        
        # --- Method 1: BITS Transfer ---
        if (-not $isCivitai) {
            try {
                # Use BITS Transfer
                if (-not (Get-Module -ListAvailable BitsTransfer)) {
                     $logBox.AppendText("WARNING: BitsTransfer module not found, skipping to fallback...`r`n")
                } else {
                    Import-Module BitsTransfer -ErrorAction SilentlyContinue
                    
                    # Start job
                    $job = Start-BitsTransfer -Source $url -Destination $dest -Asynchronous -DisplayName "ComfyUI_Cluster_DL" -Priority Foreground
                    
                    # Monitoring loop
                    while ($job.JobState -eq "Transferring" -or $job.JobState -eq "Connecting" -or $job.JobState -eq "Queued") {
                        [System.Windows.Forms.Application]::DoEvents()
                        
                        if ($script:cancelRequested) {
                            $logBox.AppendText("Cancellation Requested... stopping BITS job.`r`n")
                            Remove-BitsTransfer -BitsJob $job -ErrorAction SilentlyContinue
                            break 
                        }
                        
                        # Update visual progress if possible
                        if ($job.BytesTotal -gt 0) {
                            $pct = [math]::Round(($job.BytesTransferred / $job.BytesTotal) * 100)
                            $progressBar.Value = $pct
                            $lblProgress.Text = "BITS: {0:N2} MB / {1:N2} MB" -f ($job.BytesTransferred / 1MB), ($job.BytesTotal / 1MB)
                        }

                        Start-Sleep -Milliseconds 200
                    }
                    
                    if ($script:cancelRequested) { break }

                    # Finalize
                    if ($job.JobState -eq "Transferred") {
                        Complete-BitsTransfer -BitsJob $job
                        $downloadSuccess = $true
                        $logBox.AppendText("BITS Download DONE.`r`n")
                    } else {
                         $errDesc = $job.ErrorDescription
                         $logBox.AppendText("BITS Failed ($($job.JobState)): $errDesc`r`n")
                         Remove-BitsTransfer -BitsJob $job -ErrorAction SilentlyContinue
                    }
                }
                
            } catch {
                $logBox.AppendText("BITS Exception: $_`r`n")
            }
        } else {
            $logBox.AppendText("Skipping BITS for Civitai URL (requires User-Agent)...`r`n")
        }
        
        # --- Method 2: Fallback (WebClient) ---
        if (-not $downloadSuccess -and -not $script:cancelRequested) {
            $logBox.AppendText(">> Starting Download (WebClient)...`r`n")
            $mainForm.Refresh()

            try {
                $wc = New-Object System.Net.WebClient
                $wc.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                ; "BytesIn" = 0; "TotalBytes" = 0 })
                
                $wc.Add_DownloadFileCompleted({
                    param($sender, $e)
                    $paramSync["Complete"] = $true
                    if ($e.Error) { $paramSync["Error"] = $e.Error }
                })
                
                $wc.Add_DownloadProgressChanged({
                    param($sender, $e)
                    $paramSync["BytesIn"] = $e.BytesReceived
                    $paramSync["TotalBytes"] = $e.TotalBytesToReceive
                })
                
                $wc.DownloadFileAsync($url, $dest)
                
                while (-not $paramSync["Complete"]) {
                    [System.Windows.Forms.Application]::DoEvents()
                    if ($script:cancelRequested) {
                        $wc.CancelAsync()
                        $logBox.AppendText("Download CANCELLED by user.`r`n")
                        break
                    }
                    
                    if ($paramSync["TotalBytes"] -gt 0) {
                        $curr = $paramSync["BytesIn"]
                        $tot = $paramSync["TotalBytes"]
                        $pct = [math]::Round(($curr / $tot) * 100)
                        $progressBar.Value = $pct
                        $lblProgress.Text = "WebClient: {0:N2} MB / {1:N2} MB" -f ($curr / 1MB), ($tot / 1MB)
                    }
   $logBox.AppendText("Download CANCELLED by user.`r`n")
                        break
                    }
                    Start-Sleep -Milliseconds 100
                }
                
                if ($script:cancelRequested) { break }
                
                if ($paramSync["Error"]) {
                    throw $paramSync["Error"]
                }
                
                $downloadSuccess = $true
                $logBox.AppendText("Download DONE.`r`n")
                
            } catch {
                $logBox.AppendText("Download FAILED: $_`r`n")
                if (Test-Path $dest) { Remove-Item $dest -Force -ErrorAction SilentlyContinue }
            }
        }
        
        # --- Final Verification ---
        if ($downloadSuccess) {
             if (Test-Path $dest) {
                $finalSize = (Get-Item $dest).Length
                $logBox.AppendText("VERIFIED. Size: $finalSize bytes.`r`n")
             } else {
                $logBox.AppendText("ERROR: Reported success but file is missing!`r`n")
             }
        }

        $mainForm.Refresh()
    }
    
    if ($script:cancelRequested) {
        $logBox.AppendText("Batch Cancelled.`r`n")
    } else {
        $logBox.AppendText("Batch Completed.`r`n")
        [System.Windows.Forms.MessageBox]::Show("Download Batch Complete.", "Done")
    }
    
    # Unlock UI
    $script:isDownloading = $false
    $script:cancelRequested = $false
    $btnInstall.Enabled = $true
    $btnCancel.Enabled = $false
    $btnExit.Enabled = $true
    $btnSelectMissing.Enabled = $true
})

$mainForm.Add_Shown({
    $mainForm.Activate()
})

[void]$mainForm.ShowDialog()

<#
.SYNOPSIS
    ComfyUI-Cluster model installer by Geekatplay Studio.

.DESCRIPTION
    Reads install_manifest.json (generated from model_registry.json by
    update_manifest.py) and downloads the models you tick into your ComfyUI
    models folder.

    - Ticking a model also ticks the text encoder and VAE files it needs.
    - Downloads go to a .part file first and are renamed when complete, so a
      cancelled or failed download never leaves a broken model behind.
    - Set the HF_TOKEN environment variable to download gated Hugging Face models.
#>

param (
    [string]$ManifestPath = "install_manifest.json",
    [string]$DefaultModelRoot = "..\..\models"
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Brand = "Geekatplay Studio"
$AppTitle = "$Brand - ComfyUI-Cluster Model Installer"

# --- Theme --------------------------------------------------------------------
$ColBack   = [System.Drawing.Color]::FromArgb(24, 24, 28)
$ColPanel  = [System.Drawing.Color]::FromArgb(34, 34, 40)
$ColInput  = [System.Drawing.Color]::FromArgb(44, 44, 52)
$ColText   = [System.Drawing.Color]::FromArgb(232, 232, 236)
$ColMuted  = [System.Drawing.Color]::FromArgb(160, 160, 170)
$ColAccent = [System.Drawing.Color]::FromArgb(63, 120, 158)
$ColGood   = [System.Drawing.Color]::FromArgb(130, 200, 140)
$ColWarn   = [System.Drawing.Color]::FromArgb(230, 180, 90)
$FontUi    = New-Object System.Drawing.Font("Segoe UI", 9.5)
$FontBold  = New-Object System.Drawing.Font("Segoe UI Semibold", 9.5)
$FontTitle = New-Object System.Drawing.Font("Segoe UI Semibold", 14)
$FontMono  = New-Object System.Drawing.Font("Consolas", 9.5)

$script:cancelRequested = $false
$script:isDownloading = $false
$script:updatingChecks = $false

function New-ThemedButton([string]$text, [int]$width, [bool]$primary = $false) {
    $b = New-Object System.Windows.Forms.Button
    $b.Text = $text
    $b.Width = $width
    $b.Height = 32
    $b.FlatStyle = "Flat"
    $b.FlatAppearance.BorderSize = 1
    $b.FlatAppearance.BorderColor = $ColAccent
    $b.ForeColor = $ColText
    $b.Font = $FontUi
    $b.Margin = New-Object System.Windows.Forms.Padding(0, 0, 8, 0)
    if ($primary) { $b.BackColor = $ColAccent } else { $b.BackColor = $ColInput }
    return $b
}

function Format-Size([double]$gb) {
    if ($gb -le 0) { return "" }
    if ($gb -lt 1) { return ("{0:N0} MB" -f ($gb * 1024)) }
    return ("{0:N1} GB" -f $gb)
}

function Show-Message([string]$text, [string]$icon = "Information") {
    [void][System.Windows.Forms.MessageBox]::Show($text, $AppTitle, "OK", $icon)
}

# --- Step 1: pick the models folder ------------------------------------------
function Get-ModelRoot {
    $form = New-Object System.Windows.Forms.Form
    $form.Text = $AppTitle
    $form.ClientSize = New-Object System.Drawing.Size(640, 170)
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false
    $form.MinimizeBox = $false
    $form.BackColor = $ColBack
    $form.ForeColor = $ColText
    $form.Font = $FontUi

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text = "Where is your ComfyUI 'models' folder?"
    $lbl.Font = $FontBold
    $lbl.Location = New-Object System.Drawing.Point(16, 16)
    $lbl.Size = New-Object System.Drawing.Size(608, 22)
    $form.Controls.Add($lbl)

    $hint = New-Object System.Windows.Forms.Label
    $hint.Text = "Models are saved into its diffusion_models, text_encoders, vae and loras subfolders."
    $hint.ForeColor = $ColMuted
    $hint.AutoEllipsis = $true
    $hint.Location = New-Object System.Drawing.Point(16, 40)
    $hint.Size = New-Object System.Drawing.Size(608, 22)
    $form.Controls.Add($hint)

    $txt = New-Object System.Windows.Forms.TextBox
    $resolved = Resolve-Path $DefaultModelRoot -ErrorAction SilentlyContinue
    if ($resolved) { $txt.Text = $resolved.Path } else { $txt.Text = $DefaultModelRoot }
    $txt.BackColor = $ColInput
    $txt.ForeColor = $ColText
    $txt.BorderStyle = "FixedSingle"
    $txt.Location = New-Object System.Drawing.Point(16, 72)
    $txt.Size = New-Object System.Drawing.Size(500, 26)
    $form.Controls.Add($txt)

    $btnBrowse = New-ThemedButton "Browse..." 100
    $btnBrowse.Location = New-Object System.Drawing.Point(524, 69)
    $btnBrowse.Add_Click({
        $fbd = New-Object System.Windows.Forms.FolderBrowserDialog
        if (Test-Path $txt.Text) { $fbd.SelectedPath = $txt.Text }
        if ($fbd.ShowDialog() -eq "OK") { $txt.Text = $fbd.SelectedPath }
    })
    $form.Controls.Add($btnBrowse)

    $btnOK = New-ThemedButton "Continue" 110 $true
    $btnOK.DialogResult = "OK"
    $btnOK.Location = New-Object System.Drawing.Point(398, 120)
    $form.Controls.Add($btnOK)

    $btnQuit = New-ThemedButton "Cancel" 100
    $btnQuit.DialogResult = "Cancel"
    $btnQuit.Location = New-Object System.Drawing.Point(524, 120)
    $form.Controls.Add($btnQuit)

    $form.AcceptButton = $btnOK
    $form.CancelButton = $btnQuit
    if ($form.ShowDialog() -eq "OK" -and $txt.Text.Trim()) { return $txt.Text.Trim() }
    return $null
}

# --- Load manifest ------------------------------------------------------------
if (-not (Test-Path $ManifestPath)) {
    Show-Message "Manifest file not found: $ManifestPath`r`nRun update_manifest.py first." "Error"
    exit 1
}
try {
    $items = @((Get-Content -Raw -Encoding UTF8 $ManifestPath | ConvertFrom-Json).downloads)
} catch {
    Show-Message "install_manifest.json could not be read:`r`n$_" "Error"
    exit 1
}

$rootPath = Get-ModelRoot
if (-not $rootPath) { exit 0 }

# Folders ComfyUI treats as the same place, so files already installed elsewhere are recognised.
$FolderAliases = @{
    "diffusion_models" = @("diffusion_models", "unet", "checkpoints")
    "checkpoints"      = @("checkpoints", "diffusion_models", "unet")
    "text_encoders"    = @("text_encoders", "clip")
    "vae"              = @("vae")
    "loras"            = @("loras")
}

function Find-Installed($item) {
    $subdir = if ($item.subdir) { $item.subdir } else { "checkpoints" }
    $folders = if ($FolderAliases.ContainsKey($subdir)) { $FolderAliases[$subdir] } else { @($subdir) }
    foreach ($f in $folders) {
        $p = Join-Path (Join-Path $rootPath $f) $item.filename
        if ((Test-Path $p) -and ((Get-Item $p).Length -gt 1MB)) { return $p }
    }
    return $null
}

# --- Main window --------------------------------------------------------------
$mainForm = New-Object System.Windows.Forms.Form
$mainForm.Text = $AppTitle
$mainForm.Size = New-Object System.Drawing.Size(1120, 760)
$mainForm.MinimumSize = New-Object System.Drawing.Size(900, 600)
$mainForm.StartPosition = "CenterScreen"
$mainForm.BackColor = $ColBack
$mainForm.ForeColor = $ColText
$mainForm.Font = $FontUi

$layout = New-Object System.Windows.Forms.TableLayoutPanel
$layout.Dock = "Fill"
$layout.ColumnCount = 1
$layout.RowCount = 5
$layout.Padding = New-Object System.Windows.Forms.Padding(14)
[void]$layout.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", 100)))
[void]$layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 64)))   # header
[void]$layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 44)))   # filters
[void]$layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 68)))    # list
[void]$layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 100)))  # actions + progress
[void]$layout.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 32)))    # log
$mainForm.Controls.Add($layout)

# Header
$header = New-Object System.Windows.Forms.Panel
$header.Dock = "Fill"
$lblTitle = New-Object System.Windows.Forms.Label
$lblTitle.Text = "$Brand  |  ComfyUI-Cluster Model Installer"
$lblTitle.Font = $FontTitle
$lblTitle.AutoEllipsis = $true
$lblTitle.Dock = "Top"
$lblTitle.Height = 32
$lblRoot = New-Object System.Windows.Forms.Label
$lblRoot.Text = "Installing to: $rootPath"
$lblRoot.ForeColor = $ColMuted
$lblRoot.AutoEllipsis = $true
$lblRoot.Dock = "Top"
$lblRoot.Height = 22
$header.Controls.Add($lblRoot)
$header.Controls.Add($lblTitle)
$layout.Controls.Add($header, 0, 0)

# Filters
$filters = New-Object System.Windows.Forms.FlowLayoutPanel
$filters.Dock = "Fill"
$filters.WrapContents = $false
$filters.Padding = New-Object System.Windows.Forms.Padding(0, 6, 0, 0)

$lblVram = New-Object System.Windows.Forms.Label
$lblVram.Text = "My GPU has"
$lblVram.AutoSize = $true
$lblVram.Margin = New-Object System.Windows.Forms.Padding(0, 6, 6, 0)
$filters.Controls.Add($lblVram)

$cmbVram = New-Object System.Windows.Forms.ComboBox
$cmbVram.DropDownStyle = "DropDownList"
$cmbVram.Width = 130
$cmbVram.BackColor = $ColInput
$cmbVram.ForeColor = $ColText
$cmbVram.FlatStyle = "Flat"
[void]$cmbVram.Items.AddRange(@("any amount", "48 GB", "32 GB", "24 GB", "16 GB", "12 GB", "8 GB"))
$cmbVram.SelectedIndex = 0
$cmbVram.Margin = New-Object System.Windows.Forms.Padding(0, 2, 18, 0)
$filters.Controls.Add($cmbVram)

$lblSearch = New-Object System.Windows.Forms.Label
$lblSearch.Text = "Search"
$lblSearch.AutoSize = $true
$lblSearch.Margin = New-Object System.Windows.Forms.Padding(0, 6, 6, 0)
$filters.Controls.Add($lblSearch)

$txtSearch = New-Object System.Windows.Forms.TextBox
$txtSearch.Width = 240
$txtSearch.BackColor = $ColInput
$txtSearch.ForeColor = $ColText
$txtSearch.BorderStyle = "FixedSingle"
$txtSearch.Margin = New-Object System.Windows.Forms.Padding(0, 3, 18, 0)
$filters.Controls.Add($txtSearch)

$lblToken = New-Object System.Windows.Forms.Label
$lblToken.AutoSize = $true
$lblToken.Margin = New-Object System.Windows.Forms.Padding(0, 6, 0, 0)
if ($env:HF_TOKEN) {
    $lblToken.Text = "HF_TOKEN found: gated models can be downloaded."
    $lblToken.ForeColor = $ColGood
} else {
    $lblToken.Text = "HF_TOKEN not set: models marked 'needs token' will fail."
    $lblToken.ForeColor = $ColWarn
}
$filters.Controls.Add($lblToken)
$layout.Controls.Add($filters, 0, 1)

# List
$list = New-Object System.Windows.Forms.ListView
$list.Dock = "Fill"
$list.View = "Details"
$list.CheckBoxes = $true
$list.FullRowSelect = $true
$list.HideSelection = $true
$list.ShowItemToolTips = $true
$list.BorderStyle = "FixedSingle"
$list.BackColor = $ColPanel
$list.ForeColor = $ColText
$list.Font = $FontUi
$list.HeaderStyle = "Nonclickable"
[void]$list.Columns.Add("File", 360)
[void]$list.Columns.Add("Status", 90)
[void]$list.Columns.Add("Kind", 100)
[void]$list.Columns.Add("Family", 110)
[void]$list.Columns.Add("Size", 80, "Right")
[void]$list.Columns.Add("VRAM", 70, "Right")
[void]$list.Columns.Add("Details", 240)
$layout.Controls.Add($list, 0, 2)

# The last column takes whatever width is left, so nothing is cut off by a fixed size.
function Resize-Columns {
    $used = 0
    for ($i = 0; $i -lt $list.Columns.Count - 1; $i++) { $used += $list.Columns[$i].Width }
    $rest = $list.ClientSize.Width - $used - 4
    if ($rest -lt 160) { $rest = 160 }
    $list.Columns[$list.Columns.Count - 1].Width = $rest
}
$list.Add_Resize({ Resize-Columns })

# Actions + progress
$actions = New-Object System.Windows.Forms.TableLayoutPanel
$actions.Dock = "Fill"
$actions.ColumnCount = 1
$actions.RowCount = 3
[void]$actions.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 44)))
[void]$actions.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 24)))
[void]$actions.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 24)))

$buttons = New-Object System.Windows.Forms.FlowLayoutPanel
$buttons.Dock = "Fill"
$buttons.WrapContents = $false
$buttons.Padding = New-Object System.Windows.Forms.Padding(0, 8, 0, 0)
$btnInstall = New-ThemedButton "Download selected" 160 $true
$btnSelectMissing = New-ThemedButton "Select all missing" 150
$btnClear = New-ThemedButton "Clear selection" 130
$btnCancel = New-ThemedButton "Cancel download" 140
$btnCancel.Enabled = $false
$btnExit = New-ThemedButton "Close" 90
$lblSelection = New-Object System.Windows.Forms.Label
$lblSelection.AutoSize = $true
$lblSelection.ForeColor = $ColMuted
$lblSelection.Margin = New-Object System.Windows.Forms.Padding(8, 8, 0, 0)
$buttons.Controls.AddRange(@($btnInstall, $btnSelectMissing, $btnClear, $btnCancel, $btnExit, $lblSelection))
$actions.Controls.Add($buttons, 0, 0)

$lblProgress = New-Object System.Windows.Forms.Label
$lblProgress.Text = "Ready. Tick the models you want."
$lblProgress.Dock = "Fill"
$lblProgress.AutoEllipsis = $true
$lblProgress.TextAlign = "MiddleLeft"
$actions.Controls.Add($lblProgress, 0, 1)

$progressBar = New-Object System.Windows.Forms.ProgressBar
$progressBar.Dock = "Fill"
$progressBar.Style = "Continuous"
$actions.Controls.Add($progressBar, 0, 2)
$layout.Controls.Add($actions, 0, 3)

# Log
$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Multiline = $true
$logBox.ScrollBars = "Vertical"
$logBox.WordWrap = $true
$logBox.ReadOnly = $true
$logBox.Dock = "Fill"
$logBox.BorderStyle = "FixedSingle"
$logBox.BackColor = [System.Drawing.Color]::FromArgb(16, 16, 20)
$logBox.ForeColor = $ColGood
$logBox.Font = $FontMono
$layout.Controls.Add($logBox, 0, 4)

function Write-Log([string]$text) {
    $logBox.AppendText("$text`r`n")
}

# --- Populate -----------------------------------------------------------------
$script:rows = @()
foreach ($item in $items) {
    if (-not $item.filename) { continue }
    $subdir = if ($item.subdir) { $item.subdir } else { "checkpoints" }
    $found = Find-Installed $item
    $script:rows += New-Object PSObject -Property @{
        Item      = $item
        Filename  = [string]$item.filename
        Url       = [string]$item.url
        Dest      = (Join-Path (Join-Path $rootPath $subdir) $item.filename)
        Installed = [bool]$found
        Checked   = $false
        Needs     = @($item.needs)
    }
}

function Get-VramLimit {
    if ($cmbVram.SelectedIndex -le 0) { return 0 }
    return [int](($cmbVram.SelectedItem -split " ")[0])
}

function Update-SelectionLabel {
    $picked = @($script:rows | Where-Object { $_.Checked -and -not $_.Installed })
    $gb = 0.0
    foreach ($r in $picked) { $gb += [double]$r.Item.size_gb }
    if ($picked.Count -eq 0) {
        $lblSelection.Text = "Nothing selected."
    } else {
        $lblSelection.Text = "{0} file(s) selected, about {1}" -f $picked.Count, (Format-Size $gb)
    }
}

function Update-List {
    $limit = Get-VramLimit
    $needle = $txtSearch.Text.Trim().ToLower()
    $script:updatingChecks = $true
    $list.BeginUpdate()
    $list.Items.Clear()
    foreach ($r in $script:rows) {
        $it = $r.Item
        $vram = [int]$it.min_vram
        if ($limit -gt 0 -and $vram -gt $limit) { continue }
        $haystack = ("{0} {1} {2} {3}" -f $r.Filename, $it.kind, $it.group, $it.description).ToLower()
        if ($needle -and -not $haystack.Contains($needle)) { continue }

        $lvi = New-Object System.Windows.Forms.ListViewItem($r.Filename)
        $status = if ($r.Installed) { "Installed" } else { "Missing" }
        $details = [string]$it.description
        if ($it.gated) { $details = "Needs token. $details" }
        [void]$lvi.SubItems.Add($status)
        [void]$lvi.SubItems.Add([string]$it.kind)
        [void]$lvi.SubItems.Add([string]$it.group)
        [void]$lvi.SubItems.Add((Format-Size ([double]$it.size_gb)))
        [void]$lvi.SubItems.Add($(if ($vram -gt 0) { "$vram GB" } else { "" }))
        [void]$lvi.SubItems.Add($details)
        $lvi.ToolTipText = "{0}`r`n{1}`r`nSaves to: {2}" -f $r.Filename, $details, $r.Dest
        if ($r.Installed) { $lvi.ForeColor = $ColGood } elseif ($it.gated) { $lvi.ForeColor = $ColWarn }
        $lvi.Checked = $r.Checked
        $lvi.Tag = $r
        [void]$list.Items.Add($lvi)
    }
    $list.EndUpdate()
    $script:updatingChecks = $false
    Resize-Columns
    Update-SelectionLabel
}

# Ticking a model also ticks the companion files it needs.
$list.Add_ItemChecked({
    param($sender, $e)
    if ($script:updatingChecks) { return }
    $row = $e.Item.Tag
    $row.Checked = $e.Item.Checked
    if ($row.Checked -and $row.Needs.Count -gt 0) {
        $script:updatingChecks = $true
        foreach ($need in $row.Needs) {
            $dep = $script:rows | Where-Object { $_.Filename -eq $need } | Select-Object -First 1
            if ($dep -and -not $dep.Installed) {
                $dep.Checked = $true
                foreach ($other in $list.Items) { if ($other.Text -eq $dep.Filename) { $other.Checked = $true } }
            }
        }
        $script:updatingChecks = $false
    }
    Update-SelectionLabel
})

$cmbVram.Add_SelectedIndexChanged({ Update-List })
$txtSearch.Add_TextChanged({ Update-List })

$btnSelectMissing.Add_Click({
    foreach ($lvi in $list.Items) {
        if (-not $lvi.Tag.Installed) { $lvi.Checked = $true }
    }
})
$btnClear.Add_Click({
    foreach ($r in $script:rows) { $r.Checked = $false }
    Update-List
})
$btnExit.Add_Click({ $mainForm.Close() })
$btnCancel.Add_Click({
    if ($script:isDownloading) {
        $script:cancelRequested = $true
        $btnCancel.Enabled = $false
        Write-Log "Cancelling..."
    }
})

$mainForm.Add_FormClosing({
    if ($script:isDownloading) {
        $answer = [System.Windows.Forms.MessageBox]::Show("A download is running. Cancel it and close?", $AppTitle, "YesNo", "Warning")
        if ($answer -eq "Yes") { $script:cancelRequested = $true } else { $_.Cancel = $true }
    }
})

# --- Download -----------------------------------------------------------------
# Streams one file to "<dest>.part" and renames it when complete.
# Returns "ok", "cancelled" or an error message.
function Save-File([string]$url, [string]$dest, [string]$label) {
    $part = "$dest.part"
    $response = $null
    $inStream = $null
    $outStream = $null
    try {
        $request = [System.Net.HttpWebRequest]::Create($url)
        $request.UserAgent = "ComfyUI-Cluster-Installer"
        $request.Timeout = 30000
        $request.ReadWriteTimeout = 60000
        $request.AllowAutoRedirect = $true
        if ($env:HF_TOKEN -and $url -match "^https://huggingface\.co/") {
            $request.Headers.Add("Authorization", "Bearer $($env:HF_TOKEN)")
        }
        $response = $request.GetResponse()
        $total = $response.ContentLength
        $inStream = $response.GetResponseStream()
        $outStream = [System.IO.File]::Open($part, [System.IO.FileMode]::Create)
        $buffer = New-Object byte[] (1MB)
        $done = [long]0
        $lastUi = [DateTime]::Now
        $started = [DateTime]::Now
        while ($true) {
            $read = $inStream.Read($buffer, 0, $buffer.Length)
            if ($read -le 0) { break }
            $outStream.Write($buffer, 0, $read)
            $done += $read
            if (([DateTime]::Now - $lastUi).TotalMilliseconds -ge 150) {
                $lastUi = [DateTime]::Now
                $seconds = [Math]::Max(([DateTime]::Now - $started).TotalSeconds, 0.1)
                $speed = ($done / 1MB) / $seconds
                if ($total -gt 0) {
                    $progressBar.Value = [int][Math]::Min(100, [Math]::Floor(($done / $total) * 100))
                    $lblProgress.Text = "{0}: {1:N0} of {2:N0} MB at {3:N1} MB/s" -f $label, ($done / 1MB), ($total / 1MB), $speed
                } else {
                    $lblProgress.Text = "{0}: {1:N0} MB at {2:N1} MB/s" -f $label, ($done / 1MB), $speed
                }
                [System.Windows.Forms.Application]::DoEvents()
                if ($script:cancelRequested) { return "cancelled" }
            }
        }
        $outStream.Close(); $outStream = $null
        if ($total -gt 0 -and $done -ne $total) { return "download ended early ($done of $total bytes)" }
        if ($done -lt 1MB) { return "server sent only $done bytes (not a model file)" }
        if (Test-Path $dest) { Remove-Item $dest -Force }
        Move-Item $part $dest -Force
        return "ok"
    } catch [System.Net.WebException] {
        $code = 0
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        if ($code -eq 401 -or $code -eq 403) {
            return "access denied (HTTP $code). Accept the model licence on Hugging Face and set HF_TOKEN."
        }
        if ($code -eq 404) { return "file not found on the server (HTTP 404)" }
        return $_.Exception.Message
    } catch {
        return "$_"
    } finally {
        if ($outStream) { $outStream.Close() }
        if ($inStream) { $inStream.Close() }
        if ($response) { $response.Close() }
        if (Test-Path $part) { Remove-Item $part -Force -ErrorAction SilentlyContinue }
    }
}

$btnInstall.Add_Click({
    $queue = @($script:rows | Where-Object { $_.Checked -and -not $_.Installed })
    if ($queue.Count -eq 0) {
        Show-Message "Tick at least one missing model first."
        return
    }

    $script:isDownloading = $true
    $script:cancelRequested = $false
    foreach ($c in @($btnInstall, $btnSelectMissing, $btnClear, $btnExit, $cmbVram, $txtSearch, $list)) { $c.Enabled = $false }
    $btnCancel.Enabled = $true

    $errorLog = Join-Path $PWD "install_errors.log"
    $failed = 0
    $index = 0
    Write-Log "Downloading $($queue.Count) file(s)..."
    foreach ($row in $queue) {
        if ($script:cancelRequested) { break }
        $index++
        $progressBar.Value = 0
        $label = "[{0}/{1}] {2}" -f $index, $queue.Count, $row.Filename
        $lblProgress.Text = "${label}: connecting..."
        [System.Windows.Forms.Application]::DoEvents()

        if (-not $row.Url) {
            Write-Log "SKIP  $($row.Filename): no download URL in the manifest."
            $failed++
            continue
        }
        $dir = Split-Path $row.Dest
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

        $result = Save-File $row.Url $row.Dest $label
        if ($result -eq "ok") {
            $row.Installed = $true
            $row.Checked = $false
            $progressBar.Value = 100
            Write-Log "OK    $($row.Filename)"
        } elseif ($result -eq "cancelled") {
            Write-Log "STOP  $($row.Filename): cancelled."
        } else {
            $failed++
            Write-Log "FAIL  $($row.Filename): $result"
            $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            "[$stamp] $($row.Filename) | $($row.Url) | $result" | Out-File -FilePath $errorLog -Append -Encoding utf8
        }
    }

    $script:isDownloading = $false
    foreach ($c in @($btnInstall, $btnSelectMissing, $btnClear, $btnExit, $cmbVram, $txtSearch, $list)) { $c.Enabled = $true }
    $btnCancel.Enabled = $false
    Update-List

    if ($script:cancelRequested) {
        $lblProgress.Text = "Cancelled."
    } elseif ($failed -gt 0) {
        $lblProgress.Text = "$failed download(s) failed. Details are in the log below and in install_errors.log."
        Show-Message "$failed download(s) failed. See the log for the reason." "Warning"
    } else {
        $lblProgress.Text = "All downloads finished."
        Show-Message "All downloads finished. Restart ComfyUI or press R to refresh its model lists."
    }
    $script:cancelRequested = $false
})

$mainForm.Add_Shown({
    Update-List
    $installed = @($script:rows | Where-Object { $_.Installed }).Count
    Write-Log "$Brand ComfyUI-Cluster installer"
    Write-Log "$($script:rows.Count) files in the manifest, $installed already installed."
    $mainForm.Activate()
})

[void]$mainForm.ShowDialog()

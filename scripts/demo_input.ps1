[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,
    [string]$Profile = "steel",
    [int]$Port = 8000
)

& (Join-Path $PSScriptRoot "demo_video.ps1") -InputPath $InputPath -Profile $Profile -Port $Port

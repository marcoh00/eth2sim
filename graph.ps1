function New-TemporaryPath {
    $parent = [System.IO.Path]::GetTempPath()
    [string] $name = [System.Guid]::NewGuid()
    $path = Join-Path $parent $name
    Write-Host "$path"
    return $path
}

$filename = "$(New-TemporaryPath).pdf"

Write-Output "$filename"
dot -Tpdf -o "$filename" $args[0]
Start-Process ((Resolve-Path $filename).Path)
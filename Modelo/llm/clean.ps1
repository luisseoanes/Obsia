# Limpia carpetas y archivos generados por CMake/compilación.
# Después puedes volver a configurar con: cmake -B build -A x64
# y compilar con: cmake --build build --config Release

$root = $PSScriptRoot

Write-Host "Limpiando artefactos de compilación..." -ForegroundColor Yellow

# Carpeta build completa (donde está el .exe y todo lo generado)
if (Test-Path "$root\build") {
    Remove-Item -Path "$root\build" -Recurse -Force
    Write-Host "  Eliminado: build\" -ForegroundColor Green
}

# Generados en raíz (por si se ejecutó cmake sin -B build)
foreach ($item in @("CMakeCache.txt", "cmake_install.cmake", "CMakeFiles", "llama-build",
    "ALL_BUILD.vcxproj", "ALL_BUILD.vcxproj.filters",
    "ZERO_CHECK.vcxproj", "ZERO_CHECK.vcxproj.filters",
    "INSTALL.vcxproj", "INSTALL.vcxproj.filters", "obgyn-chatbot.slnx")) {
    $path = Join-Path $root $item
    if (Test-Path $path) {
        if (Test-Path $path -PathType Container) { Remove-Item $path -Recurse -Force }
        else { Remove-Item $path -Force }
        Write-Host "  Eliminado: $item" -ForegroundColor Green
    }
}

# Generados en app/
foreach ($item in @("CMakeFiles", "cmake_install.cmake", "INSTALL.vcxproj", "INSTALL.vcxproj.filters",
    "obgyn_chat.vcxproj", "obgyn_chat.vcxproj.filters")) {
    $path = Join-Path $root "app" $item
    if (Test-Path $path) {
        if (Test-Path $path -PathType Container) { Remove-Item $path -Recurse -Force }
        else { Remove-Item $path -Force }
        Write-Host "  Eliminado: app\$item" -ForegroundColor Green
    }
}

Write-Host "Listo. Para compilar de nuevo: cmake -B build -A x64 ; cmake --build build --config Release" -ForegroundColor Cyan

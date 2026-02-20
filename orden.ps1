# Script: ExportarEstructuraArchivos.ps1
# Descripción: Exporta la estructura completa de archivos y carpetas a un archivo txt

param(
    [string]$RutaBase = (Get-Location).Path,
    [string]$ArchivoSalida = "estructura_archivos.txt"
)

# Función para obtener la indentación según el nivel
function Get-Indentacion {
    param([int]$Nivel)
    return "  " * $Nivel
}

function Get-EstructuraCarpeta {
    param(
        [string]$Ruta,
        [int]$Nivel = 0
    )
    
    $output = @()
    $items = Get-ChildItem -Path $Ruta | Sort-Object Name
    
    foreach ($item in $items) {
        $indentacion = Get-Indentacion -Nivel $Nivel
        
        if ($item.PSIsContainer) {
            # Carpeta
            $output += "$indentacion[DIR] $($item.Name)"
            $output += Get-EstructuraCarpeta -Ruta $item.FullName -Nivel ($Nivel + 1)
        } else {
            # Archivo
            $tamano = "{0:N2} KB" -f ($item.Length / 1KB)
            $output += "$indentacion[FILE] $($item.Name) ($tamano)"
        }
    }
    
    return $output
}

# Crear el contenido del reporte
$contenido = @()
$contenido += "=" * 60
$contenido += "ESTRUCTURA DE ARCHIVOS Y CARPETAS"
$contenido += "=" * 60
$contenido += "Directorio base: $RutaBase"
$contenido += "Fecha de generacion: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$contenido += ""

# Obtener información del directorio base
$infoDirectorio = Get-Item $RutaBase
$contenido += "[DIR] $($infoDirectorio.Name) (Directorio Raiz)"
$contenido += ""

# Obtener la estructura recursiva
$estructura = Get-EstructuraCarpeta -Ruta $RutaBase
$contenido += $estructura

# Estadísticas
$contenido += ""
$contenido += "-" * 60
$archivos = Get-ChildItem -Path $RutaBase -Recurse -File
$carpetas = Get-ChildItem -Path $RutaBase -Recurse -Directory
$contenido += "ESTADISTICAS:"
$contenido += "Total de archivos: $($archivos.Count)"
$contenido += "Total de carpetas: $($carpetas.Count + 1)"  # +1 por la carpeta raíz
$contenido += "Tamaño total: {0:N2} MB" -f (($archivos | Measure-Object -Property Length -Sum).Sum / 1MB)
$contenido += "-" * 60

# Guardar en archivo
$contenido | Out-File -FilePath $ArchivoSalida -Encoding UTF8

# Mostrar resultado
Write-Host "Estructura exportada exitosamente a: $((Get-Location).Path)\$ArchivoSalida" -ForegroundColor Green
Write-Host "Resumen:" -ForegroundColor Yellow
Write-Host "   - Archivos: $($archivos.Count)" -ForegroundColor White
Write-Host "   - Carpetas: $($carpetas.Count + 1)" -ForegroundColor White
Write-Host "   - Tamaño total: {0:N2} MB" -f (($archivos | Measure-Object -Property Length -Sum).Sum / 1MB) -ForegroundColor White
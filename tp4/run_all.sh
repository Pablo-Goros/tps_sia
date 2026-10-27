#!/usr/bin/env bash
# Corre todo el TP en orden. Uso: ./run_all.sh  [--skip-search]  (salta el barrido de Kohonen)
set -euo pipefail
cd "$(dirname "$0")"
python3 ej1_eda.py
python3 ej1_pca.py
python3 ej1_kohonen.py "$@"
python3 ej1_kohonen_hp.py
python3 ej1_oja.py
python3 ej2_letras.py
python3 ej2_qr.py
python3 ej2_qr_pinv.py
python3 tests/test_teorica.py
echo "Listo. Resultados en resultados/"

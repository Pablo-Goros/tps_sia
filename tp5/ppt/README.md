# Recursos de presentación del TP5

Los PNG, SVG, tablas y registros de procedencia están en `assets/`.
`story-guide.md` describe la secuencia propuesta de diapositivas;
`generation-guide.md` explica la búsqueda de símbolos en el espacio latente.

Para regenerar los recursos, instalar las dependencias de TP5 y ejecutar desde
`tp5/` con el Python del entorno virtual:

```bash
python ppt/generate.py
python ppt/generate_search.py
python ppt/generate_story.py
python ppt/generate_targets.py
```

Los scripts utilizan las configuraciones, el dataset, el modelo final y los
resultados publicados en este repositorio. No entrenan modelos y escriben sus
salidas en `ppt/assets/`. Las referencias a archivos PowerPoint externos en las
guías documentan el material visual de origen; no son dependencias de los scripts.

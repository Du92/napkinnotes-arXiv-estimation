# arXiv + IA: contrafactual de actividad científica abierta

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución rápida

```bash
python arxiv_ai_knowledge_acceleration.py
```

Esto descarga el número total de artículos enviados a arXiv cada mes desde enero de 2015 hasta el último mes completo, usando la API oficial de arXiv. Después ajusta dos tendencias con datos hasta octubre de 2022:

- lineal;
- exponencial.

El periodo noviembre–diciembre de 2022 se visualiza, pero queda excluido tanto del ajuste como de la estimación de impacto. Desde enero de 2023 compara los datos observados con cada contrafactual.

## Ejecución con grupos disciplinares

```bash
python arxiv_ai_knowledge_acceleration.py --include-disciplines
```

Este modo solicita los metadatos mínimos de todos los artículos del intervalo para clasificar su categoría primaria. Por la política de 3 segundos entre llamadas de arXiv, puede tardar bastante. La caché permite volver a ejecutar el script sin repetir meses ya completados.

## Resultados

En `arxiv_ia_resultados/` se crean:

- `01_contrafactual_global_linear.png`
- `01_contrafactual_global_exponential.png`
- `02_exceso_mensual_exponential.png`
- `03_exceso_acumulado_exponential.png`
- `04_observado_vs_esperado_exponential.png`
- `05_comparacion_anual_exponential.png`
- CSV con la serie mensual, predicciones y métricas.
- `resumen_metricas.json`
- `informe_resumen.md`

Al usar `--include-disciplines` también se generan los gráficos por grupo y una comparación del exceso relativo.

## Nota metodológica

El resultado es una **desviación temporal respecto de una tendencia extrapolada**, no una estimación causal del “efecto de ChatGPT” ni una medición del conocimiento humano total. El número de envíos a arXiv puede cambiar por muchos factores ajenos a la IA generativa.

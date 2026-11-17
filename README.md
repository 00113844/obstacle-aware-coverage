# Fields2Cover tutorial: optimal coverage path for a paddock with trees

| File | What it is |
|---|---|
| `Fields2Cover_Paddock_Trees.ipynb` | Google Colab notebook (LaTeX-rich instructions + code). All Fields2Cover code runs in the cloud. |
| `handout/f2c_tutorial.tex` / `.pdf` | LaTeX handout that walks through the notebook: theory, TikZ schematics, what to run and what you should see, exercises, troubleshooting. |
| `tools/build_notebook.py` | Generates the notebook. Run `python tools/build_notebook.py Fields2Cover_Paddock_Trees.ipynb`. |
| `tools/nb_geom.py` | Pure numpy/shapely code embedded in the notebook: `Track`, B3 drive-around detours, section-control simulator. It can be tested without F2C. |
| `tools/check_nb.py` | Validates the notebook: schema, Python syntax, LaTeX `$` balance. |

## Run the notebook (Google Colab)
1. Open the notebook from Google Drive: right-click → *Open with* → *Google Colaboratory*.
2. Run the **Part 0** cell. It builds Fields2Cover v2.1.1 and OR-Tools v9.9 in the Colab VM, which takes about 10–15 min the first time.
   Set `USE_DRIVE_CACHE = True` to cache the build in `MyDrive/F2C_cache` so that later sessions install fast.
3. Choose *Runtime → Run all*.

Outputs: `partA_path.png`, `partB_strategies.png`, `partB_path.png`, `coverage_maps.png`, `coverage_path_trees.csv`,
`coverage_path_trees.geojson`, `scenario_comparison.csv`.

## Rebuild the handout
```
cd handout
latexmk -pdf f2c_tutorial.tex
```
Requires a TeX distribution (MiKTeX/TeX Live) with the `tikz`, `tcolorbox`, `listings` and `pifont` packages.

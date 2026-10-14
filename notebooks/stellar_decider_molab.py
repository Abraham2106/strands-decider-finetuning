import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Stellar transaction classifier

Piloto Strands Decider: **allow / escalate / deny**. El juez interviene solo al escalar; las restricciones deterministas siguen siendo independientes.

51 casos sintéticos: 33 train / 9 validation / 9 test. Los checks CPU están disponibles en GitHub. El entrenamiento GPU sigue siendo experimental; no demuestra SOTA ni fiabilidad para pagos automáticos.

Este notebook descarga revisiones fijas del entrenador, datos y código oficial. Descarga el ZIP de resultados antes de cerrar la sesión.
""")
    return


@app.cell
def _():
    import subprocess as _sp
    _gpu = _sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    print(_gpu.stdout or _gpu.stderr)
    if _gpu.returncode:
        raise RuntimeError("Selecciona RTX Pro 6000 y espera que conecte el kernel")
    return


@app.cell
def _(mo):
    run_pilot = mo.ui.run_button(label="Train and compare the pilot", kind="warn")
    run_pilot
    return (run_pilot,)


@app.cell
def _(mo, run_pilot):
    mo.stop(not run_pilot.value, mo.md("Listo: pulsa el botón para instalar dependencias, medir el modelo base y ajustar tres épocas."))
    import io as _io
    import json as _json
    import pathlib as _pathlib
    import runpy as _runpy
    import shutil as _shutil
    import subprocess as _subprocess
    import sys as _sys
    import tempfile as _tempfile
    import urllib.request as _request
    import zipfile as _zipfile

    _project_revision = "16b3bca513711cd637517a818c5de9a1ed80c046"
    _strands_revision = "ea2519129102c6482b6db4f900da0ae652399106"
    _root = _pathlib.Path(_tempfile.mkdtemp(prefix="alaia-stellar-pilot-"))
    print("Project revision:", _project_revision, flush=True)
    for _file in ("train_stellar_decider.py", "data/stellar-cases.json"):
        _url = f"https://raw.githubusercontent.com/Abraham2106/strands-decider-finetuning/{_project_revision}/{_file}"
        with _request.urlopen(_url, timeout=120) as _response:
            _bytes = _response.read()
        _dest = _root / _file
        _dest.parent.mkdir(parents=True, exist_ok=True)
        _dest.write_bytes(_bytes)

    _url = f"https://codeload.github.com/strands-labs/strands-decider/zip/{_strands_revision}"
    with _request.urlopen(_url, timeout=120) as _response:
        _source = _response.read()
    with _zipfile.ZipFile(_io.BytesIO(_source)) as _archive:
        for _member in _archive.infolist():
            if not (_root / _member.filename).resolve().is_relative_to(_root.resolve()):
                raise ValueError("Unsafe archive path")
        _archive.extractall(_root)
    _checkout = next(_root.glob("strands-decider-*"))
    print("Install pinned official Strands dependencies", flush=True)
    _subprocess.run([_sys.executable, "-m", "pip", "install", "-e", str(_checkout) + "[cuda,train]"], check=True)
    _sys.path.insert(0, str(_checkout))
    _sys.path.insert(0, str(_checkout / "src"))
    _namespace = _runpy.run_path(str(_root / "train_stellar_decider.py"), run_name="stellar_pilot")
    _cases = _json.loads((_root / "data/stellar-cases.json").read_text(encoding="utf-8"))
    print("Split checks:", _namespace["self_check"](_cases), flush=True)
    _report = _namespace["train"](_cases, _root / "run", 3)
    _report["provenance"]["project_revision"] = _project_revision
    _report["provenance"]["strands_code_commit"] = _strands_revision
    _report["provenance"]["source_data_commit"] = "9a41de6d674608f21c6531b6c85ad9e71d632a88"
    for _name, _value in (("report", _report), ("provenance", _report["provenance"])):
        (_root / "run" / f"{_name}.json").write_text(_json.dumps(_value, indent=2), encoding="utf-8")
    _zip = _shutil.make_archive(str(_root / "stellar-pilot"), "zip", _root / "run")
    mo.vstack([
        mo.md("Entrenamiento completo. Revisa las métricas antes de utilizar el clasificador."),
        mo.json(_report),
        mo.download(data=_pathlib.Path(_zip).read_bytes(), filename="stellar-pilot.zip", label="Download checkpoint and report"),
    ])
    return


if __name__ == "__main__":
    app.run()

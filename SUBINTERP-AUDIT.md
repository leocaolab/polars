# polars in own-GIL sub-interpreters: source audit

Fork: `leocaolab/polars`, branch `subinterp`, based on tag `py-1.43.2`
(commit `ae588a9`; the Rust workspace version at that tag is `0.54.4`).

Goal: run polars in CPython own-GIL sub-interpreters (PEP 684), built from
source against the per-interpreter PyO3 fork
(`leocaolab/pyo3`, branch `subinterp-per-interpreter-state`). This fork is the
version of polars we run ourselves.

This document lists, with file:line evidence, every place in the pyo3-using
crates where polars breaks the rule that makes sub-interpreters safe:

> A Python object belongs to exactly one interpreter. Anything that keeps a
> Python object beyond one call, or touches one from another thread, must
> know which interpreter it belongs to.

Crates in scope (the ones that depend on pyo3): `polars-python`,
`polars-plan`, `polars-mem-engine`, `polars-stream`, `polars-io`,
`polars-lazy`, `polars-utils`, `polars-error`, and `pyo3-polars`.

Status: audit complete for the pyo3-using crates at this tag. The CALLER-thread
caveat in P3 is an open measurement.

---

## Pitfall classes

These come from making PyO3 itself per-interpreter and from running polars
on that fork. Each one was observed, not assumed.

| class | what goes wrong | how it showed up |
|---|---|---|
| **P1** process-global Python objects | a `static` caches a Python object; the first interpreter to fill it wins, and every other interpreter gets *that* interpreter's object | `map_elements` in interpreter 2 received interpreter 1's `pl.Int64` class and failed with "cannot parse input of type 'Int64' into Polars data type (given: Int64)" |
| **P2** `intern!` / `Interned` statics | same as P1 for interned strings, which are **mortal** (see below), so their refcounts are shared across interpreters | measured on 3.12, 3.13, 3.14 |
| **P3** `Python::attach` on a foreign thread | on a thread PyO3 never attached (rayon, executor, tokio), `Python::attach` goes through `PyGILState_Ensure`, which binds the **main** interpreter | `write_csv` / `write_ipc` / `write_parquet` / `write_ndjson` to a Python file-like aborted the process (writes landed in interpreter 0) |
| **P4** PyO3's global deferred-decref pool | PyO3-level, not polars: a `Py<T>` dropped while detached is queued in one process-wide pool and later released by whichever interpreter drains it | SIGABRT within 1–4 s with ≥4 interpreters and Python callbacks |
| **P5** caches keyed by type-object address | per-interpreter heap types have different addresses; stale keys outlive their interpreter and a reused address can match the wrong entry | found in this audit (`LUT`) |
| **P6** process-global Rust state with Python-visible behaviour | not a memory-safety problem, but one interpreter's setting or signal affects all others | found in this audit |
| **P7** C / C++ dependencies | numpy (single-phase C extension) and pyarrow cannot be fixed at the PyO3 level | known |

### Measured facts this audit relies on

- **Interned strings are not immortal.** A runtime-interned string
  (`sys.intern("".join([...]))`, the same path as PyO3's `PyString::intern`)
  reports `sys._is_immortal(s) == False` on 3.14.7, in the main interpreter and
  in a sub-interpreter. On 3.12.13 and 3.13.15 its refcount is 2, so it is
  mortal there too. Sharing one across interpreters therefore shares a live
  refcount. (This corrects an earlier assumption in the PyO3 fork's notes that
  interned strings are "mostly immortal" on 3.12+.)
- **`datetime` types are shared; `decimal.Decimal` is not.** On 3.14.7,
  `id(datetime.datetime)`, `datetime.date`, `datetime.timedelta` and
  `datetime.timezone` are identical in the main interpreter and a
  sub-interpreter (static builtin types). `id(decimal.Decimal)` differs (a
  per-interpreter heap type). So PyO3's process-global datetime C-API capsule
  is **not** a problem, but anything that caches the `Decimal` type is.

---

## P1: process-global Python objects

Every `static` in scope whose type holds a Python object.

| file:line | static | holds | filled by |
|---|---|---|---|
| `polars-python/src/py_modules.rs:4` | `POLARS` | the `polars` module | first `polars(py)` call |
| `polars-python/src/py_modules.rs:5` | `POLARS_PLR` | `polars._plr` | first `polars_rs(py)` |
| `polars-python/src/py_modules.rs:6` | `UTILS` | `polars._utils` | first `pl_utils(py)` |
| `polars-python/src/py_modules.rs:7` | `SERIES` | `pl.Series` | first `pl_series(py)` |
| `polars-python/src/py_modules.rs:8` | `DATAFRAME` | `pl.DataFrame` | first `pl_df(py)` |
| `polars-python/src/on_startup.rs:107` | `WARN_FUNCTION` (`OnceLock`) | the Python warning callback | `register_startup_deps`, once per **process** (guarded by `POLARS_REGISTRY_INIT_LOCK`, `on_startup.rs:106`) |
| `polars-python/src/dataset/dataset_provider_funcs.rs:174` | `FN_POOL_WRAP_CLS` (`LazyLock`) | a Python class | first use |
| `polars-python/src/catalog/unity.rs:37-40` | `CATALOG_INFO_CLS`, `NAMESPACE_INFO_CLS`, `TABLE_INFO_CLS`, `COLUMN_INFO_CLS` | Python classes | first use |
| `polars-python/src/conversion/any_value.rs:545` | `NDARRAY_TYPE` | `numpy.ndarray` | first use |
| `polars-python/src/conversion/any_value.rs:552` | `DECIMAL_TYPE` | `decimal.Decimal` (**per-interpreter type**, see above) | first use |
| `polars-python/src/conversion/any_value.rs:181` | `LUT` (`Mutex<HashMap<TypeObjectKey, _>>`) | `Py<PyType>` inside every key | see P5 |
| `polars-plan/src/plans/optimizer/expand_datasets.rs:537` | `PY_SCAN_RESOLVE_THREADPOOL_CLS` | a Python class | first use |
| `polars-utils/src/python_convert_registry.rs:31, 55, 69, 83` | `WRAP_DF`, `CLS` ×3 | Python classes | first use |
| `pyo3-polars/pyo3-polars/src/lib.rs:59, 62` | `POLARS`, `SERIES` | the `polars` module, `pl.Series` | first use |

How `POLARS` produces the `map_elements` bug: `impl IntoPyObject for
&Wrap<DataType>` (`polars-python/src/conversion/mod.rs:217`) starts with
`let pl = polars(py)` and then `pl.getattr("Int64")` (`mod.rs:235`). In
interpreter 2 that returns interpreter 1's `Int64`.

`WARN_FUNCTION` means every polars warning, from every interpreter, calls the
first interpreter's Python function. `polars-error/src/warning.rs:4`
(`WARNING_FUNCTION`) routes into it.

## P2: interned strings

| where | count |
|---|---|
| `intern!(...)` in `polars-python` | 100 |
| `intern!(...)` in `pyo3-polars` | 47 |
| `intern!(...)` in `polars-plan` / `polars-stream` / `polars-mem-engine` / `polars-io` | 9 / 2 / 1 / 1 |
| explicit `pyo3::sync::Interned` statics, `polars-python/src/interned.rs:1-9` | 9 |

`intern!` expands to a `static Interned`, which holds a process-global
`PyOnceLock<Py<PyString>>` (PyO3 `src/sync.rs:237-246`). The fix belongs in
PyO3 (make `Interned` per-interpreter), not in 160 polars call sites.

## P3: `Python::attach` and thread context

Occurrences of `Python::attach` (there is no `Python::with_gil` at this tag):

| crate | total | FOREIGN | CALLER | UNCLEAR |
|---|---:|---:|---:|---:|
| `polars-python` | 45 | 29 | 10 | 6 |
| `polars-plan`, `polars-mem-engine`, `polars-stream`, `polars-io`, `polars-utils`, `polars-error`, `pyo3-polars` | 43 | 18 (+5 on some paths) | 17 | 2 |

FOREIGN = at least one verified caller path runs on an engine thread (rayon,
tokio, or the polars-async executor), so the attach goes through
`PyGILState_Ensure` into the **main** interpreter.

**Important caveat on CALLER.** Every polars-python entry point runs the query
inside `enter_polars*` (`polars-python/src/utils.rs:85-117`), which releases the
thread with `py.detach(...)`. In PyO3 0.29 that resets the attach count, so a
later `Python::attach`, **even on the caller's own thread**, is a fresh attach
through `PyGILState_Ensure`. Whether that returns the caller's sub-interpreter
thread state depends on how CPython bound that OS thread's gilstate slot. This
is **unverified** and must be measured before any CALLER row is trusted.

### FOREIGN sites, by feature

| feature | sites (file:line) | thread |
|---|---|---|
| Python UDFs (`map_batches`, LazyFrame `map`) | `polars-python/src/on_startup.rs:34, 38` via `polars-plan/src/dsl/python_dsl/python_udf.rs:84, 108, 175` | rayon (in-memory), executor/tokio (streaming) |
| **warnings** | `polars-python/src/on_startup.rs:90` (`warning_function`, installed at `:290`), fired by `polars_warn!` anywhere, e.g. `polars-async/src/lib.rs:42` | any |
| Object dtype | `on_startup.rs:170, 176, 184`; `conversion/mod.rs:744, 752, 761, 837` (`ObjectValue` Clone/Hash/Eq/Default) | rayon |
| plan callbacks (rolling map, `sink_batches`, group-by apply) | `on_startup.rs:195, 200, 222, 228`; `polars-plan/src/callback.rs:284` | rayon / tokio blocking |
| extension types | `polars-python/src/extension.rs:33` | reader threads |
| file-like sinks | `polars-python/src/file.rs:135, 148, 198` | tokio (`polars-io/src/utils/file.rs:296, 301`) |
| sink callbacks, file providers, Iceberg commit | `polars-plan/src/dsl/options/sink.rs:501, 529`; `file_provider.rs:165`; `on_startup.rs:190` | tokio blocking |
| Python scans / IO plugins / pyarrow datasets | `polars-stream/src/physical_plan/to_graph.rs:1511`; `io/python_dataset.rs:31` | tokio blocking |
| dataset & Delta providers | `polars-python/src/dataset/dataset_provider_funcs.rs:21, 36, 98, 175`; `delta/dv_provider_funcs.rs:35` | tokio blocking |
| cloud credential providers | `polars-io/src/cloud/credential_provider.rs:558, 609, 637, 714, 820, 861`; `cloud/options.rs:495` | tokio workers |
| parquet key-value metadata fn | `polars-io/src/parquet/write/key_value_metadata.rs:190` | tokio worker |
| `collect_with_callback` | `polars-python/src/lazyframe/general.rs:656` | tokio blocking |
| convert registry (and its lazy imports, which then cache the **main** interpreter's objects process-wide) | `polars-utils/src/python_convert_registry.rs:32, 43, 56, 70` | tokio blocking |

Python objects that are moved into engine structures and may be **dropped** on
an engine thread (so they go through PyO3's global deferred-decref pool, P4):
`PyFileLikeObject` (`file.rs:26-32`), `ObjectValue` (`conversion/mod.rs:737`),
UDF lambdas (`map/lazy.rs:38-46`), `PythonObject` wrappers in plan nodes, and
`Py` buffer owners inside arrow arrays (`file.rs:81-92`,
`series/construction.rs:51`, `interop/numpy/utils.rs:25`).

Already fixed on the PyO3-fork build of polars (`POLARS-PATCH.md` in the PyO3
fork): the five methods of `PyFileLikeObject` attach through a captured
`pyo3::sync::InterpreterHandle`.

### Thread pools used during a query

1. rayon `THREAD_POOL`, threads `polars-{i}`: `polars-core/src/runtime.rs:196-203`.
2. tokio runtime `ASYNC`: `polars-async/src/lib.rs:36-49` (workers `min(max_threads, 32)`, up to 512 blocking threads).
3. polars-async executor `GLOBAL_SCHEDULER`, threads `async-executor-{t}`: `polars-async/src/executor/mod.rs:47, 399-406`.

`RAYON.block_on` (`polars-core/src/runtime.rs:153-190`) moves work from a rayon
thread to the tokio blocking pool.

## P5: caches keyed by type-object address

`polars-python/src/conversion/any_value.rs:142-181`: `LUT` maps
`TypeObjectKey { type_object: Py<PyType>, address: usize }` to a conversion
function, in one process-global `Mutex<HashMap<…>>`.

- For static builtin types (`int`, `str`, `datetime`, …) the address is the
  same in every interpreter, so entries are shared correctly.
- For per-interpreter heap types (`decimal.Decimal`, user classes) each
  interpreter adds its own entry, and the `Py<PyType>` keeps that
  interpreter's type alive after the interpreter is gone.
- After an interpreter is destroyed, a new type in another interpreter can be
  allocated at the same address and match the stale key, selecting the wrong
  conversion function: **silently wrong data**.

## P6: process-global Rust state with Python-visible behaviour

Memory-safe, but shared by every interpreter in the process:

| file:line | state | effect across interpreters |
|---|---|---|
| `polars-core/src/runtime.rs:196/206` | `THREAD_POOL` (one rayon pool) | all interpreters share the worker threads; Python callbacks from it hit P3 |
| `polars-core/src/fmt.rs:46-50` | float precision / format, thousands and decimal separators | `pl.Config` in one interpreter changes printing in all |
| `polars-core/src/random.rs:5` | `POLARS_GLOBAL_RNG_STATE` | seeding in one interpreter affects the others |
| `polars-error/src/abort.rs:27` | `ABORT_STATE` | an abort / KeyboardInterrupt raised for one interpreter's query can affect others |
| `polars-python/src/timeout.rs:13, 60-85` | timeout thread | when `POLARS_TIMEOUT_MS` expires it calls `std::process::exit(1)`: one interpreter's query ends the whole process |
| `polars-core/src/chunked_array/object/registry.rs:48` | `GLOBAL_OBJECT_REGISTRY` | registered once per process (`on_startup.rs`) |
| `polars-core/src/datatypes/extension/registry.rs:88` | extension-type registry | shared registrations |
| environment variables (`POLARS_*`, read throughout, e.g. `polars-core/src/fmt.rs:738`) | process environment | shared by nature: `os.environ` is one per process in CPython |

## P7: C / C++ dependencies

- `numpy` interop (`polars-python/src/interop/numpy/*`, `series/numpy_ufunc.rs`,
  `series/construction.rs`) via the rust-numpy crate. numpy is a single-phase C
  extension: it does not load in a strict own-GIL sub-interpreter at all, so
  these paths are unavailable there (they fail at import, cleanly).
- `pyarrow` interop: a C++ library with its own process state; out of scope.

## Verified non-issues

- PyO3's process-global datetime C-API capsule: the types it points to are
  static builtin types, identical in every interpreter (measured above).
- `CALL_PYTHON_COLUMNS_UDF`, `CALL_PYTHON_DF_UDF`
  (`polars-plan/src/dsl/python_dsl/python_udf.rs:18, 21`),
  `DATASET_PROVIDER_VTABLE`, `DELTA_DV_PROVIDER_VTABLE`: function-pointer
  tables, not Python objects. (What those functions do when called is P3.)
- Plugin library registry (`polars-plan/.../plugin.rs:13`), regex cache
  (`thread_local`), CPU-feature caches, allocator: plain Rust data.

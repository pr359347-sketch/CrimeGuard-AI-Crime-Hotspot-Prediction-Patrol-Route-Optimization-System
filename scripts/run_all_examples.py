import runpy
from pathlib import Path

# Resolve repository root robustly by walking parents and looking for the examples folder
here = Path(__file__).resolve()
repo_root = None
# First try: walk parents and look directly for 01.app/examples
for p in [here.parent, *here.parents]:
    if (p / '01.app' / 'examples').exists():
        repo_root = p
        break

# Second try: some filesystems show multiple names for the same folder
# so scan siblings of each parent for a directory that contains 01.app
if repo_root is None:
    for p in [here.parent, *here.parents]:
        try:
            for child in p.iterdir():
                if (child / '01.app' / 'examples').exists():
                    repo_root = child
                    break
        except Exception:
            pass
        if repo_root is not None:
            break

if repo_root is None:
    repo_root = here.parent.parent
    # Final fallback: search upward tree for any '01.app' directory and use its parent
    try:
        top_search = here.parent.parent
        found = list(top_search.rglob('01.app'))
        if found:
            repo_root = found[0].parent
    except Exception:
        pass

examples = [
    '01.app/examples/example_basic_usage.py',
    '01.app/examples/example_route_optimization.py',
    '01.app/examples/example_model_evaluation.py',
]

log_path = repo_root / 'scripts' / 'run_all_examples.log'
with open(log_path, 'w', encoding='utf-8') as log:
    log.write(f'Working dir: {repo_root}\n')
    log.write(f'resolved __file__ -> {here}\n')
    log.write(f'repo_root -> {repo_root}\n')
    for example in examples:
        log.write('\n=== RUN ' + example + ' ===\n')
        candidate = repo_root / example
        if not candidate.exists():
            # try to locate the example anywhere under repo_root
            found = list(repo_root.rglob(Path(example).name))
            if found:
                candidate = found[0]
                log.write(f'NOTE: example not at expected path, using {candidate}\n')
            else:
                log.write(f'FAIL: FileNotFoundError: {example} not found under {repo_root}\n')
                continue

        try:
            runpy.run_path(str(candidate), run_name='__main__')
            log.write('SUCCESS\n')
        except Exception as exc:
            log.write(f'FAIL: {type(exc).__name__}: {exc}\n')
            import traceback
            traceback.print_exc(file=log)
    log.write('\nDONE\n')

print('Log written to', log_path)

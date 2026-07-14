import os, runpy
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
examples = [
    '01.app/examples/example_basic_usage.py',
    '01.app/examples/example_route_optimization.py',
    '01.app/examples/example_model_evaluation.py',
]
log_path = os.path.join(root, 'scripts', 'run_all_examples.log')
with open(log_path, 'w', encoding='utf-8') as log:
    log.write(f'Working dir: {root}
')
    for example in examples:
        path = os.path.join(root, example)
        log.write('
=== RUN ' + example + ' ===
')
        try:
            runpy.run_path(path, run_name='__main__')
            log.write('SUCCESS
')
        except Exception as exc:
            import traceback
            traceback.print_exc(file=log)
            log.write(f'FAIL: {type(exc).__name__}: {exc}
')
print('Log written to', log_path)

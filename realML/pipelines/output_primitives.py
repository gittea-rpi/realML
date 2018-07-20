# helper class from Michigan's SPIDER pipeline examples, originally at
# https://gitlab.datadrivendiscovery.org/michigan/spider/blob/master/spider/pipelines/output_primitive.py

# Output pipelines in JSON.

import argparse
import json
import os
import sys

import realML.pipelines.all

def load_args():
    parser = argparse.ArgumentParser(description = "Output a primitive's pipelines in the standard format.")

    parser.add_argument(
        'primitive', action = 'store', metavar = 'PRIMITIVE',
        help = "the D3M entrypoint of the primitive whose pipelines will be output"
    )

    parser.add_argument(
        'outdir', action = 'store', metavar = 'OUTPUT_DIR',
        help = "where to write the file"
    )

    parser.add_argument(
        '-o', '--one', action = "store_true",
        help = "only generate at most one pipeline"
    )

    arguments = parser.parse_args()

    return arguments.primitive, os.path.abspath(arguments.outdir), arguments.one

def main():
    primitive, out_dir, only_one = load_args()

    if (primitive not in realML.pipelines.all.get_primitives()):
        print("Could not locate pipelines for primitive: %s." % (primitive), file = sys.stderr)
        return

    os.makedirs(out_dir, exist_ok = True)

    for pipeline_class in realML.pipelines.all.get_pipelines(primitive):
        datasets = pipeline_class().get_datasets()

        for dataset in datasets:
            pipeline = pipeline_class()

            out_path = os.path.join(out_dir, "%s.json" % (pipeline.get_id()))
            with open(out_path, 'w') as file:
                file.write(pipeline.get_json())

            meta_path = os.path.join(out_dir, "%s.meta" % (pipeline.get_id()))
            meta_info = {
                'problem': realML.pipelines.datasets.get_problem_id(dataset),
                'train_inputs': [ realML.pipelines.datasets.get_train_id(dataset) ],
                'test_inputs': [ realML.pipelines.datasets.get_problem_id(dataset) ],
            }
            with open(meta_path, 'w') as file:
                json.dump(meta_info, file, indent = 4)

            if (only_one):
                break

if __name__ == '__main__':
    main()

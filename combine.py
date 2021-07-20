import os, json, argparse

# Getting input parameters

ap = argparse.ArgumentParser()
req = ap.add_argument_group('required arguments')
req.add_argument(
    '-c', '--configs', type=str, nargs='+', required=True,
    help='Give a list of configuration file names.'
)
req.add_argument(
    '-n', '--name', type=str, required=True,
    help='Give the name of the resulting configurations file.'
)

args = vars(ap.parse_args())

# Load configuration files
config = []
for i in args['configs']:
    with open(os.path.join('input', 'configurations', i), 'r') as file:
        config = config + json.load(file)

# Save union of configurations as new file
with open(os.path.join('input', 'configurations', args['name']), 'w') as file:
    json.dump(config, file)
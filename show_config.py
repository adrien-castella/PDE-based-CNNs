import os, json, argparse

# Getting input parameters
ap = argparse.ArgumentParser()
req = ap.add_argument_group('required arguments')
ap.add_argument(
    '-n', '--name', type=str,
    help='Name of the configuration.'
)
args = vars(ap.parse_args())

# List available configuration file or print the chosen one
if args['name'] == None:
    print(json.dumps(os.listdir(os.path.join('input', 'configurations')), indent=2))
else:
    with open(os.path.join('input', 'configurations', args['name']), 'r') as file:
        print(json.dumps(json.load(file), indent=2))
import subprocess

# Exit based on softmax thresholds between 0 and 1, in steps of 0.05.
for i in range(21):
    threshold = i * 0.05
    with open('./scripts/softmax_threshold.sh', 'r') as fopen:
        s = fopen.read()
    s = s.replace('PLACEHOLDER', str(threshold))
    subprocess.run(s, shell=True)

# Static layer exiting.
for i in range(1, 7):
    with open('./scripts/static_layer.sh', 'r') as fopen:
        s = fopen.read()
    s = s.replace('PLACEHOLDER', str(i))
    subprocess.run(s, shell=True)
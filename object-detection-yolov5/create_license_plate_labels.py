labels = []

labels.append('')

for i in range(10):
    labels.append(str(i))

for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    labels.append(c)

special_chars = ['-', ' ', '.']
for char in special_chars:
    labels.append(char)

while len(labels) < 525: # pad to 525
    labels.append('')

with open('labels_license_plate.txt', 'w') as f:
    for label in labels:
        f.write(label + '\n')

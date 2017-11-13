Die Struktur der Daten ist identisch. Die Bilder liegen wieder abgerollt in einer Zeile. Diesmal jedoch nach Farbkanälen geordnet. Es ergibt sich nun also für trainingsDaten.npz die Form (60,3072), da die 60 Bilder alle 32x32 Pixel und je drei Farbkanäle haben. Die ersten 1024 Pixel jeder Zeile sind dabei die Rot-Werte, die nächsten 1024 die Grün-Werte und die letzte 1024 die Blau-Werte

Laden der Daten:
d = np.load('./trainingsDaten.npz')
trImgs = d['data']
trLabels = d['labels']

Kanäle ansteuern:
trImgs[0,:1024] #-> Rot-Kanal des ersten Bildes
trImgs[0,1024:2048] #-> Blau-Kanal des ersten Bildes
trImgs[0,2048:] #-> Grün-Kanal des ersten Bildes


Die beiden Dateien trainingsDaten.npz und validierungsDaten.npz beinhalten jeweils einen dict. Jeder dieser dicts hat einen Schlüssel 'data' und einen Schlüssel 'labels'. Unter dem Schlüssel 'data' sind die Bilder als 2D-Array gespeichert. Jede Zeile entspricht dabei einem Bild, jede Spalte einem Pixel bzw. dessen Grauwert. Entsprechend hat das 'data'-Array in trainingsDaten.npz die Form (60,1024), da die 60 Bilder alle 32x32 Pixel haben. Unter dem Schlüssel 'labels' findet sich ein 1D-Array, dass für jedes Bild das korrekte Label, also die korrekte Klassenzuordnung, vorhält.

Laden der Daten:
d = np.load('./trainingsDaten.npz')
trImgs = d['data']
trLabels = d['labels']

Erstes Bild und Label auslesen:
img1 = trImgs[0,:] #-> [180 178 186 ...,  77  73  75]
label1 = trLabels[0] #-> 1

Label - Objektklasse
1 - Auto
4 - Hirsch
8 - Schiff

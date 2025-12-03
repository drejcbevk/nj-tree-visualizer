# Neighbor-Joining Tree Visualizer

Orodje za izračun in interaktivno vizualizacijo Neighbor-Joining (NJ) filogenetskih dreves iz poljubnih razdaljskih matrik ali Orange3 podatkovnih tabel.

- lasten **NJ algoritmični modul**
- **Orange3 distance matrike** (Euclidean, Manhattan, Hamming)
- interaktivno vizualizacijo:  
  **drsnik za rezanje drevesa**, barvne gruče, hover-highlight, linijska debla, etikete


## Funkcionalnosti

- Implementacija **neighbor_joining_core** brez zunanjih odvisnosti
- Pretvorniki: `neighbor_joining_orange()`, `to_newick()`
- Interaktivni prikaz z barvanjem gruč in poudarjanjem z miško
- Demo vizualizacije: Iris, Zoo, Housing in sintetične matrike

## Struktura repozitorija

Projekt je organiziran tako, da jasno loči:

- **glavno implementacijo algoritma**
- **prikaz in interaktivne elemente**
- **povezave z Orange3**
- **primere uporabe (demos)**

### Glavne mape

repo/
│
├── src/
│ ├── nj_core.py
│ ├── nj_orange.py
│ └── nj_visualization.py
│
├── demos/
│ └── demo_orange_datasets.py
│
├── requirements.txt
├── .gitignore
└── README.md

### Opis datotek

#### `src/`
Modularni del projekta, kjer se nahaja celotna logika.

- **`nj_core.py`**  
  Čista implementacija Neighbor-Joining algoritma.  
  Vsebuje:
  - `TreeNode` razred  
  - `neighbor_joining_core()`  
  - pretvorbe (npr. `to_newick()`)  
  - preverjanje konsistentnosti v primerjavi z BioPython / scikit-bio (če vključeno)

- **`nj_orange.py`**  
  Povezovalni modul za uporabo Orange3 distance matrik (`DistMatrix`).  
  Vsebuje:
  - `neighbor_joining_orange(dm, labels)`  
  - obravnavo Orange datasetov

- **`nj_visualization.py`**  
  Interaktivni prikaz NJ drevesa z Matplotlib.  
  Funkcionalnosti:
  - drsnik za rezanje drevesa po razdalji  
  - barvanje gruč  
  - hover highlight (debelejši robovi, črne oznake)  
  - dinamično posodabljanje prikaza  
  - demo vizualizacija (20×20 sintetična matrika)

#### `demos/`
Primeri uporabe, ločeni od implementacije.

- **`demo_orange_datasets.py`**  
  Trije praktični prikazi:
  - Iris: 15 primerkov iz vsakega razreda, evklidske razdalje  
  - Zoo: Manhattan razdalje, oznake iz meta-polja *name*  
  - Housing: prvih 60 vrstic, evklidske razdalje  

  Vsak primer:
  1. pripravi podatke  
  2. izračuna razdalje (Euclidean / Manhattan / Hamming)  
  3. generira NJ drevo  
  4. prikaže interaktivno vizualizacijo  

#### Ostalo

- **`requirements.txt`**  
  Seznam odvisnosti (numpy, matplotlib, Orange3, biopython, scikit-bio …)

- **`.gitignore`**  
  Izključitev virtualnega okolja, cache map in build artefaktov.

- **`README.md`**  
  Opis projekta, navodila za namestitev in zagon.

## Namestitev

```bash
pip install -r requirements.txt

"""
Analisi esplorativa dei dataset (per la tesina).
Output: tabella a schermo, analisi_esplorativa.csv e il grafico lunghezza_tracce.png.
"""
from os import listdir
from os.path import exists, join

import matplotlib.pyplot as plt
import pandas as pd
import torch

DATASET = ['Receipt', 'Helpdesk_no_resources', 'BPI20R', 'BPI20P', 'BPI12WC']
ARTIFICIALI = {'artificial_start', 'artificial_end'}

righe, lunghezze = [], {}
for dataset in DATASET:
    file_g = join('dataset', f'{dataset}_processed.g')
    if not exists(file_g):
        print(f'{file_g} non trovato: dataset saltato')
        continue

    # ---------- tracce (file .g) ----------
    df = pd.read_csv(file_g)
    nodi = df[df['type'] == 'v']
    archi = df[df['type'] == 'e']
    eventi = nodi[~nodi['activity'].isin(ARTIFICIALI)]

    lung = eventi.groupby('case_id').size()                       # eventi reali per traccia
    lunghezze[dataset] = lung.values

    # varianti: sequenze di attività distinte (nodi in ordine di indice = ordine cronologico)
    varianti = (eventi.sort_values(['case_id', 'node1'])
                .groupby('case_id')['activity'].apply(tuple).nunique())

    # attività parallele: eventi reali con più di un arco entrante (artificial_end escluso)
    entranti = archi.groupby(['case_id', 'node2']).size().rename('n_entranti').reset_index()
    entranti = entranti.merge(eventi[['case_id', 'node1']], left_on=['case_id', 'node2'],
                              right_on=['case_id', 'node1'])
    paralleli = 100 * (entranti['n_entranti'] > 1).sum() / len(eventi)

    # insieme di ogni traccia
    insiemi = nodi.drop_duplicates('case_id')['set'].value_counts()

    riga = {
        'dataset': dataset,
        'tracce': lung.size,
        'eventi': len(eventi),
        'attivita': eventi['activity'].nunique(),
        'varianti': varianti,
        'lung_min': lung.min(),
        'lung_max': lung.max(),
        'lung_media': round(lung.mean(), 1),
        'lung_mediana': lung.median(),
        'nodi_paralleli_%': round(paralleli, 1),
        'tracce_train': insiemi.get('train', 0),
        'tracce_val': insiemi.get('validation', 0),
        'tracce_test': insiemi.get('test', 0),
        'org_resource': 'org_resource' in df.columns,
    }

    # ---------- prefissi (file .pt) ----------
    cartella = join('dataset', f'{dataset}_tensors')
    if exists(cartella):
        prefissi = [torch.load(join(cartella, f), weights_only=False)
                    for f in listdir(cartella) if f.endswith('.pt')]
        set_pref = pd.Series([p.set for p in prefissi]).value_counts()
        target = pd.Series([p.next_activity for p in prefissi]).value_counts()
        riga.update({
            'prefissi': len(prefissi),
            'pref_train': set_pref.get('train', 0),
            'pref_val': set_pref.get('validation', 0),
            'pref_test': set_pref.get('test', 0),
            'classi': target.size,
            'classe_piu_frequente_%': round(100 * target.iloc[0] / target.sum(), 1),
            'classe_meno_frequente': int(target.iloc[-1]),
        })
    righe.append(riga)

tabella = pd.DataFrame(righe)
pd.set_option('display.width', 250)
print(tabella.T.to_string(header=False))
tabella.to_csv('analisi_esplorativa.csv', index=False)
print('\nTabella salvata in analisi_esplorativa.csv')

# ---------- grafico: distribuzione della lunghezza delle tracce ----------
if lunghezze:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(list(lunghezze.values()), showfliers=True)
    ax.set_xticks(range(1, len(lunghezze) + 1), list(lunghezze.keys()))
    ax.set_yscale('log')
    ax.set(title='Lunghezza delle tracce (numero di eventi)', ylabel='Eventi per traccia (scala log.)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig('lunghezza_tracce.png', dpi=150)
    print('Grafico salvato in lunghezza_tracce.png')

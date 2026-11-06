from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score


DATASET = [
    'BPI12WC',
    'BPI20P',
    'BPI20R',
    'Helpdesk_no_resources',
    'Receipt',
]
MODELLI = {
    'GraphSAGE': 'models_sage',
    'GATv2': 'models_gatv2',
    'GGNN': 'models_ggnn',
    'GCN': 'models_gcn',
}

# Posizione della legenda in ogni grafico (scelta per non coprire le curve)
POSIZIONE_LEGENDA = {
    'Receipt': dict(loc='upper left', bbox_to_anchor=(0.0, 0.88)),
    'BPI20P': dict(loc='upper left', bbox_to_anchor=(0.0, 0.88)),
    'Helpdesk_no_resources': dict(loc='lower right'),
    'BPI20R': dict(loc='lower right'),
    'BPI12WC': dict(loc='lower left'),
}

tabelle = []
for dataset in DATASET:
    risultati_per_modello = {}
    for nome_modello, cartella in MODELLI.items():
        file_metriche = Path(cartella) / f'{dataset}_metrics.csv'
        metriche = pd.read_csv(file_metriche)
        trial = int(metriche.loc[metriche['val'].idxmin(), 'trial'])

        file_risultati = Path(cartella) / dataset / f'prefix_results_it_{trial}.csv'
        risultati = pd.read_csv(file_risultati)
        colonne_richieste = {'prefix_size', 'prediction', 'label'}
        colonne_mancanti = colonne_richieste.difference(risultati.columns)
        if colonne_mancanti:
            raise ValueError(
                f'{file_risultati} non contiene le colonne richieste: '
                f'{sorted(colonne_mancanti)}'
            )

        righe = []
        for lunghezza, gruppo in risultati.groupby('prefix_size'):
            f1 = f1_score(
                gruppo['label'],
                gruppo['prediction'],
                average='macro',
                zero_division=0,
            )
            righe.append({
                'dataset': dataset,
                'modello': nome_modello,
                'lunghezza_prefisso': int(lunghezza),
                'numero_prefissi': len(gruppo),
                'f1_macro': f1 * 100,
            })

        risultati_per_modello[nome_modello] = pd.DataFrame(righe)

    tabella = pd.concat(risultati_per_modello.values(), ignore_index=True)
    tabelle.append(tabella)
    print(f'\n=== {dataset} ===')
    print(tabella.to_string(index=False, formatters={'f1_macro': '{:.2f}'.format}))

    # ---------- grafico ----------
    fig, ax = plt.subplots(figsize=(10, 6))

    # asse sinistro: F1 macro di ogni modello
    for nome_modello, risultati in risultati_per_modello.items():
        risultati = risultati.sort_values('lunghezza_prefisso')
        ax.plot(
            risultati['lunghezza_prefisso'],
            risultati['f1_macro'],
            marker='o',
            markersize=4,
            linewidth=2,
            label=nome_modello,
        )

    ax.set(
        title=f'F1 macro per lunghezza del prefisso - {dataset}',
        xlabel='Lunghezza del prefisso (numero di nodi)',
        ylabel='F1 macro (%)',
        ylim=(0, 105),
    )
    ax.grid(True, linestyle='--', alpha=0.5)

    # asse destro: numero di prefissi per lunghezza (uguale per tutti i modelli)
    numero = (tabella.drop_duplicates('lunghezza_prefisso')
              .sort_values('lunghezza_prefisso'))
    ax2 = ax.twinx()
    ax2.plot(
        numero['lunghezza_prefisso'],
        numero['numero_prefissi'],
        linestyle='--',
        color='black',
        linewidth=1.5,
        label='Numero di prefissi',
    )
    ax2.set_ylabel('Numero di prefissi nel test set')
    ax2.set_ylim(0, numero['numero_prefissi'].max() * 1.05)

    # una sola legenda per le due curve, disegnata sopra le linee
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, framealpha=0.95,
               **POSIZIONE_LEGENDA.get(dataset, dict(loc='upper left')))

    fig.tight_layout()
    file_plot = f'f1_macro_per_lunghezza_{dataset}.png'
    fig.savefig(file_plot, dpi=150)
    plt.close(fig)
    print(f'Grafico salvato in: {file_plot}')

file_csv = 'f1_macro_per_lunghezza_tutti_dataset.csv'
pd.concat(tabelle, ignore_index=True).to_csv(file_csv, index=False)
print(f'\nTabella completa salvata in: {file_csv}')

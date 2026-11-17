"""Visualize the historical factorial search from its saved configuration."""
from pathlib import Path
from itertools import product
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

TP=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'assets'
OUT.mkdir(exist_ok=True)
config=json.loads((TP/'configs/comparison-broad.json').read_text())
architectures=[(35,20,2,20,35),(35,40,2,40,35),(35,60,2,60,35),
               (35,30,15,2,15,30,35),(35,40,20,10,2,10,20,40,35)]
rates={}
for v in config['variants']:
    rates.setdefault(v['optimizer']['name'],set()).add(v['optimizer']['learning_rate'])
actual={(tuple(v['architecture']),v['optimizer']['name'],v['optimizer']['learning_rate']) for v in config['variants']}
expected={(a,o,r) for a,o in product(architectures,rates) for r in rates[o]}
assert len(actual)==len(config['variants'])==60 and actual==expected
assert config['seeds']==[42,43,44]
assert all(v['batch_size']==32 and v['max_epochs']==3000 and v['evaluate_every']==100 and v['stop_on_tolerance'] for v in config['variants'])
INK='#26384a';BLUE='#368dc4';TEAL='#008f86';GRAY='#9aa9b8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'text.color':INK,'axes.labelcolor':INK,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
def save(fig,name):
    for ext in ('png','svg'):
        fig.savefig(OUT/(name+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
def decimal(v):return str(v).replace('.',',')

fig,ax=plt.subplots(figsize=(16,8))
ax.set(xlim=(0,16),ylim=(0,8));ax.axis('off')
for x,w in [(0.1,7.6),(8.3,7.6)]:
    ax.add_patch(FancyBboxPatch((x,2.4),w,5.3,boxstyle='round,pad=0.02',facecolor='#f2f6f8',edgecolor='none'))
ax.text(.4,7.2,'5 ARQUITECTURAS',fontsize=22,weight='bold',color=BLUE)
descriptions=['Base','Más ancha','Todavía más ancha','Más profunda','Mayor profundidad']
for i,(desc,a) in enumerate(zip(descriptions,architectures)):
    y=6.45-i*.78
    ax.text(.4,y,desc,fontsize=13,color=BLUE)
    ax.text(.4,y-.33,' → '.join(map(str,a)),fontsize=15,weight='bold')
ax.text(8,4.8,'×',fontsize=32,ha='center',va='center')
ax.text(8.6,7.2,'4 OPTIMIZADORES · 3 TASAS CADA UNO',fontsize=19,weight='bold',color=TEAL)
for i,(key,title) in enumerate([('adam','Adam'),('rmsprop','RMSProp'),('momentum','Momentum'),('sgd','SGD')]):
    y=6.35-i*.9
    ax.text(8.7,y,title,fontsize=18,weight='bold')
    ax.text(11,y,' / '.join(decimal(r) for r in sorted(rates[key])),fontsize=17)
ax.text(8.7,2.8,'Cada arquitectura se cruza con las 12 opciones.',fontsize=14,color=TEAL)
ax.text(8,1.65,'60 configuraciones × 3 semillas (42, 43, 44) = 180 corridas',ha='center',fontsize=23,weight='bold')
ax.text(8,.8,'Pérdida: cuadrática · Activación interna y latente: tanh · Salida: sigmoide',ha='center',fontsize=17)
ax.text(8,.25,'32 patrones · lote 32 · hasta 3.000 épocas · tolerancia: ≤ 1 píxel por carácter',ha='center',fontsize=16)
save(fig,'story-search-design')

# Recorded historical aggregates; do not imply the missing models were rerun.
values=[4,4.3333,4.6667,2,1.6667]
best=['Adam · 0,002']*3+['RMSProp · 0,002']*2
fig,ax=plt.subplots(figsize=(13,6.4))
labels=[f'{desc}\n'+ ' → '.join(map(str,a)) for desc,a in zip(descriptions,architectures)]
DATASET_BLUE='#08306b'  # Endpoint of the Blues colormap used for the dataset.
bars=ax.barh(range(5),values,color=DATASET_BLUE,height=.57)
for i,(value,opt) in enumerate(zip(values,best)):
    ax.text(value+.10,i,f'{value:.2f}'.replace('.',',')+' px',va='center',fontsize=15,weight='bold')
    ax.text(.12,i,opt,va='center',fontsize=13,color='white',weight='bold')
ax.set(yticks=range(5),yticklabels=labels,xlim=(0,5.65),xlabel='Media del error máximo por carácter (3 semillas)')
ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
save(fig,'story-search-results')

# Presentation labels 1–3 correspond to historical RNG seeds 42–44.
fig,ax=plt.subplots(figsize=(10,3.1))
ax.axis('off')
table=ax.table(cellText=[['1','2 píxeles','No'],['2','2 píxeles','No'],['3','1 píxel','Sí']],
               colLabels=['Semilla','Máximo error por carácter','¿Cumple?'],
               colWidths=[.2,.56,.24],cellLoc='center',loc='center',bbox=[0,0,1,1])
table.auto_set_font_size(False)
table.set_fontsize(17)
for (row,col),cell in table.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_linewidth(2)
    if row==0:
        cell.set_facecolor(DATASET_BLUE)
        cell.set_text_props(color='white',weight='bold')
    else:
        cell.set_facecolor('#edf4fb' if row%2 else '#f7faff')
        cell.set_text_props(color=DATASET_BLUE,weight='bold' if col==2 else 'normal')
save(fig,'story-selected-seeds-table')
(OUT/'search-provenance.json').write_text(json.dumps({'design_source':'tps_sia/tp5/configs/comparison-broad.json','architectures':architectures,'optimizer_rates':{o:sorted(r) for o,r in rates.items()},'seeds':config['seeds'],'configurations':60,'runs':180,'full_cross_verified':True,'result_source':'tps_sia/tp5/comparison.md, Criterio y resultados','best_variant_mean_max_pixel_errors':values,'historical_models_recomputed':False},ensure_ascii=False,indent=2),encoding='utf-8')
print('Generated search design, results and seed table in PNG/SVG; verified all 60 combinations and 180 planned runs.')

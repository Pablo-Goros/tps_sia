"""Generate loss, learning-rate and optimizer figures; no model training."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent/'assets'
OUT.mkdir(exist_ok=True)
TP=OUT.parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#26384a','axes.labelcolor':'#26384a','savefig.facecolor':'white'})
TEAL='#008f86'; BLUE='#368dc4'; GRAY='#9aa9b8'
def save(fig,name):
    for ext in ('png','svg'):
        fig.savefig(OUT/(name+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)

# Historical results are transcribed from comparison.md because the original
# run directories are unavailable in this checkout. Do not imply recomputation.
fig,ax=plt.subplots(figsize=(9,4.7))
bars=ax.bar(['0,0015','0,002\n(referencia)','0,0025'],[0,1,1],color=[GRAY,TEAL,BLUE],width=.55)
ax.bar_label(bars,labels=['0/3','1/3','1/3'],padding=7,fontsize=18)
ax.set(xlabel='Tasa de aprendizaje de RMSProp',ylabel='Corridas que cumplen la tolerancia',ylim=(0,3.4),yticks=range(4))
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'story-06-learning-rate')

p=np.linspace(.0001,.9999,1500)
quadratic=(1-p)*p*(1-p)
bce=1-p
assert np.isclose((1-.01)*.01*(1-.01),.009801)
assert np.isclose((1-.01)/((1-.01)*.01*(1-.01)),1/(.01*.99))
fig,ax=plt.subplots(figsize=(10,5.4))
ax.plot(p,quadratic,color=GRAY,lw=3,label='Cuadrática + sigmoide')
ax.plot(p,bce,color=TEAL,lw=3,label='BCE + sigmoide')
ax.scatter([.01,.01],[.009801,.99],color=[GRAY,TEAL],s=60,zorder=5)
ax.annotate('Píxel esperado: 1\nPredicción: 0,01\n\nCuadrática: 0,0098\nBCE: 0,99',xy=(.01,.99),xytext=(.35,.55),fontsize=13,bbox=dict(boxstyle='round,pad=.7',fc='#f2f6f8',ec='none'),arrowprops=dict(arrowstyle='->',color=TEAL))
ax.set(xlabel='Salida de la sigmoide p (píxel esperado: y = 1)',ylabel='Magnitud del gradiente respecto del logit u',xlim=(-.02,1.02),ylim=(-.035,1.08))
ax.legend(loc='upper right',frameon=False,fontsize=12);ax.grid(alpha=.15)
save(fig,'story-07-output-gradient')

summary=json.loads((TP/'results/presentation-campaign-2026-11-10/01-loss-latent/summary.json').read_text())
# Use aggregates directly and require unambiguous identification of controls.
groups=summary['aggregates']
control=next(g for g in groups if g['variant']=='reference')
chosen=next(g for g in groups if g['variant']=='bce-tanh')
assert control['success_fraction']==1/3 and chosen['success_fraction']==1
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,key,label in [(axes[0],'success_fraction','Corridas que cumplen (de 3)'),(axes[1],'mean_pixel_errors_mean','Píxeles incorrectos por carácter (media)')]:
    vals=[g[key]*(3 if key=='success_fraction' else 1) for g in [control,chosen]]
    bars=ax.bar(['Cuadrática','BCE'],vals,color='#08306b',width=.55)
    ax.bar_label(bars,labels=['1/3','3/3'] if key=='success_fraction' else [f'{v:.3f}'.replace('.',',') for v in vals],padding=7,fontsize=17)
    ax.set_ylabel(label);ax.set_ylim(0,max(vals)*1.3);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_yticks(range(4));fig.tight_layout(w_pad=3)
save(fig,'story-08-loss-result')
(OUT/'story-provenance.json').write_text(json.dumps({'learning_rate':{'source':'tps_sia/tp5/comparison.md, Tasas fijas cercanas de RMSProp','values':[{'lr':.0015,'successes':0,'runs':3},{'lr':.002,'successes':1,'runs':3},{'lr':.0025,'successes':1,'runs':3}],'historical_artifacts_available':False},'gradient':{'kind':'analytical illustration, not measured training gradients','output':'p=sigmoid(u), u=2*beta*h','quadratic':'L=0.5*(p-y)^2; dL/du=(p-y)*p*(1-p)','bce':'dL/du=p-y','target':1},'loss_result':{'source':'tps_sia/tp5/results/presentation-campaign-2026-11-10/01-loss-latent/summary.json','control':control,'bce':chosen}},ensure_ascii=False,indent=2),encoding='utf-8')
optimizer_summary=json.loads((TP/'results/presentation-campaign-2026-11-10/02-optimizer/summary.json').read_text())
optimizer_groups=[next(g for g in optimizer_summary['aggregates'] if g['variant']==v)
                  for v in ('bce-tanh','bce-tanh-adam')]
assert all(g['runs']==3 and g['success_fraction']==1 for g in optimizer_groups)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
for ax,key,ylabel in [(axes[0],'success_fraction','Corridas dentro de tolerancia (de 3)'),
                      (axes[1],'max_pixel_errors_mean','Media del error máximo por carácter (píxeles)')]:
    vals=[g[key]*(3 if key=='success_fraction' else 1) for g in optimizer_groups]
    bars=ax.bar(['RMSProp','Adam'],vals,color='#08306b',width=.55)
    labels=['3/3','3/3'] if key=='success_fraction' else [f'{v:.2f}'.replace('.',',') for v in vals]
    ax.bar_label(bars,labels=labels,padding=8,fontsize=19,weight='normal',color='#08306b')
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_ylim(0,max(vals)*1.32)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_yticks(range(4))
fig.tight_layout(w_pad=3)
save(fig,'story-optimizer-comparison')
(OUT/'optimizer-provenance.json').write_text(json.dumps({
    'source':'tps_sia/tp5/results/presentation-campaign-2026-11-10/02-optimizer/summary.json',
    'loss':'bce','latent_activation':'tanh','learning_rate':.002,'seeds':[42,43,44],
    'aggregates':optimizer_groups},ensure_ascii=False,indent=2),encoding='utf-8')
print('Generated 4 story figures in PNG/SVG; verified loss and optimizer comparisons against saved aggregates.')

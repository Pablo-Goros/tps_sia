from pathlib import Path
import sys, json, hashlib, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TP=Path(__file__).resolve().parents[1]
ROOT=TP.parents[1]
OUT=Path(__file__).resolve().parent/'assets'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(TP))
from src.dataset import load_dataset
from src.autoencoder import Autoencoder
ds=load_dataset()
model=Autoencoder.load(TP/'results/final/model.npz')
z=model.encode(ds.vectors)
pred=model.reconstruct(ds.vectors)
binary=(pred>=.5).astype(int).reshape(-1,7,5)
saved=json.loads((TP/'results/final/latent/latent.json').read_text())
assert np.allclose(z,[p['coordinates'] for p in saved['points']],atol=1e-12)
assert np.array_equal(binary,ds.images)
labels=[chr(c) if c!=127 else 'DEL' for c in range(96,128)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26384a','text.color':'#26384a','axes.titleweight':'bold','savefig.facecolor':'white'})
BLUE='#368dc4'; TEAL='#008f86'; INK='#26384a'; GRAY='#9aa9b8'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(4,8,figsize=(12,6.6))
for i,ax in enumerate(axes.flat):
    ax.imshow(ds.images[i],cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    ax.set_title(labels[i],fontsize=15,pad=5); ax.axis('off')
fig.subplots_adjust(wspace=.35,hspace=.5)
save(fig,'01-dataset')

fig,ax=plt.subplots(figsize=(13,4.8))
arch=[35,40,20,10,2,10,20,40,35]
for i,w in enumerate(arch):
    x=i*1.45; h=.8+2.1*w/40
    color=TEAL if i==4 else BLUE if i<4 else '#7194af'
    ax.add_patch(plt.Rectangle((x-.48,-h/2),.96,h,facecolor=color,alpha=.95))
    ax.text(x,0,str(w),ha='center',va='center',color='white',fontsize=22,weight='bold')
    if i<8: ax.annotate('',xy=(x+.95,0),xytext=(x+.5,0),arrowprops=dict(arrowstyle='->',color=INK,lw=1.5))
ax.text(2.15,-2,'ENCODER',ha='center',fontsize=15,color=BLUE)
ax.text(5.8,-2,'LATENTE 2D',ha='center',fontsize=15,color=TEAL)
ax.text(9.4,-2,'DECODER',ha='center',fontsize=15,color='#7194af')
ax.text(5.8,-2.9,'tanh en capas internas y latente, sigmoide en la salida',ha='center',fontsize=16)
ax.set(xlim=(-.9,12.6),ylim=(-3.4,1.8)); ax.axis('off')
save(fig,'03-final-architecture')

fig,ax=plt.subplots(figsize=(10,4.8))
names=['35–20–2–20–35','35–40–2–40–35','35–60–2–60–35','35–30–15–2–15–30–35','35–40–20–10–2–10–20–40–35']
values=[4,4.3333,4.6667,2,1.6667]
bars=ax.barh(names[::-1],values[::-1],color=[TEAL,BLUE,GRAY,GRAY,GRAY])
ax.bar_label(bars,labels=['1,67','2,00','4,67','4,33','4,00'],padding=7)
ax.set(xlim=(0,5.25),xlabel='Media del máximo error por carácter (píxeles)')
ax.axvline(1,color=TEAL,ls='--',alpha=.5)
ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
save(fig,'02-architecture-comparison')

fig,axes=plt.subplots(1,2,figsize=(11,4.6))
names=['Cuadrática\ntanh','BCE\ntanh','Cuadrática\nlineal','BCE\nlineal']
for ax,vals,lab in zip(axes,[[1,3,0,3],[1.6667,1,2,1]],['Corridas que cumplen (de 3)','Media del máximo error (píxeles)']):
    bars=ax.bar(names,vals,color=[GRAY,TEAL,GRAY,BLUE],width=.65)
    ax.bar_label(bars,fmt='%.2g',padding=5)
    ax.set_ylabel(lab);ax.set_ylim(0,max(vals)*1.25);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_yticks(range(4));fig.tight_layout(w_pad=3)
save(fig,'04-loss-latent-comparison')

summary=json.loads((TP/'results/presentation-campaign-2026-11-10/06-confirmation/summary.json').read_text())
fig,ax=plt.subplots(figsize=(10,4.8))
for variant,color,label in [('confirmation-reference',GRAY,'Cuadrática + RMSProp'),('confirmation-finalist',TEAL,'BCE + Adam')]:
    rows=sorted([r for r in summary['runs'] if r['variant']==variant],key=lambda r:r['seed'])
    ax.plot([r['seed'] for r in rows],[r['max_pixel_errors'] for r in rows],marker='o',ms=8,lw=2.2,color=color,label=label)
ax.axhline(1,color=INK,ls='--',alpha=.6,label='Tolerancia: ≤ 1 píxel por carácter')
ax.set(xlabel='Semilla de inicialización',ylabel='Máximo error por carácter (píxeles)',xticks=range(45,50),yticks=range(8),ylim=(-.35,7.6))
ax.legend(loc='upper left',frameon=False,fontsize=11);ax.grid(alpha=.15)
save(fig,'06-new-seeds-confirmation')

fig,axes=plt.subplots(4,8,figsize=(13,7.8))
for i,ax in enumerate(axes.flat):
    pair=np.concatenate([ds.images[i],np.zeros((7,2)),binary[i]],axis=1)
    ax.imshow(pair,cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    ax.set_title(labels[i],fontsize=14,pad=4);ax.axis('off')
fig.suptitle('Original  |  Reconstrucción',fontsize=19,y=.98)
fig.subplots_adjust(wspace=.35,hspace=.5)
save(fig,'07-reconstructions')

fig,ax=plt.subplots(figsize=(8.4,8))
ax.scatter(z[:,0],z[:,1],s=42,color=BLUE,zorder=3)
offsets={'a':(6,7),'o':(6,-13),'s':(4,-14),'e':(7,5),'z':(-13,7),'v':(5,9),'x':(5,8),'~':(5,8),'r':(5,8),'b':(7,5),'DEL':(-32,8),'y':(-12,-13),'w':(6,-12)}
for label,(x,y) in zip(labels,z):
    dx,dy=offsets.get(label,(6,7))
    ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=12,color=INK)
ax.set(xlabel='$z_1$',ylabel='$z_2$',xlim=(-1.14,1.16),ylim=(-1.13,1.13))
ax.set_aspect('equal'); ax.grid(alpha=.2);ax.set_axisbelow(True)
save(fig,'08-latent-space')
with (OUT/'08-latent-coordinates.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(['ascii_code','label','z1','z2'])
    writer.writerows((c,l,*point) for c,l,point in zip(range(96,128),labels,z))
(OUT/'provenance.json').write_text(json.dumps({'model':'tps_sia/tp5/results/final/model.npz','model_sha256':hashlib.sha256((TP/'results/final/model.npz').read_bytes()).hexdigest(),'run':'confirmation-finalist-seed-49','dataset_sha256':ds.sha256,'latent_verified_against_saved_coordinates':True,'exact_reconstructions':32,'threshold':.5,'historical_architecture_values_source':'tps_sia/tp5/comparison.md, campaña factorial histórica','confirmation_source':'tps_sia/tp5/results/presentation-campaign-2026-11-10/06-confirmation/summary.json'},ensure_ascii=False,indent=2),encoding='utf-8')
print('Generated 7 figures in PNG/SVG; verified 32 latent codes and exact reconstructions.')

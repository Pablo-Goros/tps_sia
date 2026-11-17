"""Generate the figures of exercise 1.b (denoising autoencoder).

Reads only results/denoising/: no training and no model loading. Key numbers
are checked against the CSV files before plotting.
"""
from pathlib import Path
import csv, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent/'assets'
OUT.mkdir(exist_ok=True)
TP=OUT.parents[1]
CAMP=TP/'results/denoising'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#26384a','axes.labelcolor':'#26384a','xtick.color':'#26384a','ytick.color':'#26384a','savefig.facecolor':'white'})
TEAL='#008f86'; BLUE='#368dc4'; GRAY='#9aa9b8'; INK='#26384a'; RED='#c0392b'
LEVELS=[0,.05,.1,.2,.3]
NOISY='Entrada ruidosa'; AE='AE de 1.a (sin ruido al entrenar)'; DAE2='DAE código 2'; DAE10='DAE código 10'
SEEDS=(42,43,44)
labels=[chr(c) if c!=127 else 'DEL' for c in range(96,128)]

def es(value,decimals=2):
    return f'{value:.{decimals}f}'.replace('.',',')
def save(fig,name):
    for ext in ('png','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
def read_csv(name):
    with (CAMP/'evaluation'/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))
def glyph(ax,vector):
    ax.imshow(np.asarray(vector).reshape(7,5),cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([])
    for side in ax.spines.values(): side.set_visible(False)

# ---------- Data ----------
by_level={(r['model'],float(r['level'])):r for r in read_csv('by-level.csv') if r['character_index']==''}
by_char=read_csv('by-character.csv')
agg={(r['variant'],float(r['level'])):r for r in read_csv('seed-aggregates.csv')}
metrics=json.loads((CAMP/'evaluation/metrics.json').read_text(encoding='utf-8'))
selection=json.loads((CAMP/'selection.json').read_text(encoding='utf-8'))
final=np.load(CAMP/'collections/final.npz')
examples=np.array(metrics['example_indices']).reshape(len(LEVELS),6)

def level_values(model,key):
    return [float(by_level[(model,p)][key]) for p in LEVELS]
def seed_values(variant,key):
    return [float(agg[(variant,p)][key]) for p in LEVELS]
def seed_mean(variant,key):
    return [np.mean([float(by_level[(f'{variant}-seed-{s}',p)][key]) for s in SEEDS]) for p in LEVELS]
def check(values,expected,decimals=3):
    assert [round(v,decimals) for v in values]==expected,(values,expected)

# ---------- Checks against the authorized numbers ----------
assert all(int(by_level[('noisy-input',p)]['count'])==(32 if p==0 else 3200) for p in LEVELS)
check(level_values('noisy-input','mean_pixel_errors'),[0,1.767,3.466,6.893,10.568])
check(level_values('ae-reference','mean_pixel_errors'),[0,6.463,9.26,12.606,15.032])
check(seed_values('winner','mean_pixel_errors_mean'),[.26,.553,1.327,4.654,9.032])
check(seed_values('winner','mean_pixel_errors_min')[1:],[.326,1.141,4.465,8.811])
check(seed_values('winner','mean_pixel_errors_max')[1:],[.713,1.423,4.869,9.161])
check(seed_values('latent-10','mean_pixel_errors_mean'),[0,.111,.5,2.935,7.183])
check(seed_values('latent-10','mean_pixel_errors_min')[1:],[.094,.471,2.886,7.098])
check(seed_values('latent-10','mean_pixel_errors_max')[1:],[.12,.547,2.966,7.304])
check(level_values('noisy-input','exact_fraction'),[1,.166,.029,.001,0])
check(level_values('ae-reference','exact_fraction'),[1,.349,.188,.084,.045])
check(seed_values('winner','exact_fraction_mean'),[.844,.767,.656,.355,.132])
check(seed_values('latent-10','exact_fraction_mean'),[1,.955,.852,.496,.173])
# The aggregates file agrees with the per-seed rows.
for variant in ('winner','latent-10'):
    assert np.allclose(seed_values(variant,'mean_pixel_errors_mean'),seed_mean(variant,'mean_pixel_errors'))
# The seed ranges of the two variants do not overlap at any level.
assert all(float(agg[('latent-10',p)]['mean_pixel_errors_max'])<float(agg[('winner',p)]['mean_pixel_errors_min']) for p in LEVELS)
assert all(float(by_level[(f'latent-10-seed-{s}',p)]['mean_pixel_errors'])<float(by_level[(f'winner-seed-{t}',p)]['mean_pixel_errors']) for p in LEVELS for s in SEEDS for t in SEEDS)
check(level_values('ae-reference','worsened_fraction')[2:4],[.73,.80],2)
check(seed_mean('latent-10','worsened_fraction')[1:],[.009,.027,.111,.193])
validation={a['variant']:a['mean_score'][0] for a in selection['aggregates']}
check([validation['latent-10'],validation['winner']],[1.23,2.28],2)
assert selection['run_id']==metrics['selected_run']=='latent-10-seed-44'
best={v:min((r for r in selection['runs'] if r['variant']==v),key=lambda r:(r['best']['score'],r['seed']))['run_id'] for v in ('winner','latent-10')}
assert best=={'winner':'winner-seed-43','latent-10':'latent-10-seed-44'}
assert sum(r['best']['epoch']==3000 for r in selection['runs'])==4
# Frozen examples: first realization of characters 0, 1, 8, 16, 24 and 31 at each level.
assert metrics['out_of_training_range']==[.3]
for row,p in zip(examples,LEVELS):
    assert np.allclose(final['levels'][row],p) and list(final['character_indices'][row])==[0,1,8,16,24,31]
    assert not final['realization_indices'][row].any()

# ---------- 1. Noise levels on one character ----------
CHARACTER=4  # position in the frozen examples: character 24, 'x'
fig,axes=plt.subplots(1,5,figsize=(10,3.9))
for ax,row,p in zip(axes,examples,LEVELS):
    index=row[CHARACTER]
    flipped=int((final['inputs'][index]!=final['targets'][index]).sum())
    glyph(ax,final['inputs'][index])
    ax.set_title('Original' if p==0 else f'p = {es(p)}',fontsize=20,pad=10,color=INK if p<.3 else GRAY,weight='bold')
    ax.set_xlabel('sin ruido' if p==0 else f'{flipped} píxeles\ninvertidos',fontsize=15,labelpad=10)
axes[4].text(.5,-.46,'(solo evaluación)',transform=axes[4].transAxes,ha='center',fontsize=13,color=GRAY)
fig.subplots_adjust(wspace=.3)
save(fig,'denoising-01-noise-levels')
noise_example={'character':labels[24],'indices':[int(r[CHARACTER]) for r in examples]}

# ---------- 2. The two architectures ----------
fig,ax=plt.subplots(figsize=(10,6.2))
for y,code,color,name in [(2.6,2,BLUE,DAE2+'  (la red de 1.a)'),(-2.6,10,TEAL,DAE10)]:
    arch=[35,40,20,10,code,10,20,40,35]
    assert arch==next(r['effective_model']['architecture'] for r in selection['runs'] if r['variant']==('winner' if code==2 else 'latent-10'))
    for i,w in enumerate(arch):
        x=i*1.45; h=.8+2.1*w/40
        ax.add_patch(plt.Rectangle((x-.48,y-h/2),.96,h,facecolor=color if i==4 else GRAY))
        ax.text(x,y,str(w),ha='center',va='center',color='white',fontsize=22,weight='bold')
        if i<8: ax.annotate('',xy=(x+.95,y),xytext=(x+.5,y),arrowprops=dict(arrowstyle='->',color=INK,lw=1.5))
    ax.text(-.48,y+2.05,name,fontsize=19,weight='bold',color=color)
    ax.text(5.8,y-1.15,f'código: {code}',ha='center',fontsize=16,color=color)
ax.set(xlim=(-.9,12.6),ylim=(-4.6,5.2)); ax.axis('off')
save(fig,'denoising-02-architectures')

# ---------- 3. Pixel errors by noise level ----------
fig,ax=plt.subplots(figsize=(8.8,5.3))
ax.axvspan(.25,.32,color=GRAY,alpha=.18,lw=0)
ax.text(.285,17.2,'fuera del rango\nentrenado',ha='center',va='top',fontsize=12)
ax.plot(LEVELS,level_values('ae-reference','mean_pixel_errors'),color=GRAY,lw=2.6,marker='s',ms=8,label=AE)
ax.plot(LEVELS,level_values('noisy-input','mean_pixel_errors'),color=GRAY,lw=2.6,ls='--',marker='o',ms=8,label=NOISY)
for variant,color,name in [('winner',BLUE,DAE2),('latent-10',TEAL,DAE10)]:
    ax.fill_between(LEVELS,seed_values(variant,'mean_pixel_errors_min'),seed_values(variant,'mean_pixel_errors_max'),color=color,alpha=.25,lw=0)
    ax.plot(LEVELS,seed_values(variant,'mean_pixel_errors_mean'),color=color,lw=3,marker='o',ms=8,label=name+' (3 semillas)')
for model in ('ae-reference','noisy-input'):
    ax.annotate(es(float(by_level[(model,.1)]['mean_pixel_errors'])),(.1,float(by_level[(model,.1)]['mean_pixel_errors'])),xytext=(0,9),textcoords='offset points',ha='center',fontsize=14)
ax.annotate(es(float(agg[('latent-10',.1)]['mean_pixel_errors_mean'])),(.1,float(agg[('latent-10',.1)]['mean_pixel_errors_mean'])),xytext=(0,-22),textcoords='offset points',ha='center',fontsize=14,color=TEAL,weight='bold')
ax.set(xlabel='Probabilidad de invertir cada píxel (p)',ylabel='Píxeles erróneos por imagen',xticks=LEVELS,xlim=(-.012,.32),ylim=(-1.6,17.5),yticks=[0,5,10,15])
ax.set_xticklabels(['0','0,05','0,10','0,20','0,30'])
ax.legend(loc='upper left',frameon=False,fontsize=13);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'denoising-03-error-by-level')

# ---------- 4. Code 10 against code 2 ----------
fig,axes=plt.subplots(1,2,figsize=(9.2,5.5),gridspec_kw={'width_ratios':[1.9,1]})
ax=axes[0]
ax.axvspan(.25,.32,color=GRAY,alpha=.18,lw=0)
ax.text(.285,1.1,'fuera del rango\nentrenado',ha='center',va='top',fontsize=11)
for variant,color,name,offset in [('winner',BLUE,DAE2,-24),('latent-10',TEAL,DAE10,11)]:
    mean=seed_values(variant,'exact_fraction_mean')
    ax.fill_between(LEVELS,seed_values(variant,'exact_fraction_min'),seed_values(variant,'exact_fraction_max'),color=color,alpha=.25,lw=0)
    ax.plot(LEVELS,mean,color=color,lw=3,marker='o',ms=8,label=name)
    for p,v in zip(LEVELS,mean):
        ax.annotate(f'{int(round(v,3)*100+.5)} %',(p,v),xytext=(4 if offset>0 else -4,offset),textcoords='offset points',ha='left' if offset>0 else 'right',fontsize=13,color=color,weight='bold')
ax.set(xlabel='Probabilidad de invertir cada píxel (p)',ylabel='Imágenes recuperadas exactas',xticks=LEVELS,xlim=(-.07,.32),ylim=(0,1.12),yticks=[0,.25,.5,.75,1])
ax.set_xticklabels(['0','0,05','0,10','0,20','0,30']);ax.set_yticklabels(['0 %','25 %','50 %','75 %','100 %'])
ax.legend(loc='lower left',frameon=False,fontsize=13);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax=axes[1]
clean=[float(agg[(v,0)]['mean_pixel_errors_mean']) for v in ('winner','latent-10')]
low,high=float(agg[('winner',0)]['mean_pixel_errors_min']),float(agg[('winner',0)]['mean_pixel_errors_max'])
bars=ax.bar(['DAE\ncódigo 2','DAE\ncódigo 10'],clean,color=[BLUE,TEAL],width=.55)
ax.errorbar([0],[clean[0]],yerr=[[clean[0]-low],[high-clean[0]]],color=INK,capsize=7,lw=1.8)
ax.text(.2,clean[0]+.01,es(clean[0]),fontsize=18,color=BLUE,weight='bold',va='bottom')
ax.text(1,.012,'0',fontsize=18,color=TEAL,weight='bold',ha='center',va='bottom')
ax.set(ylabel='Píxeles erróneos con p = 0\n(imágenes limpias)',ylim=(0,.56));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.set_yticks([0,.1,.2,.3,.4,.5]);ax.set_yticklabels(['0','0,1','0,2','0,3','0,4','0,5'])
fig.tight_layout(w_pad=1.5)
save(fig,'denoising-04-code-comparison')

# ---------- 5. Fixed examples ----------
rows=[('Original',None),(NOISY,'noisy-input'),('AE de 1.a\n(sin ruido al entrenar)','ae-reference'),(DAE2,best['winner']),(DAE10,best['latent-10'])]
predictions={model:np.load(CAMP/'evaluation'/f'{model}.npz') for _,model in rows if model}
example_errors={}
def examples_figure(level,name):
    indices=examples[LEVELS.index(level)]
    fig,axes=plt.subplots(5,6,figsize=(9.6,8.4))
    for r,(title,model) in enumerate(rows):
        for c,index in enumerate(indices):
            ax=axes[r,c]
            target=final['targets'][index]
            image=target if model is None else final['inputs'][index] if model=='noisy-input' else predictions[model]['binary'][index]
            glyph(ax,image)
            if model:
                errors=int((np.asarray(image)!=target).sum())
                assert errors==int(predictions[model]['pixel_errors'][index])
                example_errors.setdefault(name,{}).setdefault(model,[]).append(errors)
                ax.set_xlabel(f'{errors} err',fontsize=14,color=RED if errors>1 else INK,weight='bold' if errors>1 else 'normal',labelpad=4)
            else:
                ax.set_title(labels[final['character_indices'][index]],fontsize=17,pad=6,weight='bold')
        axes[r,0].set_ylabel(title,rotation=0,ha='right',va='center',fontsize=15,labelpad=14,color=TEAL if model==best['latent-10'] else BLUE if model==best['winner'] else INK,weight='bold' if model==best['latent-10'] else 'normal')
    fig.subplots_adjust(wspace=.25,hspace=.42)
    save(fig,name)
examples_figure(.2,'denoising-05-examples-p020')
examples_figure(.1,'denoising-06-examples-p010')
# The values quoted in report.md for p = 0,20.
assert example_errors['denoising-05-examples-p020']=={'noisy-input':[7,6,8,4,7,8],'ae-reference':[13,22,22,8,17,0],'winner-seed-43':[7,0,10,8,9,0],'latent-10-seed-44':[3,0,3,0,0,0]}

# ---------- 6. Error by character (backup) ----------
by_character=np.zeros((4,32))
for r in by_char:
    if r['model'].startswith('latent-10') and float(r['level'])>0:
        by_character[LEVELS.index(float(r['level']))-1,int(r['character_index'])]+=float(r['mean_pixel_errors'])/len(SEEDS)
assert np.allclose(by_character.mean(axis=1),seed_values('latent-10','mean_pixel_errors_mean')[1:])
fig,ax=plt.subplots(figsize=(13,3.9))
ax.imshow(by_character/by_character.max(axis=1,keepdims=True),cmap='Blues',vmin=0,vmax=1.25,aspect='auto',interpolation='nearest')
for (r,c),v in np.ndenumerate(by_character):
    ax.text(c,r,es(v,1),ha='center',va='center',fontsize=10,color='white' if v/by_character[r].max()>.7 else INK)
ax.set_xticks(range(32));ax.set_xticklabels(labels,fontsize=13);ax.xaxis.tick_top()
ax.set_yticks(range(4));ax.set_yticklabels([f'p = {es(p)}' for p in LEVELS[1:]],fontsize=14)
ax.tick_params(length=0)
for side in ax.spines.values(): side.set_visible(False)
ax.set_xlabel('Píxeles erróneos por imagen del DAE código 10 (media de 3 semillas). El color es relativo al máximo de cada nivel.',fontsize=12,labelpad=10)
save(fig,'denoising-07-by-character')
# Exploratory: Spearman correlation between the error and the Hamming distance to the nearest other character.
clean=final['targets'][:32]
assert list(final['character_indices'][:32])==list(range(32))
distance=(clean[:,None,:]!=clean[None,:,:]).sum(axis=2); np.fill_diagonal(distance,35)
def ranks(values):
    values=np.asarray(values,float)
    return np.array([(values<v).sum()+((values==v).sum()+1)/2 for v in values])
spearman=[float(np.corrcoef(ranks(distance.min(axis=1)),ranks(row))[0,1]) for row in by_character]
assert spearman[0]<-.75 and abs(spearman[3])<.1

# ---------- Provenance ----------
inputs=['evaluation/by-level.csv','evaluation/by-character.csv','evaluation/seed-aggregates.csv','evaluation/metrics.json','selection.json','collections/final.npz']+[f'evaluation/{m}.npz' for m in predictions]
(OUT/'denoising-provenance.json').write_text(json.dumps({
    'campaign':'tps_sia/tp5/results/denoising',
    'inputs_sha256':{p:hashlib.sha256((CAMP/p).read_bytes()).hexdigest() for p in inputs},
    'selected_run':selection['run_id'],'sample_models':best,
    'example_indices':metrics['example_indices'],'example_policy':metrics['example_policy'],
    'noise_example':noise_example,'example_pixel_errors':example_errors,
    'validation_mean_pixel_errors':validation,
    'spearman_error_vs_min_hamming':dict(zip(['0.05','0.1','0.2','0.3'],spearman)),
    'trained_or_loaded_models':False},ensure_ascii=False,indent=2),encoding='utf-8')
print('Generated 7 denoising figures in PNG/SVG; verified the key numbers against results/denoising.')

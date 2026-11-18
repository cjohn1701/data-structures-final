"""Reproduce retrospective MLB teammate-network study. Run from repository root."""
from pathlib import Path
from collections import Counter
from itertools import combinations
import hashlib,json,platform,urllib.request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import sklearn
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
FEATURES=['density','tenure','players']; KEYS=['yearID','teamID','lgID']

def prepare():
    manifest=[]
    checks={'Teams': '9d1c390df5d2a422175f6bcfb56d099cfb9c5805586ba5e3f8c895a9952423ad', 'Appearances': 'c1f5ed768ef4161d5a8813ba0d6061fb984acb442a393d53078f376811a2f6f3'}
    for n in ['Teams','Appearances']:
        p=ROOT/'data/raw'/f'{n}.csv';p.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/f91e6ed4824ad5cdb75c3a5495d228294d91a184/csv/Lahman/{n}.csv'
        if not p.exists(): urllib.request.urlretrieve(url,p)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==checks[n], 'Input checksum mismatch: '+n
        manifest.append({'file':str(p.relative_to(ROOT)),'url':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'retrieved':'2026-11-18'})
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    a=pd.read_csv(ROOT/'data/raw/Appearances.csv');t=pd.read_csv(ROOT/'data/raw/Teams.csv')
    a=a[a.yearID.between(1995,2024)&a.lgID.isin(['AL','NL'])].copy()
    before=len(a);a=a.drop_duplicates(KEYS+['playerID'])
    assert a[KEYS+['playerID']].notna().all().all()
    history=Counter(); rows=[]
    for year,season in a.groupby('yearID',sort=True):
        current=set()
        for key,g in season.groupby(KEYS):
            ps=sorted(g.playerID.unique()); pairs=list(combinations(ps,2)); vals=[history[x] for x in pairs]
            rows.append(dict(zip(KEYS,key),players=len(ps),density=np.mean(np.array(vals)>0),tenure=np.mean(vals)))
            current.update(pairs)
        history.update(current) # never update within a year; each pair/year counted once
    f=pd.DataFrame(rows);f=f[f.yearID.between(2005,2024)]
    t=t[t.yearID.between(2005,2024)&t.lgID.isin(['AL','NL'])]
    d=t[KEYS+['name','W','L']].merge(f,on=KEYS,how='outer',validate='one_to_one',indicator=True)
    assert (d._merge=='both').all(), d[d._merge!='both']
    d=d.drop(columns='_merge');d['win_pct']=d.W/(d.W+d.L)
    assert len(d)==600 and not d.duplicated(KEYS).any() and not d.isna().any().any()
    assert d.groupby('yearID').size().eq(30).all()
    d=d.sort_values(KEYS).reset_index(drop=True)
    d.to_csv(ROOT/'data/processed/team_seasons.csv',index=False)
    (OUT/'quality_checks.json').write_text(json.dumps({'rows':len(d),'columns':len(d.columns),'duplicate_appearance_rows_removed':before-len(a),'missing_cells':int(d.isna().sum().sum()),'unmatched_keys':0,'train_rows':450,'test_rows':150},indent=2))
    return d

def models():
    return {'Mean baseline':DummyRegressor(),'Linear regression':LinearRegression(),'Ridge regression':make_pipeline(StandardScaler(),Ridge(alpha=1)), 'Random forest':RandomForestRegressor(n_estimators=250,max_depth=5,min_samples_leaf=8,random_state=42,n_jobs=-1)}
def score(y,p): return {'MAE_pp':100*mean_absolute_error(y,p),'RMSE_pp':100*np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p)}
def run():
    d=prepare();tr=d[d.yearID<2020];te=d[d.yearID>=2020]
    scores=[];pred=te.copy();fits={}
    for n,m in models().items():
        m.fit(tr[FEATURES],tr.win_pct);p=m.predict(te[FEATURES]);fits[n]=m;pred[n]=p;scores.append({'model':n,**score(te.win_pct,p)})
    pd.DataFrame(scores).to_csv(OUT/'model_metrics.csv',index=False)
    pred['residual_pp']=100*(pred.win_pct-pred['Linear regression']);pred['abs_error_pp']=abs(pred.residual_pp)
    pred.to_csv(OUT/'test_predictions.csv',index=False)
    # Predefined expanding chronological validation folds, no test-set tuning
    cv=[]
    for start,end in [(2011,2013),(2014,2016),(2017,2019)]:
        train=d[d.yearID<start];val=d[d.yearID.between(start,end)]
        for n,m in models().items():
            m.fit(train[FEATURES],train.win_pct);cv.append({'validation_years':f'{start}-{end}','model':n,**score(val.win_pct,m.predict(val[FEATURES]))})
    pd.DataFrame(cv).to_csv(OUT/'chronological_validation.csv',index=False)
    # Secondary checks reuse the existing holdout and are descriptive, not independent confirmation
    sens=[]
    sets={'All features':FEATURES,'Density and roster size':['density','players'],'Tenure and roster size':['tenure','players'],'Roster size only':['players'],'Density only':['density'],'Tenure only':['tenure']}
    for name,cols in sets.items():
        m=LinearRegression().fit(tr[cols],tr.win_pct)
        for period,sub in [('2020-2024',te),('2021-2024',te[te.yearID>2020])]:
            sens.append({'features':name,'test_years':period,**score(sub.win_pct,m.predict(sub[cols]))})
    sens.append({'features':'Mean baseline','test_years':'2021-2024',**score(te[te.yearID>2020].win_pct,fits['Mean baseline'].predict(te[te.yearID>2020][FEATURES]))})
    pd.DataFrame(sens).to_csv(OUT/'feature_sensitivity.csv',index=False)
    co=fits['Linear regression'].coef_;pd.DataFrame({'feature':FEATURES,'coefficient_raw':co,'change_for_reporting_unit_pp':co*np.array([.10,.10,10])*100,'reporting_unit':['0.10 density','0.10 seasons per pair','10 players']}).to_csv(OUT/'coefficients.csv',index=False)
    groups=[]
    for col in ['yearID','lgID']:
        for key,g in pred.groupby(col): groups.append({'group_type':col,'group':str(key),'n':len(g),**score(g.win_pct,g['Linear regression'])})
    cutoff=tr.players.median()
    for label,g in pred.groupby(np.where(pred.players>cutoff,f'players > {cutoff:g}',f'players <= {cutoff:g}')):
        groups.append({'group_type':'roster size','group':label,'n':len(g),**score(g.win_pct,g['Linear regression'])})
    pd.DataFrame(groups).to_csv(OUT/'error_groups.csv',index=False)
    pred.nlargest(10,'abs_error_pp')[KEYS+['name','win_pct','Linear regression','residual_pp']].to_csv(OUT/'largest_errors.csv',index=False)
    d[['win_pct']+FEATURES].describe().T.to_csv(OUT/'descriptive_statistics.csv')
    d[['win_pct']+FEATURES].corr().to_csv(OUT/'correlations.csv')
    # Paired season-level resampling describes uncertainty over only five observed test years
    losses=pred.assign(gain=100*(abs(pred.win_pct-pred['Mean baseline'])-abs(pred.win_pct-pred['Linear regression']))).groupby('yearID').gain.mean()
    rng=np.random.default_rng(42);boot=rng.choice(losses.values,size=(10000,len(losses)),replace=True).mean(axis=1)
    (OUT/'summary.json').write_text(json.dumps({'density_correlation':float(d.density.corr(d.win_pct)),'density_tenure_correlation':float(d.density.corr(d.tenure)),'players_correlation':float(d.players.corr(d.win_pct)),'linear_intercept':float(fits['Linear regression'].intercept_),'mean_mae_gain_pp':float(losses.mean()),'season_bootstrap_95_percentile_interval':np.quantile(boot,[.025,.975]).tolist(),'python':platform.python_version(),'sklearn':sklearn.__version__},indent=2))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    def save(n): plt.tight_layout();plt.savefig(OUT/f'{n}.png',bbox_inches='tight');plt.close()
    plt.figure(figsize=(7,4));plt.hist(100*d.win_pct,bins=18,color='#155e63',edgecolor='white');plt.xlabel('Regular-season win percentage (%)');plt.ylabel('Team-seasons');plt.title('Distribution of team performance, 2005–2024');save('fig01_win_distribution')
    plt.figure(figsize=(7,4));plt.hist(100*d.density,bins=18,color='#155e63',edgecolor='white');plt.xlabel('Prior teammate network density (%)');plt.ylabel('Team-seasons');plt.title('Prior teammate connections vary across rosters');save('fig02_density_distribution')
    plt.figure(figsize=(7,4));plt.scatter(100*d.density,100*d.win_pct,alpha=.40,s=20,color='#155e63');plt.xlabel('Prior teammate network density (%)');plt.ylabel('Regular-season win percentage (%)');plt.title(f'Density and performance: r = {d.density.corr(d.win_pct):.2f}');save('fig03_density_vs_wins')
    plt.figure(figsize=(7,4));plt.scatter(d.tenure,100*d.win_pct,alpha=.40,s=20,color='#155e63');plt.xlabel('Average prior shared seasons per player pair');plt.ylabel('Regular-season win percentage (%)');plt.title('Shared tenure and team performance');save('fig04_tenure_vs_wins')
    plt.figure(figsize=(7,4));plt.scatter(d.players,100*d.win_pct,alpha=.40,s=20,color='#155e63');plt.xlabel('Number of players with a season appearance');plt.ylabel('Regular-season win percentage (%)');plt.title('Full-season roster size and team performance');save('fig05_roster_vs_wins')
    plt.figure(figsize=(7,4));plt.scatter(100*pred['Linear regression'],pred.residual_pp,alpha=.65,color='#155e63');plt.axhline(0,color='black',lw=1);plt.xlabel('Predicted win percentage (%)');plt.ylabel('Actual minus predicted (percentage points)');plt.title('Linear model residuals in 2020–2024');save('fig06_residuals')
    sc=pd.DataFrame(scores);plt.figure(figsize=(7,4));plt.barh(sc.model,sc.MAE_pp,color=['#8b8b8b','#155e63','#357f83','#9d6d31']);plt.xlabel('Held-out MAE (percentage points; lower is better)');plt.xlim(0,8);plt.title('Models offer small gains over the mean baseline');save('fig07_model_comparison')
    plt.figure(figsize=(7,4));plt.scatter(100*pred.win_pct,100*pred['Linear regression'],alpha=.65,color='#155e63');plt.plot([20,75],[20,75],c='black',lw=1);plt.xlabel('Actual win percentage (%)');plt.ylabel('Predicted win percentage (%)');plt.title('Predictions compress extreme team outcomes');save('fig08_actual_predicted')
    print(pd.DataFrame(scores).to_string(index=False));print((OUT/'summary.json').read_text());print(pd.DataFrame(groups).to_string(index=False))
if __name__=='__main__':run()

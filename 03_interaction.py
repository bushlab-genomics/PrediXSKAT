#test of normal vs uniform quantiletransforms for kernal mask. use config.txt

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from fastlmmclib import quadform as qf
from sklearn.preprocessing import QuantileTransformer
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--causal", default=1.0, type=float, help="Causal percent")
parser.add_argument("-p", "--positive", default=1.0, type=float, help="Positive percent")
parser.add_argument("-n", "--sample", default=3000.0, type=float, help="subsample number")
parser.add_argument("-m", "--mineffect", default=0.2, type=float, help="min effect mult")
parser.add_argument("-o", "--output", default="results.txt", help='output file')
args = vars(parser.parse_args())

causal_args = args['causal']
positive_args= args['positive']
num_args = args['sample']
mineffect_args = args['mineffect']
arg_list = [causal_args, positive_args]
file_path = args['output']

print(' causal=', causal_args, ' positive=', positive_args)
print(' output in ', file_path)

geno = pd.read_csv('/rawdata/anon_genotypes.csv', header=None)
app = pd.read_csv('/rawdata/anon_grex.csv', header=None)

def generate_sample(geno, app, num=2000):
    df = geno.sample(n=int(num), axis=1)
    mafs = pd.DataFrame(index=df.index)
    mafs['MAF'] = df.sum(axis=1) / len(df.columns)
    vars_to_drop = mafs.loc[mafs['MAF']<= (1/num)].index.to_list() + mafs.loc[mafs['MAF']>0.01].index.to_list()
    df = df.drop(index=vars_to_drop).sort_index()
    df = df[sorted(df.columns)]
    mafs=mafs.drop(index=vars_to_drop)
    app_df = app.loc[app.index.isin(df.columns)]['ABCA7'].sort_index()
    
    w_diag = sp.sparse.diags(sp.stats.beta.pdf(mafs['MAF'], 1, 25))
    kernel = df.transpose().values @ w_diag @ df.values

    qt = QuantileTransformer(output_distribution='normal')
    X_qt = qt.fit_transform(app_df.values.reshape(-1, 1))
    kern_normal = (2-0.2*(abs(X_qt - X_qt.transpose())))
    kern_normal[kern_normal < 1] = 1

    qt1 = QuantileTransformer(output_distribution='uniform')
    X_qt1 = qt1.fit_transform(app_df.values.reshape(-1, 1))
    kern_uniform = 1.5-abs(X_qt1 - X_qt1.transpose())

    kernel_alt_normal = kernel * kern_normal    
    kernel_alt_uniform = kernel * kern_uniform    
    return df, mafs, app_df, kernel, kernel_alt_normal, kernel_alt_uniform

def generate_risk(df, mafs, app, causal, positive, mineffect=0.5):
    betas = mineffect * abs(np.log10(mafs['MAF']))
    indices_to_zero = betas.sample(int(mafs.shape[0] * (1.0 - causal))).index
    betas.loc[indices_to_zero] = 0
    nonzero = betas.loc[betas != 0]
    indices_to_flip = nonzero.sample(int(nonzero.shape[0] * (1 - positive))).index
    betas.loc[indices_to_flip] = -1 * betas.loc[indices_to_flip]
    alphas = pd.Series(np.random.normal(size=df.shape[1]), index=df.columns)

    risk = alphas + df.T @ betas + 0.1 * app_df + 0.2 * (df.sum() *  app_df)
    threshold = np.quantile(risk, 0.532) #prevalence
    phenos = risk.apply(lambda y: 1 if y > threshold else 0)
    return phenos

def skat_test_covar(df, app_df, kernel, phenos):
    pi = LogisticRegression(penalty=None).fit(app_df.values.reshape(-1, 1), phenos).predict_proba(app_df.values.reshape(-1, 1))[:,1]
    residual = phenos - pi
    Qt = residual.transpose() @ kernel @ residual
    n = df.shape[1]
    X1 = np.hstack([np.ones((n, 1)), app_df.values.reshape(-1, 1)])
    V=sp.sparse.diags(pi * (1-pi))
    inner = X1.transpose() @ V @ X1
    inv_inner = np.linalg.pinv(inner)
    P = V - V @ X1 @ inv_inner @ X1.transpose() @ V
    P_rt1 = sp.linalg.sqrtm(P)
    PKP1 = P_rt1 @ kernel @ P_rt1
    phis1 = sp.sparse.linalg.eigsh(PKP1, k=df.shape[0], return_eigenvectors=False)
    phis1 = np.sort(phis1[np.where(phis1 > 1e-8)])[::-1]
    phis1 = np.ascontiguousarray(phis1)
    pval = qf.qf(Qt, phis1, acc=1e-7)[0]
    return pval

def skat_test_basic(df, app_df, kernel, phenos):
    pi = phenos.mean()
    residual = phenos - pi
    Qt = residual.transpose() @ kernel @ residual
    n = df.shape[1]
    V1 = np.identity(n) - (1/n) * np.ones((n, n))
    P = pi * (1 - pi) * V1
    P_rt1 = sp.linalg.sqrtm(P)
    PKP1 = P_rt1 @ kernel @ P_rt1
    phis1 = sp.sparse.linalg.eigsh(PKP1, k=df.shape[0], return_eigenvectors=False)
    phis1 = np.sort(phis1[np.where(phis1 > 1e-8)])[::-1]
    phis1 = np.ascontiguousarray(phis1)
    pval = qf.qf(Qt, phis1, acc=1e-7)[0]
    return pval

def add_results_to_file(file_path, arg_list, results):
    writeline = ','.join(map(str, arg_list))+',' + ','.join(map(str, results))+'\n'
    with open('/home/yxy1313/SKATproject/Sims/slurm/results/'+file_path, 'a') as file:
        file.write(writeline)
        
for i in tqdm(range(1, 201)):
    df, mafs, app_df, kernel, kernel_alt_normal, kernel_alt_uniform = generate_sample(geno, app, num_args)
    phenos = generate_risk(df, mafs, app_df, causal_args, positive_args, mineffect_args)
    pval00 = skat_test_basic(df, app_df, kernel, phenos)
    #pval10 = skat_test_basic(df, app_df, kernel_alt_normal, phenos)
    pval20 = skat_test_basic(df, app_df, kernel_alt_uniform, phenos)
    #pval01 = skat_test_covar(df, app_df, kernel, phenos)
    #pval11 = skat_test_covar(df, app_df, kernel_alt_normal, phenos)
    #pval21 = skat_test_covar(df, app_df, kernel_alt_uniform, phenos)
    #results = [pval00, pval10, pval20, pval01, pval11, pval21]
    results = [pval00, pval20]
    add_results_to_file(file_path, arg_list, results)

print('all tests completed')
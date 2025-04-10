import numpy as np
from statsmodels.datasets import grunfeld

from ipca import IPCA

########## load data ##########
"""
Grunfeld (1950) Investment Data
source: http://statmath.wu-wien.ac.at/~zeileis/grunfeld/
220 observations (11 US firms from 1935-1954)

invest  - Gross investment in 1947 dollars
value   - Market value as of Dec. 31 in 1947 dollars
capital - Stock of plant and equipment in 1947 dollars
firm    - General Motors, US Steel, General Electric, Chrysler,
        Atlantic Refining, IBM, Union Oil, Westinghouse, Goodyear,
        Diamond Match, American Steel
year    - 1935 - 1954
"""
data = grunfeld.load_pandas().data

########## preprocessing ##########

# convert date
data.year = data.year.astype(np.int64)

# establish unique IDs
N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(), np.arange(1, N+1)+5))
data.firm = data.firm.apply(lambda x: ID[x])

# rearrange ordering
data = data[['firm', 'year', 'invest', 'value', 'capital']]

# prepare pre-specified factors test vars
PSF1 = np.random.randn(len(np.unique(data.loc[:, 'year'])), 1)
PSF1 = PSF1.reshape((1, -1))
PSF2 = np.random.randn(len(np.unique(data.loc[:, 'year'])), 2)
PSF2 = PSF2.reshape((2, -1))

# set entity-time index and prepare independent variables (value, capital) and dependent variable (invest, analogous to return)
data = data.set_index(['firm', 'year'])
data_y = data['invest']
data_x = data.drop('invest', axis=1)

########## test IPCA ##########
regr = IPCA(n_factors=2, intercept=False)

regr = regr.fit(X=data_x, y=data_y)

# print("R2total", regr.score(X=data_x, y=data_y))
# print("R2pred", regr.score(X=data_x, y=data_y, mean_factor=True))
# print("R2total_x", regr.score(X=data_x, y=data_y, data_type="portfolio"))
# print("R2pred_x", regr.score(X=data_x, y=data_y, mean_factor=True,
#                              data_type="portfolio"))
# print(regr.Gamma)
# print(regr.Factors)
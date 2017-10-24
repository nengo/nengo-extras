import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels as sm


data = [1.0,-1.0,2.0,3.0,-1.0,0.0,2.0,-3.0,1.0,0.0,1.0,-1.0,3.0,-2.0]

# sarimax_model = SARIMAX(data, order=(2,0,2), enforce_invertibility=False, enforce_stationarity=False).fit(disp=False)
# print sarimax_model.summary()


arma_model = ARMA(data, order=(2,2)).fit([0,0,0,0], transparams=False, trend='nc', disp=False)
print arma_model.summary()
print arma_model.params
print arma_model.maparams
print arma_model.sigma2**0.5
# print arma_model.sigma2



# print arma_generate_sample(arma_model.maparams, [27.6513,-25.6847], 10, sigma=0.0016)

print arma_generate_sample(
    ar=arma_model.arparams,
    ma=arma_model.maparams,
    nsample=10,
    sigma=arma_model.sigma2**0.5
)



# arma_model = ARMA(data, order=(2,2)).fit(start_params=[0,0,0,0],disp=False)
# print arma_model.summary()




# np.random.seed(12345)
# arparams = np.array([.75, -.25])
# maparams = np.array([.65, .35])
# ar = np.r_[1, -arparams] # add zero-lag and negate
# print ar
# ma = np.r_[1, maparams] # add zero-lag
# print ma
# y = arma_generate_sample(ar, ma, 250)
# model = ARMA(y, (2, 2)).fit(trend='nc', disp=0)
# print model.summary()
# print model.params
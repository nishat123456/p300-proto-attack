# Fixed development optimization results

One fixed development character, three model seeds. Not population TPR, independent confirmation or robustness proof.

All76 rows per seed finite, actual budgets respected, target identities consistent; trace lengths and best-objective monotonicity checked.

Counts below are settings on the same character, not independent trials or TPR estimates. Longer-run differences do not prove global convergence.

## Model seed 20261013

Zero raw score-gradient methods at the clean probe: []

|Attack|Budget|Target reached|Accepted by methods|Seconds|
|---|---|---|---|---|
|FGSM|0.05|True|confidence,entropy,bu,lid|1.367|
|Gaussian_control|0.05|False||0.009|
|source_template|0.05|True|confidence,entropy,bu,lid|0.007|
|PGD|0.05|True|confidence,entropy,bu,lid|9.605|
|FGSM|0.1|True|confidence,entropy,bu|0.014|
|Gaussian_control|0.1|False||0.001|
|source_template|0.1|True|confidence,entropy,bu,lid|0.006|
|PGD|0.1|True|confidence,entropy,bu,lid|9.682|
|FGSM|0.2|True|confidence,entropy,bu|0.014|
|Gaussian_control|0.2|False||0.001|
|source_template|0.2|True|confidence,entropy,bu|0.006|
|PGD|0.2|True|confidence,entropy,bu,lid|9.818|
|FGSM|0.3|True|confidence,entropy|0.014|
|Gaussian_control|0.3|False||0.001|
|source_template|0.3|True|confidence,entropy,bu|0.005|
|PGD|0.3|True|confidence,entropy,bu|10.010|
|FGSM|0.5|True|confidence,entropy|0.014|
|Gaussian_control|0.5|False||0.001|
|source_template|0.5|True|confidence,entropy,bu|0.006|
|PGD|0.5|True|confidence,entropy,bu|10.175|

|Detector|Weight|Target reached 200/400|Own detector accepted targeted 200/400|Seconds 200/400|
|---|---|---|---|---|
|confidence|0.0|True/True|True/True|10.430/34.102|
|confidence|0.1|True/True|True/True|20.085/65.273|
|confidence|1.0|True/True|True/True|19.471/65.425|
|confidence|10.0|True/True|True/True|19.350/65.375|
|entropy|0.0|True/True|True/True|10.219/33.925|
|entropy|0.1|True/True|True/True|19.585/65.169|
|entropy|1.0|True/True|True/True|19.617/65.039|
|entropy|10.0|True/True|True/True|19.616/65.052|
|bu|0.0|True/True|True/True|10.186/33.934|
|bu|0.1|True/True|True/True|70.145/233.485|
|bu|1.0|True/True|True/True|70.174/233.420|
|bu|10.0|True/True|True/True|70.111/233.294|
|lid|0.0|True/True|True/True|10.161/33.784|
|lid|0.1|True/True|True/True|20.165/67.183|
|lid|1.0|True/True|True/True|20.212/67.092|
|lid|10.0|True/True|True/True|20.136/67.125|
|md|0.0|True/True|False/False|10.192/33.872|
|md|0.1|True/True|True/True|20.022/66.829|
|md|1.0|True/True|True/True|20.006/66.743|
|md|10.0|True/True|True/True|20.031/66.639|
|md_adjusted|0.0|True/True|False/False|10.169/33.870|
|md_adjusted|0.1|True/True|True/True|31.304/104.149|
|md_adjusted|1.0|True/True|True/True|31.280/104.156|
|md_adjusted|10.0|True/True|True/True|31.224/104.073|
|kde|0.0|True/True|False/False|10.160/33.894|
|kde|0.1|True/True|True/True|21.282/71.085|
|kde|1.0|True/True|True/True|21.342/71.099|
|kde|10.0|True/True|True/True|21.313/71.081|

|Detector|Weight|Steps/restarts|Initial/final best objective|
|---|---|---|---|
|confidence|0.0|200/3|8.56546/0|
|confidence|0.1|200/3|8.56571/1.9208e-05|
|confidence|1.0|200/3|8.56795/0.000192081|
|confidence|10.0|200/3|8.59032/0.0019208|
|confidence|0.0|400/5|8.56546/0|
|confidence|0.1|400/5|8.56571/1.9208e-05|
|confidence|1.0|400/5|8.56795/0.00019208|
|confidence|10.0|400/5|8.59032/0.0019208|
|entropy|0.0|200/3|8.56546/0|
|entropy|0.1|200/3|8.56931/0.000371568|
|entropy|1.0|200/3|8.60396/0.00371568|
|entropy|10.0|200/3|8.95046/0.0371568|
|entropy|0.0|400/5|8.56546/0|
|entropy|0.1|400/5|8.56931/0.000371568|
|entropy|1.0|400/5|8.60396/0.00371568|
|entropy|10.0|400/5|8.95046/0.0371568|
|bu|0.0|200/3|8.56546/0|
|bu|0.1|200/3|8.56621/0.000224116|
|bu|1.0|200/3|8.57292/0.00224116|
|bu|10.0|200/3|8.64003/0.0224116|
|bu|0.0|400/5|8.56546/0|
|bu|0.1|400/5|8.56621/0.000224116|
|bu|1.0|400/5|8.57292/0.00224116|
|bu|10.0|400/5|8.64003/0.0224116|
|lid|0.0|200/3|8.56546/0|
|lid|0.1|200/3|8.59684/0.00189013|
|lid|1.0|200/3|8.87926/0.0211329|
|lid|10.0|200/3|11.7035/0.253678|
|lid|0.0|400/5|8.56546/0|
|lid|0.1|400/5|8.59684/0.000786483|
|lid|1.0|400/5|8.87926/0.0111918|
|lid|10.0|400/5|11.7035/0.0855764|
|md|0.0|200/3|8.56546/0|
|md|0.1|200/3|8.61265/3.12128e-06|
|md|1.0|200/3|9.03737/3.26827e-05|
|md|10.0|200/3|13.2846/0.000340668|
|md|0.0|400/5|8.56546/0|
|md|0.1|400/5|8.61265/2.37101e-06|
|md|1.0|400/5|9.03737/1.75099e-05|
|md|10.0|400/5|13.2846/0.000217618|
|md_adjusted|0.0|200/3|8.56546/0|
|md_adjusted|0.1|200/3|8.61418/3.61414e-06|
|md_adjusted|1.0|200/3|9.05265/3.72996e-05|
|md_adjusted|10.0|200/3|13.4373/0.000292989|
|md_adjusted|0.0|400/5|8.56546/0|
|md_adjusted|0.1|400/5|8.61418/2.43208e-06|
|md_adjusted|1.0|400/5|9.05265/2.23441e-05|
|md_adjusted|10.0|400/5|13.4373/0.000260543|
|kde|0.0|200/3|8.56546/0|
|kde|0.1|200/3|8.6487/0.000122315|
|kde|1.0|200/3|9.39785/0.00127051|
|kde|10.0|200/3|16.8893/0.00947847|
|kde|0.0|400/5|8.56546/0|
|kde|0.1|400/5|8.6487/6.18472e-05|
|kde|1.0|400/5|9.39785/0.000726564|
|kde|10.0|400/5|16.8893/0.00613881|

## Model seed 20261014

Zero raw score-gradient methods at the clean probe: []

|Attack|Budget|Target reached|Accepted by methods|Seconds|
|---|---|---|---|---|
|FGSM|0.05|True|confidence,entropy,bu,lid|0.493|
|Gaussian_control|0.05|False||0.003|
|source_template|0.05|True|confidence,entropy,lid|0.005|
|PGD|0.05|True|confidence,entropy,bu,lid|10.002|
|FGSM|0.1|True|confidence,entropy,bu|0.014|
|Gaussian_control|0.1|False||0.001|
|source_template|0.1|True|confidence,entropy,bu|0.005|
|PGD|0.1|True|confidence,entropy,bu,lid|10.287|
|FGSM|0.2|True|confidence,entropy|0.015|
|Gaussian_control|0.2|False||0.001|
|source_template|0.2|True|confidence,entropy,bu|0.006|
|PGD|0.2|True|confidence,entropy,bu,lid|10.557|
|FGSM|0.3|True|confidence,entropy|0.015|
|Gaussian_control|0.3|False||0.001|
|source_template|0.3|True|confidence,entropy,bu|0.005|
|PGD|0.3|True|confidence,entropy,bu|10.457|
|FGSM|0.5|True|confidence,entropy|0.015|
|Gaussian_control|0.5|False||0.001|
|source_template|0.5|True|confidence,entropy,bu|0.005|
|PGD|0.5|True|confidence,entropy,bu|10.246|

|Detector|Weight|Target reached 200/400|Own detector accepted targeted 200/400|Seconds 200/400|
|---|---|---|---|---|
|confidence|0.0|True/True|True/True|10.071/33.780|
|confidence|0.1|True/True|True/True|19.330/65.426|
|confidence|1.0|True/True|True/True|19.584/65.352|
|confidence|10.0|True/True|True/True|19.911/65.504|
|entropy|0.0|True/True|True/True|10.189/33.943|
|entropy|0.1|True/True|True/True|19.611/65.199|
|entropy|1.0|True/True|True/True|19.625/65.156|
|entropy|10.0|True/True|True/True|19.589/65.250|
|bu|0.0|True/True|True/True|10.201/33.971|
|bu|0.1|True/True|True/True|70.225/233.068|
|bu|1.0|True/True|True/True|70.026/233.084|
|bu|10.0|True/True|True/True|69.985/233.358|
|lid|0.0|True/True|True/True|10.197/33.952|
|lid|0.1|True/True|True/True|20.248/67.465|
|lid|1.0|True/True|True/True|20.197/67.494|
|lid|10.0|True/True|True/True|20.203/67.416|
|md|0.0|True/True|False/False|10.224/33.855|
|md|0.1|True/True|True/True|20.082/66.613|
|md|1.0|True/True|True/True|20.019/66.651|
|md|10.0|True/True|True/True|20.035/66.663|
|md_adjusted|0.0|True/True|False/False|10.188/33.852|
|md_adjusted|0.1|True/True|True/True|31.259/104.147|
|md_adjusted|1.0|True/True|True/True|31.254/104.194|
|md_adjusted|10.0|True/True|True/True|31.255/104.227|
|kde|0.0|True/True|False/False|10.185/33.882|
|kde|0.1|True/True|True/True|21.357/70.971|
|kde|1.0|True/True|True/True|21.355/71.146|
|kde|10.0|True/True|True/True|21.358/71.204|

|Detector|Weight|Steps/restarts|Initial/final best objective|
|---|---|---|---|
|confidence|0.0|200/3|8.7595/0|
|confidence|0.1|200/3|8.77406/1.15082e-05|
|confidence|1.0|200/3|8.9051/0.000115082|
|confidence|10.0|200/3|10.2154/0.00115082|
|confidence|0.0|400/5|8.7595/0|
|confidence|0.1|400/5|8.77406/1.15082e-05|
|confidence|1.0|400/5|8.9051/0.000115082|
|confidence|10.0|400/5|10.2154/0.00115082|
|entropy|0.0|200/3|8.7595/0|
|entropy|0.1|200/3|8.78003/9.97112e-05|
|entropy|1.0|200/3|8.96476/0.000997112|
|entropy|10.0|200/3|10.8121/0.00997112|
|entropy|0.0|400/5|8.7595/0|
|entropy|0.1|400/5|8.78003/9.97112e-05|
|entropy|1.0|400/5|8.96476/0.000997112|
|entropy|10.0|400/5|10.8121/0.00997112|
|bu|0.0|200/3|8.7595/0|
|bu|0.1|200/3|8.77536/0.000618257|
|bu|1.0|200/3|8.91809/0.00618257|
|bu|10.0|200/3|10.3454/0.0618257|
|bu|0.0|400/5|8.7595/0|
|bu|0.1|400/5|8.77536/0.000618257|
|bu|1.0|400/5|8.91809/0.00618257|
|bu|10.0|400/5|10.3454/0.0618257|
|lid|0.0|200/3|8.7595/0|
|lid|0.1|200/3|8.76077/0.00561967|
|lid|1.0|200/3|8.77213/0.0429091|
|lid|10.0|200/3|8.88579/0.190769|
|lid|0.0|400/5|8.7595/0|
|lid|0.1|400/5|8.76077/0.00125963|
|lid|1.0|400/5|8.77213/0.00616071|
|lid|10.0|400/5|8.88579/0.0787477|
|md|0.0|200/3|8.7595/0|
|md|0.1|200/3|9.52914/6.57458e-08|
|md|1.0|200/3|16.4558/6.45533e-07|
|md|10.0|200/3|85.7229/6.00435e-06|
|md|0.0|400/5|8.7595/0|
|md|0.1|400/5|9.52914/2.82266e-08|
|md|1.0|400/5|16.4558/4.4332e-07|
|md|10.0|400/5|85.7229/3.34523e-06|
|md_adjusted|0.0|200/3|8.7595/0|
|md_adjusted|0.1|200/3|9.52088/6.69647e-08|
|md_adjusted|1.0|200/3|16.3733/1.31683e-06|
|md_adjusted|10.0|200/3|84.8975/9.02311e-06|
|md_adjusted|0.0|400/5|8.7595/0|
|md_adjusted|0.1|400/5|9.52088/4.66092e-08|
|md_adjusted|1.0|400/5|16.3733/2.4628e-07|
|md_adjusted|10.0|400/5|84.8975/2.85181e-06|
|kde|0.0|200/3|8.7595/0|
|kde|0.1|200/3|8.81326/1.05517e-05|
|kde|1.0|200/3|9.29704/7.35309e-05|
|kde|10.0|200/3|14.1348/0.000564264|
|kde|0.0|400/5|8.7595/0|
|kde|0.1|400/5|8.81326/3.50505e-06|
|kde|1.0|400/5|9.29704/2.09354e-05|
|kde|10.0|400/5|14.1348/0.000233167|

## Model seed 20261015

Zero raw score-gradient methods at the clean probe: []

|Attack|Budget|Target reached|Accepted by methods|Seconds|
|---|---|---|---|---|
|FGSM|0.05|True|confidence,entropy,bu|0.498|
|Gaussian_control|0.05|False||0.003|
|source_template|0.05|True|confidence,entropy,bu,lid|0.006|
|PGD|0.05|True|confidence,entropy,bu,lid|10.024|
|FGSM|0.1|True|confidence,entropy,bu|0.014|
|Gaussian_control|0.1|False||0.001|
|source_template|0.1|True|confidence,entropy,bu|0.006|
|PGD|0.1|True|confidence,entropy,bu,lid|10.275|
|FGSM|0.2|True|confidence,entropy,bu|0.015|
|Gaussian_control|0.2|False||0.001|
|source_template|0.2|True|confidence,entropy,bu|0.005|
|PGD|0.2|True|confidence,entropy,bu,lid|10.543|
|FGSM|0.3|True|confidence,entropy,bu|0.015|
|Gaussian_control|0.3|False||0.001|
|source_template|0.3|True|confidence,entropy,bu|0.005|
|PGD|0.3|True|confidence,entropy,bu|10.508|
|FGSM|0.5|True|confidence,entropy|0.015|
|Gaussian_control|0.5|False||0.001|
|source_template|0.5|True|confidence,entropy,bu|0.005|
|PGD|0.5|True|confidence,entropy|10.241|

|Detector|Weight|Target reached 200/400|Own detector accepted targeted 200/400|Seconds 200/400|
|---|---|---|---|---|
|confidence|0.0|True/True|True/True|10.090/33.825|
|confidence|0.1|True/True|True/True|19.276/65.445|
|confidence|1.0|True/True|True/True|19.595/65.303|
|confidence|10.0|True/True|True/True|19.933/65.584|
|entropy|0.0|True/True|True/True|10.197/33.863|
|entropy|0.1|True/True|True/True|19.606/65.147|
|entropy|1.0|True/True|True/True|19.583/65.202|
|entropy|10.0|True/True|True/True|19.533/65.238|
|bu|0.0|True/True|True/True|10.204/33.829|
|bu|0.1|True/True|True/True|70.189/233.592|
|bu|1.0|True/True|True/True|70.159/233.364|
|bu|10.0|True/True|True/True|70.111/233.525|
|lid|0.0|True/True|True/True|10.183/33.860|
|lid|0.1|True/True|True/True|20.198/67.237|
|lid|1.0|True/True|True/True|20.220/67.182|
|lid|10.0|True/True|True/True|20.176/67.216|
|md|0.0|True/True|False/False|10.215/34.030|
|md|0.1|True/True|True/True|20.056/66.810|
|md|1.0|True/True|True/True|20.031/66.572|
|md|10.0|True/True|True/True|20.067/66.672|
|md_adjusted|0.0|True/True|False/False|10.202/33.891|
|md_adjusted|0.1|True/True|True/True|31.351/104.301|
|md_adjusted|1.0|True/True|True/True|31.299/104.237|
|md_adjusted|10.0|True/True|True/True|31.300/104.072|
|kde|0.0|True/True|False/False|10.157/33.944|
|kde|0.1|True/True|True/True|21.259/71.193|
|kde|1.0|True/True|True/True|21.285/71.240|
|kde|10.0|True/True|True/True|21.312/71.181|

|Detector|Weight|Steps/restarts|Initial/final best objective|
|---|---|---|---|
|confidence|0.0|200/3|7.83268/0|
|confidence|0.1|200/3|7.83268/3.04023e-08|
|confidence|1.0|200/3|7.8327/3.04023e-07|
|confidence|10.0|200/3|7.8329/3.04023e-06|
|confidence|0.0|400/5|7.83268/0|
|confidence|0.1|400/5|7.83268/3.04023e-08|
|confidence|1.0|400/5|7.8327/3.04023e-07|
|confidence|10.0|400/5|7.8329/3.04023e-06|
|entropy|0.0|200/3|7.83268/0|
|entropy|0.1|200/3|7.83327/2.20128e-05|
|entropy|1.0|200/3|7.83857/0.000220128|
|entropy|10.0|200/3|7.89162/0.00220128|
|entropy|0.0|400/5|7.83268/0|
|entropy|0.1|400/5|7.83327/2.20128e-05|
|entropy|1.0|400/5|7.83857/0.000220128|
|entropy|10.0|400/5|7.89162/0.00220128|
|bu|0.0|200/3|7.83268/0|
|bu|0.1|200/3|7.85655/0.00470978|
|bu|1.0|200/3|8.07137/0.0470978|
|bu|10.0|200/3|10.2196/0.470978|
|bu|0.0|400/5|7.83268/0|
|bu|0.1|400/5|7.85655/0.00470978|
|bu|1.0|400/5|8.07137/0.0470978|
|bu|10.0|400/5|10.2196/0.470978|
|lid|0.0|200/3|7.83268/0|
|lid|0.1|200/3|7.83308/1.63374e-06|
|lid|1.0|200/3|7.83672/1.60431e-05|
|lid|10.0|200/3|7.87307/0.000134106|
|lid|0.0|400/5|7.83268/0|
|lid|0.1|400/5|7.83308/1.63374e-06|
|lid|1.0|400/5|7.83672/1.60431e-05|
|lid|10.0|400/5|7.87307/8.39929e-06|
|md|0.0|200/3|7.83268/0|
|md|0.1|200/3|7.86687/1.28113e-07|
|md|1.0|200/3|8.17463/1.32581e-06|
|md|10.0|200/3|11.2522/1.12167e-05|
|md|0.0|400/5|7.83268/0|
|md|0.1|400/5|7.86687/4.11145e-08|
|md|1.0|400/5|8.17463/5.71804e-07|
|md|10.0|400/5|11.2522/5.62943e-06|
|md_adjusted|0.0|200/3|7.83268/0|
|md_adjusted|0.1|200/3|7.86818/9.66078e-08|
|md_adjusted|1.0|200/3|8.18767/1.39392e-06|
|md_adjusted|10.0|200/3|11.3826/6.51243e-06|
|md_adjusted|0.0|400/5|7.83268/0|
|md_adjusted|0.1|400/5|7.86818/3.96874e-08|
|md_adjusted|1.0|400/5|8.18767/6.89363e-07|
|md_adjusted|10.0|400/5|11.3826/6.19761e-06|
|kde|0.0|200/3|7.83268/0|
|kde|0.1|200/3|7.84479/1.57269e-05|
|kde|1.0|200/3|7.95377/0.000116952|
|kde|10.0|200/3|9.04356/0.00144767|
|kde|0.0|400/5|7.83268/0|
|kde|0.1|400/5|7.84479/7.82111e-06|
|kde|1.0|400/5|7.95377/3.9467e-05|
|kde|10.0|400/5|9.04356/0.000556645|


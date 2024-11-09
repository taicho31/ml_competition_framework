# home_credit_2024

https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability


## 実行方法
~~~
sudo chmod 666 /var/run/docker.sock
docker compose up -d --build
~~~

## 構成
~~~
├── Dockerfile
├── docker-compose.yml
├── main
│   ├── data
│   ├── ml_common
│   ├── unique_script
│   ├── lightgbm.ipynb
│   ├── catboost.ipynb
│   └── xgboost.ipynb
~~~

## アクセス
- jupyter notebook: http://localhost:8080/
- MLFlow: http://localhost:5000/
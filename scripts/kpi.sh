main_function() {
python -u train.py kpi anomaly_0 --loader anomaly --batch-size 4 --repr-dims 320 --max-threads 8 --seed 1 --eval --save_ckpt
python -u train.py kpi anomaly_1 --loader anomaly --batch-size 4 --repr-dims 320 --max-threads 8 --seed 2 --eval --save_ckpt
python -u train.py kpi anomaly_2 --loader anomaly --batch-size 4 --repr-dims 320 --max-threads 8 --seed 3 --eval --save_ckpt

python -u train.py kpi anomaly_coldstart_0 --loader anomaly_coldstart --batch-size 4 --repr-dims 320 --max-threads 8 --seed 1 --eval --save_ckpt
python -u train.py kpi anomaly_coldstart_1 --loader anomaly_coldstart --batch-size 4 --repr-dims 320 --max-threads 8 --seed 2 --eval --save_ckpt
python -u train.py kpi anomaly_coldstart_2 --loader anomaly_coldstart --batch-size 4 --repr-dims 320 --max-threads 8 --seed 3 --eval --save_ckpt
}
rm training/kpi_res.txt
main_function 2>&1 | tee -a training/kpi_res.txt